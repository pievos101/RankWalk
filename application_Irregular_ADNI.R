# =========================================================
# LIBRARIES
# =========================================================
library(MASS)
library(fda)
library(aricode)
library(reticulate)
library(aba)
library(survival)

# =========================================================
# LOAD DATA
# =========================================================
df <- as.data.frame(aba::adnimerge)

# =========================================================
# KEEP VARIABLES
# =========================================================
keep_cols <- c(
  "RID", "VISCODE", "YEARS_bl",
  "MMSE", "ADAS13", "CDRSB",
  "ConvertedToDementia"
)

df <- df[, keep_cols]
df <- df[complete.cases(df$RID, df$YEARS_bl), ]

df$RID <- as.numeric(df$RID)
df$time <- as.numeric(df$YEARS_bl)

biomarkers <- c("MMSE","ADAS13", "CDRSB")

for (v in biomarkers) {
  df[[v]] <- suppressWarnings(as.numeric(df[[v]]))
}

# =========================================================
# LOG TRANSFORM
# =========================================================
df$ADAS13 <- log1p(df$ADAS13)
df$CDRSB  <- log1p(df$CDRSB)

# MMSE inversion (optional signal boost)
if ("MMSE" %in% names(df)) {
  df$MMSE <- max(df$MMSE, na.rm = TRUE) - df$MMSE
}

# =========================================================
# FILTER SUBJECTS (>= 5 VISITS)
# =========================================================
visit_counts <- table(df$RID)
keep_ids <- as.numeric(names(visit_counts[visit_counts >= 5]))
df <- df[df$RID %in% keep_ids, ]

# =========================================================
# SUBJECT OUTCOME
# =========================================================
dem_per_subject <- tapply(
  df$ConvertedToDementia,
  df$RID,
  function(x) as.numeric(any(x == 1, na.rm = TRUE))
)

dem_per_subject[is.na(dem_per_subject)] <- 0

true_labels <- data.frame(
  RID = as.numeric(names(dem_per_subject)),
  dem = as.numeric(dem_per_subject)
)

# =========================================================
# BALANCE CLASSES
# =========================================================
n1 <- sum(true_labels$dem == 1)
n0 <- sum(true_labels$dem == 0)
n_min <- min(n1, n0)

#set.seed(1)
ids_1 <- sample(true_labels$RID[true_labels$dem == 1], n_min)
ids_0 <- sample(true_labels$RID[true_labels$dem == 0], n_min)

keep_ids <- c(ids_1, ids_0)

df <- df[df$RID %in% keep_ids, ]
true_labels <- true_labels[true_labels$RID %in% keep_ids, ]

# =========================================================
# SUMMARY
# =========================================================
cat("\n================ DATA SUMMARY ================\n")
cat("Subjects:", length(unique(df$RID)), "\n")
cat("Visits:", nrow(df), "\n")
cat("Dementia prevalence:", mean(true_labels$dem), "\n")
cat("=============================================\n\n")

# =========================================================
# FPCA FEATURES (ONCE ONLY)
# =========================================================
fpca_features <- list()

for (v in biomarkers) {

  tmp <- df[, c("RID", "time", v)]
  tmp <- tmp[complete.cases(tmp), ]

  Ly <- split(tmp[[v]], tmp$RID)
  Lt <- split(tmp$time, tmp$RID)

  fp <- try(FPCA(Ly = Ly, Lt = Lt,
                 optns = list(dataType = "Sparse")),
            silent = TRUE)

  if (inherits(fp, "try-error")) next

  scores <- fp$xiEst
  if (is.null(dim(scores))) scores <- matrix(scores, ncol = 1)

  rownames(scores) <- names(Ly)
  scores[!is.finite(scores)] <- 0

  fpca_features[[v]] <- scores
}

ids <- sort(unique(df$RID))
X_fpca <- NULL

for (v in names(fpca_features)) {

  S <- fpca_features[[v]]

  tmp <- matrix(0, nrow = length(ids), ncol = ncol(S))
  rownames(tmp) <- ids

  common <- intersect(rownames(S), ids)

  tmp[match(common, ids), ] <- S[common, , drop = FALSE]

  X_fpca <- if (is.null(X_fpca)) tmp else cbind(X_fpca, tmp)
}

X_fpca <- scale(X_fpca)

# =========================================================
# PYTHON GNN
# =========================================================
Sys.setenv(RETICULATE_PYTHON = path.expand("~/rankwalk-venv/bin/python"))
use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)

Longdat_list <- list()

for (v in biomarkers) {

  tmp <- df[, c("RID", "time", v)]
  tmp <- tmp[complete.cases(tmp), ]

  Longdat_list[[v]] <- data.frame(
    subject = tmp$RID,
    time = tmp$time,
    outcome = v,
    y = tmp[[v]]
  )
}

Longdat2 <- do.call(rbind, Longdat_list)
Longdat2 <- Longdat2[complete.cases(Longdat2), ]

py_run_string("
import numpy as np
import pandas as pd
import torch
from rankwalk import build_temporal_graph_grid, train_gnn, compute_jaccard_fast

def run_rankwalk_gnn(df, epochs=120):

    df = pd.DataFrame(df)

    G, _ = build_temporal_graph_grid(
        df,
        k_similarity=5,
        n_bins=5,
        overlap=0.5
    )

    nodes = list(G.nodes())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x, t = [], []

    for n in nodes:
        x.append(G.nodes[n]['features'])
        t.append(G.nodes[n]['time'])

    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    t = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(1)

    t = (t - t.mean()) / (t.std() + 1e-8)
    #x = torch.cat([x, t], dim=1)

    edges, et = [], []
    for u, v, a in G.edges(data=True):
        edges.append([u, v])
        et.append(a['edge_type'])
        edges.append([v, u])
        et.append(a['edge_type'])

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    edge_type = torch.tensor(et, dtype=torch.long, device=device)

    J = compute_jaccard_fast(edge_index, G.number_of_nodes(), device=device)

    emb = train_gnn(
        x, edge_index, edge_type, J,
        epochs=epochs,
        lr=1e-3,
        walk_length=20,
        top_k=10,
        device=device
    )

    return emb.detach().cpu().numpy(), np.array([G.nodes[n]['subject'] for n in nodes])
")

run_gnn <- function() {
  res <- py$run_rankwalk_gnn(Longdat2, 100L)
  list(emb = res[[1]], sub = as.numeric(res[[2]]))
}

# =========================================================
# 30 RUN EXPERIMENT
# =========================================================
n_runs <- 30

results <- matrix(NA, nrow = n_runs, ncol = 6)
colnames(results) <- c(
  "FPCA_ARI", "GNN_ARI",
  "FPCA_C", "GNN_C",
  "FPCA_LR", "GNN_LR"
)

for (r in 1:n_runs) {

  cat("\n================ RUN", r, "================\n")

  # -------------------------
  # FPCA clustering
  # -------------------------
  #set.seed(r)
  cl_fpca <- kmeans(X_fpca, centers = 2, nstart = 50)$cluster

  fpca_df <- data.frame(
    RID = as.numeric(rownames(X_fpca)),
    cluster = cl_fpca
  )

  fpca_merge <- merge(true_labels, fpca_df, by = "RID")

  ARI_fpca <- ARI(fpca_merge$dem, fpca_merge$cluster)

  surv_fpca <- merge(
    aggregate(time ~ RID, df, max),
    true_labels,
    by = "RID"
  )

  surv_fpca <- merge(surv_fpca, fpca_df, by = "RID")

  lr_fpca <- survdiff(Surv(time, dem) ~ cluster, data = surv_fpca)
  lr_fpca_stat <- sum((lr_fpca$obs - lr_fpca$exp)^2 / lr_fpca$exp)

  ci_fpca <- concordance(Surv(time, dem) ~ cluster,
                         data = surv_fpca)$concordance

  # -------------------------
  # GNN
  # -------------------------
  #set.seed(r)

 gnn_out <- run_gnn()

  emb <- gnn_out$emb
  sub <- gnn_out$sub

  subjects <- sort(unique(sub))

  feat <- do.call(
    rbind,
    lapply(subjects, function(s) {

      idx <- which(sub == s)

      E <- emb[idx, , drop = FALSE]

      mu <- colMeans(E)

      if (nrow(E) > 1) {
        sdv <- apply(E, 2, sd)
      } else {
        sdv <- rep(0, ncol(E))
      }

      c(mu, sdv)
    })
  )

  rownames(feat) <- subjects

  feat[!is.finite(feat)] <- 0

  feat <- scale(feat)

  cl_gnn <- kmeans(
    feat,
    centers = 2,
    nstart = 50
  )$cluster

  gnn_df <- data.frame(
    RID = as.numeric(subjects),
    cluster = cl_gnn
  )

  gnn_merge <- merge(
    true_labels,
    gnn_df,
    by = "RID"
  )

  ARI_gnn <- ARI(
    gnn_merge$dem,
    gnn_merge$cluster
  )

  surv_gnn <- merge(
    aggregate(time ~ RID, df, max),
    true_labels,
    by = "RID"
  )

  surv_gnn <- merge(surv_gnn, gnn_df, by = "RID")

  lr_gnn <- survdiff(Surv(time, dem) ~ cluster, data = surv_gnn)
  lr_gnn_stat <- sum((lr_gnn$obs - lr_gnn$exp)^2 / lr_gnn$exp)

  ci_gnn <- concordance(Surv(time, dem) ~ cluster,
                        data = surv_gnn)$concordance

  # -------------------------
  # STORE
  # -------------------------
  results[r, ] <- c(
    ARI_fpca,
    ARI_gnn,
    ci_fpca,
    ci_gnn,
    lr_fpca_stat,
    lr_gnn_stat
  )

  cat("FPCA ARI:", ARI_fpca, "\n")
  cat("GNN ARI:", ARI_gnn, "\n")
  print(results)
}

# =========================================================
# FINAL SUMMARY
# =========================================================
cat("\n================ FINAL (30 RUNS) ================\n")
print(colMeans(results, na.rm = TRUE))
print(apply(results, 2, sd, na.rm = TRUE))
cat("===============================================\n")