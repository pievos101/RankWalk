
# =========================================================
# LIBRARIES
# =========================================================
library(MASS)
library(survival)
library(Hmisc)
library(reshape2)
library(ggplot2)
library(fda)
library(reticulate)
library(fdapace)

# =========================================================
# PYTHON (GNN)
# =========================================================
Sys.setenv(RETICULATE_PYTHON = path.expand("~/rankwalk-venv/bin/python"))
use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)

py_run_string("
import numpy as np
import pandas as pd
import torch

from rankwalk import build_temporal_graph_grid, compute_jaccard_fast, train_gnn

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
        edges.append([u, v]); et.append(a['edge_type'])
        edges.append([v, u]); et.append(a['edge_type'])

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

    return {
        'embeddings': emb.detach().cpu().numpy(),
        'subjects': np.array([G.nodes[n]['subject'] for n in nodes])
    }
")

# =========================================================
# DATA
# =========================================================
load("pbc2.RData")

feature_cols <- c("albumin", "alkaline", "serBilir", "SGOT")

# =========================================================
# SAFE NUMERIC CLEANING
# =========================================================
to_numeric_safe <- function(x) {
  if (is.numeric(x)) return(x)
  x <- gsub(",", ".", x)
  x <- gsub("[^0-9\\.\\-]", "", x)
  x <- suppressWarnings(as.numeric(x))
  x[!is.finite(x)] <- NA
  x
}

pbc2_clean <- pbc2

for (v in feature_cols) {
  pbc2_clean[[v]] <- log1p(to_numeric_safe(pbc2_clean[[v]]))
}

pbc2_clean <- pbc2_clean[is.finite(pbc2_clean$time), ]

# =========================================================
# SURVIVAL DATA (CLEAN + CONSISTENT)
# =========================================================
surv_data <- unique(pbc2[, c("id", "years", "event")])
surv_data$id <- as.numeric(surv_data$id)
surv_data$time <- surv_data$years
surv_data$event <- as.numeric(surv_data$event == 2)

# =========================================================
# SAFE METRICS
# =========================================================
safe_cindex <- function(score, surv) {

  if (length(unique(score)) < 2) return(NA)

  S <- Surv(surv$time, surv$event)

  out <- try(rcorr.cens(score, S), silent = TRUE)

  if (inherits(out, "try-error")) return(NA)

  as.numeric(out["C Index"])
}

safe_logrank <- function(cluster, surv) {

  df <- data.frame(
    time = surv$time,
    event = surv$event,
    cluster = as.factor(cluster)
  )

  df <- df[complete.cases(df), ]

  if (length(unique(df$cluster)) < 2) return(NA)
  if (table(df$cluster)[1] < 5 || table(df$cluster)[2] < 5) return(NA)

  S <- Surv(df$time, df$event)

  tryCatch(
    survdiff(S ~ cluster, data = df)$chisq,
    error = function(e) NA
  )
}

# =========================================================
# ALIGNMENT (SAFE + NON-COLLAPSING)
# =========================================================
align_clusters <- function(cluster, merged) {

  cluster <- as.numeric(cluster)

  risk <- tapply(merged$time, cluster, mean, na.rm = TRUE)

  if (any(is.na(risk))) {
    risk[is.na(risk)] <- max(risk, na.rm = TRUE) + 1
  }

  ord <- order(risk)

  map <- setNames(seq_along(ord), ord)

  aligned <- map[as.character(cluster)]

  as.numeric(aligned)
}

# =========================================================
# LOOP
# =========================================================
n_iter <- 20
RES <- matrix(NA, n_iter, 4)
colnames(RES) <- c("FPCA_C", "FPCA_LR", "GNN_C", "GNN_LR")

for (ii in 1:n_iter) {

  # =========================
  # FPCA
  # =========================
  fpca_features <- list()

  for (v in feature_cols) {

    tmp <- data.frame(
      id = pbc2_clean$id,
      time = pbc2_clean$time,
      y = pbc2_clean[[v]]
    )

    tmp <- tmp[is.finite(tmp$y) & is.finite(tmp$time), ]

    Ly <- split(tmp$y, tmp$id)
    Lt <- split(tmp$time, tmp$id)

    fp <- try(
      FPCA(Ly = Ly, Lt = Lt,
           optns = list(dataType = "Sparse")),
      silent = TRUE
    )

    if (inherits(fp, "try-error")) next

    scores <- fp$xiEst

    if (is.null(dim(scores))) scores <- matrix(scores, ncol = 1)

    rownames(scores) <- names(Ly)

    # REMOVE EMPTY / NA ROWS (CRITICAL FIX)
    scores <- scores[apply(scores, 1, function(x) any(is.finite(x))), , drop = FALSE]

    fpca_features[[v]] <- scores
  }

  ids <- sort(unique(pbc2_clean$id))
  X_fpca <- NULL

  for (v in names(fpca_features)) {

    S <- fpca_features[[v]]

    tmp <- matrix(0, nrow = length(ids), ncol = ncol(S))
    rownames(tmp) <- ids

    common <- intersect(rownames(S), ids)

    tmp[match(common, ids), ] <- S[common, , drop = FALSE]

    X_fpca <- if (is.null(X_fpca)) tmp else cbind(X_fpca, tmp)
  }

  X_fpca[!is.finite(X_fpca)] <- 0
  X_fpca <- scale(X_fpca)

  # DROP BAD ROWS (CRITICAL)
  keep <- apply(X_fpca, 1, function(x) all(is.finite(x)))
  X_fpca <- X_fpca[keep, , drop = FALSE]

  ids_fpca <- as.numeric(rownames(X_fpca))
  surv_fpca <- surv_data[surv_data$id %in% ids_fpca, ]

  # =========================
  # CLUSTERING
  # =========================
  cl_fpca <- kmeans(X_fpca, centers = 3, nstart = 100)$cluster
  names(cl_fpca) <- rownames(X_fpca)

  merged_fpca <- merge(surv_fpca,
                       data.frame(id = as.numeric(names(cl_fpca)),
                                  cluster = cl_fpca),
                       by = "id")

  if (length(unique(merged_fpca$cluster)) < 2) next

  cl_fpca_aligned <- align_clusters(merged_fpca$cluster, merged_fpca)

  cat("FPCA Samples:",length(cl_fpca_aligned),"\n")

  # =========================
  # METRICS
  # =========================
  c_fpca <- safe_cindex(cl_fpca_aligned, merged_fpca)
  lr_fpca <- safe_logrank(cl_fpca_aligned, merged_fpca)

  cat("FPCA:", c_fpca, lr_fpca, "\n")

  # =========================
  # GNN
  # =========================
  Longdat_list <- list()

  for (i in seq_along(feature_cols)) {

    v <- feature_cols[i]

    tmp <- data.frame(
      subject = pbc2_clean$id,
      time = pbc2_clean$time,
      outcome = i,
      y = pbc2_clean[[v]]
    )

    tmp <- tmp[is.finite(tmp$y) & is.finite(tmp$time), ]
    Longdat_list[[i]] <- tmp
  }

  Longdat2 <- do.call(rbind, Longdat_list)

  res <- py$run_rankwalk_gnn(Longdat2, 100L)

  emb <- res$embeddings
  sub <- as.numeric(res$subjects)

  subjects <- sort(unique(sub))

  feat <- lapply(subjects, function(g) {
    idx <- which(sub == g)
    colMeans(emb[idx, , drop = FALSE])
  })

  feat <- do.call(rbind, feat)
  rownames(feat) <- subjects
  feat <- scale(feat)

  keep <- apply(feat, 1, function(x) all(is.finite(x)))
  feat <- feat[keep, , drop = FALSE]

  cl_gnn <- kmeans(feat, centers = 3, nstart = 100)$cluster
  names(cl_gnn) <- rownames(feat)

  merged_gnn <- merge(surv_data,
                      data.frame(id = as.numeric(names(cl_gnn)),
                                 cluster = cl_gnn),
                      by = "id")

  if (length(unique(merged_gnn$cluster)) < 2) next

  cl_gnn_aligned <- align_clusters(merged_gnn$cluster, merged_gnn)
  cat("RankWalk Samples:", length(cl_gnn_aligned),"\n")

  c_gnn <- safe_cindex(cl_gnn_aligned, merged_gnn)
  lr_gnn <- safe_logrank(cl_gnn_aligned, merged_gnn)

  cat("GNN:", c_gnn, lr_gnn, "\n")

  RES[ii, ] <- c(c_fpca, lr_fpca, c_gnn, lr_gnn)
  print(RES)
}

# =========================================================
# PLOT
# =========================================================
df <- melt(as.data.frame(RES))
colnames(df) <- c("Method", "Value")

ggplot(df, aes(Method, Value, fill = Method)) +
  geom_boxplot() +
  theme_minimal()