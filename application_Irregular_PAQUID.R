# =========================================================
# LIBRARIES
# =========================================================
library(MASS)
library(aricode)
library(fda)
library(MASS)
library(fda)
library(survival)
library(survcomp)

# =========================================================
# LOAD DATA
# =========================================================
load("paquid.rda")

paquid <- as.data.frame(paquid)

# =========================================================
# KEEP ONLY PATIENTS WITH > 2 VISITS
# =========================================================
visit_counts <- table(paquid$ID)

keep_ids <- names(visit_counts[visit_counts > 3])

paquid <- paquid[paquid$ID %in% keep_ids, ]

# =========================================================
# VARIABLES
# =========================================================
vars <- c("MMSE", "BVRT", "IST", "CESD")

# =========================================================
# SAFE NUMERIC CONVERSION (CRITICAL FIX)
# =========================================================
for (v in vars) {
  paquid[[v]] <- suppressWarnings(as.numeric(as.character(paquid[[v]])))
}

paquid$age <- as.numeric(paquid$age)
paquid$ID  <- as.numeric(paquid$ID)
paquid$dem <- as.numeric(paquid$dem)

# =========================================================
# USE AGE AS TIME
# =========================================================
paquid <- paquid[order(paquid$ID, paquid$age), ]
paquid$time <- paquid$age

# =========================================================
# LOG TRANSFORM (stabilises FPCA)
# =========================================================
for (v in vars) {
  paquid[[v]] <- log1p(paquid[[v]])
}

# =========================================================
# SAFE FILTERING (NO is.finite ON DATA.FRAMES!)
# =========================================================
Ymat <- data.matrix(paquid[, vars])

keep <- rowSums(is.finite(Ymat)) > 0

paquid_clean <- paquid[keep, ]
paquid_clean[, vars] <- Ymat[keep, ]

# remove invalid time
paquid_clean <- paquid_clean[is.finite(paquid_clean$time), ]

# =========================================================
# SUBJECT-LEVEL SURVIVAL DATA
# =========================================================

surv_df <- do.call(
  rbind,
  lapply(split(paquid_clean, paquid_clean$ID), function(df) {

    df <- df[order(df$age), ]

    event <- any(df$dem == 1, na.rm = TRUE)

    if (event) {
      event_time <- min(df$age[df$dem == 1], na.rm = TRUE)
    } else {
      event_time <- max(df$age, na.rm = TRUE)
    }

    baseline_age <- min(df$age, na.rm = TRUE)

    data.frame(
      ID = unique(df$ID),
      time = event_time - baseline_age,
      event = as.numeric(event)
    )
  })
)

surv_df <- surv_df[is.finite(surv_df$time), ]


# =========================================================
# GROUND TRUTH (DEMENTIA STATUS)
# =========================================================
ids <- unique(paquid_clean$ID)

true_labels <- data.frame(
  ID = ids,
  dem = sapply(ids, function(i) {
    as.numeric(any(paquid_clean$dem[paquid_clean$ID == i] == 1, na.rm = TRUE))
  })
)

# =========================================================
# FPCA FEATURES
# =========================================================
fpca_features <- list()

for (v in vars) {

  tmp <- paquid_clean[
    is.finite(paquid_clean[[v]]) &
      is.finite(paquid_clean$time),
    c("ID", "time", v)
  ]

  colnames(tmp)[3] <- "y"

  Ly <- split(tmp$y, tmp$ID)
  Lt <- split(tmp$time, tmp$ID)

  fp <- try(
    FPCA(
      Ly = Ly,
      Lt = Lt,
      optns = list(dataType = "Sparse")
    ),
    silent = TRUE
  )

  if (inherits(fp, "try-error")) next

  scores <- fp$xiEst
  if (is.null(dim(scores))) scores <- matrix(scores, ncol = 1)

  rownames(scores) <- names(Ly)

  scores[!is.finite(scores)] <- 0

  fpca_features[[v]] <- scores
}

# =========================================================
# MERGE FPCA FEATURES
# =========================================================
ids_all <- sort(unique(paquid_clean$ID))

X_fpca <- NULL

for (v in names(fpca_features)) {

  S <- fpca_features[[v]]

  tmp <- matrix(0, nrow = length(ids_all), ncol = ncol(S))
  rownames(tmp) <- ids_all

  common <- intersect(rownames(S), ids_all)

  tmp[match(common, ids_all), ] <- S[common, , drop = FALSE]

  X_fpca <- if (is.null(X_fpca)) tmp else cbind(X_fpca, tmp)
}

X_fpca[!is.finite(X_fpca)] <- 0
X_fpca <- scale(X_fpca)

# =========================================================
# CLUSTERING (K = 2)
# =========================================================
#set.seed(1)

cl_fpca <- kmeans(X_fpca, centers = 3, nstart = 50)$cluster

fpca_df <- data.frame(
  ID = as.numeric(rownames(X_fpca)),
  cluster = cl_fpca
)

# =========================================================
# FPCA SURVIVAL EVALUATION
# =========================================================

fpca_eval <- merge(fpca_df, surv_df, by = "ID")

fpca_eval$cluster <- factor(fpca_eval$cluster)

# --------------------------
# LOG-RANK TEST
# --------------------------

lr_fpca <- survdiff(
  Surv(time, event) ~ cluster,
  data = fpca_eval
)

logrank_fpca <- 1 - pchisq(
  lr_fpca$chisq,
  df = length(lr_fpca$n) - 1
)

# --------------------------
# COX MODEL
# --------------------------

cox_fpca <- coxph(
  Surv(time, event) ~ cluster,
  data = fpca_eval
)

#risk_fpca <- predict(cox_fpca, type = "lp")

# --------------------------
# C-INDEX
# --------------------------

#cindex_fpca <- concordance.index(
#  x = risk_fpca,
#  surv.time = fpca_eval$time,
#  surv.event = fpca_eval$event
#)$c.index

cindex_fpca = cox_fpca$concordance[6]

cat("\n====================\n")
cat("FPCA C-index :", round(cindex_fpca, 3), "\n")
cat("FPCA LogRank :", signif(logrank_fpca, 4), "\n")
cat("====================\n")

# =========================================================
# DONE
# =========================================================
# =========================================================
# PYTHON: GNN (RankWalk)
# =========================================================
Sys.setenv(RETICULATE_PYTHON = path.expand("~/rankwalk-venv/bin/python"))
use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)

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
        overlap=0.7
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

    return {
        'embeddings': emb.detach().cpu().numpy(),
        'subjects': np.array([G.nodes[n]['subject'] for n in nodes])
    }
")

# =========================================================
# LONG FORMAT FOR GNN
# =========================================================
Longdat_list <- list()

for (v in vars) {

  tmp <- paquid_clean[
    is.finite(paquid_clean[[v]]) &
      is.finite(paquid_clean$time),
    c("ID", "time", v)
  ]

  colnames(tmp)[3] <- "y"

  Longdat_list[[v]] <- data.frame(
    subject = tmp$ID,
    time = tmp$time,
    outcome = v,
    y = tmp$y
  )
}

Longdat2 <- do.call(rbind, Longdat_list)

Longdat2 <- Longdat2[is.finite(Longdat2$y), ]

# =========================================================
# RUN GNN
# =========================================================
res <- py$run_rankwalk_gnn(Longdat2, 100L)

emb <- res$embeddings
sub <- as.numeric(res$subjects)

subjects <- sort(unique(sub))

# =========================================================
# FLATTEN NODE EMBEDDINGS PER SUBJECT
# =========================================================

feat <- lapply(subjects, function(g) {

  idx <- which(sub == g)

  E <- emb[idx, , drop = FALSE]

  as.numeric(t(E))
})

# ensure equal length vectors
max_len <- max(sapply(feat, length))

feat <- t(sapply(feat, function(x) {
  c(x, rep(0, max_len - length(x)))
}))

rownames(feat) <- subjects

feat[!is.finite(feat)] <- 0

feat <- scale(feat)

# =========================================================
# GNN CLUSTERING
# =========================================================
cl_gnn <- kmeans(feat, centers = 3, nstart = 50)$cluster

gnn_df <- data.frame(
  ID = as.numeric(names(cl_gnn)),
  cluster = cl_gnn
)

# =========================================================
# GNN SURVIVAL EVALUATION
# =========================================================

gnn_eval <- merge(gnn_df, surv_df, by = "ID")

gnn_eval$cluster <- factor(gnn_eval$cluster)

# --------------------------
# LOG-RANK TEST
# --------------------------

lr_gnn <- survdiff(
  Surv(time, event) ~ cluster,
  data = gnn_eval
)

logrank_gnn <- 1 - pchisq(
  lr_gnn$chisq,
  df = length(lr_gnn$n) - 1
)

# --------------------------
# COX MODEL
# --------------------------

cox_gnn <- coxph(
  Surv(time, event) ~ cluster,
  data = gnn_eval
)

#risk_gnn <- predict(cox_gnn, type = "lp")

# --------------------------
# C-INDEX
# --------------------------

#cindex_gnn <- concordance.index(
#  x = risk_gnn,
#  surv.time = gnn_eval$time,
#  surv.event = gnn_eval$event
#)$c.index

cindex_gnn = cox_gnn$concordance[6]

cat("GNN C-index :", round(cindex_gnn, 3), "\n")
cat("GNN LogRank :", signif(logrank_gnn, 4), "\n")

cat("\n====================\n")
cat("FPCA RESULTS\n")
cat("  C-index :", round(cindex_fpca, 3), "\n")
cat("  LogRank :", signif(logrank_fpca, 4), "\n\n")

cat("GNN RESULTS\n")
cat("  C-index :", round(cindex_gnn, 3), "\n")
cat("  LogRank :", signif(logrank_gnn, 4), "\n")
cat("====================\n")