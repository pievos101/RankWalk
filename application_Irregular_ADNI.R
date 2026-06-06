# =========================================================
# LIBRARIES
# =========================================================
library(MASS)
library(fda)
library(aricode)
library(reticulate)
library(aba)
library(survival)
library(fdapace)

# =========================================================
# LOAD DATA
# =========================================================
df <- as.data.frame(aba::adnimerge)

keep_cols <- c(
  "RID", "VISCODE", "YEARS_bl",
  "MMSE", "ADAS13", "CDRSB",
  "ConvertedToDementia"
)

df <- df[, keep_cols]
df <- df[complete.cases(df$RID, df$YEARS_bl), ]

df$RID <- as.numeric(df$RID)
df$time <- as.numeric(df$YEARS_bl)

biomarkers <- c("MMSE","ADAS13","CDRSB")

for (v in biomarkers) {
  df[[v]] <- suppressWarnings(as.numeric(df[[v]]))
}

# =========================================================
# TRANSFORMS
# =========================================================
df$ADAS13 <- log1p(df$ADAS13)
df$CDRSB  <- log1p(df$CDRSB)

df$MMSE <- max(df$MMSE, na.rm=TRUE) - df$MMSE

# =========================================================
# FILTER SUBJECTS
# =========================================================
visit_counts <- table(df$RID)
keep_ids <- as.numeric(names(visit_counts[visit_counts >= 5]))
df <- df[df$RID %in% keep_ids, ]

# =========================================================
# SUBJECT LABELS + SURVIVAL TIME
# =========================================================
dem_per_subject <- tapply(
  df$ConvertedToDementia,
  df$RID,
  function(x) as.numeric(any(x == 1, na.rm=TRUE))
)

dem_per_subject[is.na(dem_per_subject)] <- 0

true_labels <- data.frame(
  RID = as.numeric(names(dem_per_subject)),
  dem = as.numeric(dem_per_subject)
)

subject_time <- aggregate(time ~ RID, df, max)

# =========================================================
# BALANCE CLASSES
# =========================================================
set.seed(1)

n1 <- sum(true_labels$dem == 1)
n0 <- sum(true_labels$dem == 0)
n_min <- min(n1, n0)

ids_1 <- sample(true_labels$RID[true_labels$dem==1], n_min)
ids_0 <- sample(true_labels$RID[true_labels$dem==0], n_min)

keep_ids <- c(ids_1, ids_0)

df <- df[df$RID %in% keep_ids, ]
true_labels <- true_labels[true_labels$RID %in% keep_ids, ]
subject_time <- subject_time[subject_time$RID %in% keep_ids, ]

# =========================================================
# FPCA
# =========================================================
fpca_features <- list()

for (v in biomarkers) {

  tmp <- df[, c("RID","time",v)]
  tmp <- tmp[complete.cases(tmp), ]

  if (length(unique(tmp$RID)) < 10) next

  Ly <- split(tmp[[v]], tmp$RID)
  Lt <- split(tmp$time, tmp$RID)

  fp <- try(
    FPCA(Ly=Ly, Lt=Lt,
         optns=list(dataType="Sparse")),
    silent=TRUE
  )

  if (inherits(fp,"try-error")) next

  S <- fp$xiEst
  if (is.null(dim(S))) S <- matrix(S, ncol=1)

  rownames(S) <- names(Ly)
  S[!is.finite(S)] <- 0

  fpca_features[[v]] <- S
}

ids <- sort(unique(df$RID))
X_fpca <- NULL

for (v in names(fpca_features)) {

  S <- fpca_features[[v]]

  tmp <- matrix(0, nrow=length(ids), ncol=ncol(S))
  rownames(tmp) <- ids

  common <- intersect(rownames(S), ids)
  tmp[match(common,ids),] <- S[common,,drop=FALSE]

  X_fpca <- if (is.null(X_fpca)) tmp else cbind(X_fpca,tmp)
}

X_fpca <- scale(X_fpca)

# =========================================================
# PYTHON: RANKWALK GNN
# =========================================================
Sys.setenv(RETICULATE_PYTHON = path.expand("~/rankwalk-venv/bin/python"))
use_python(Sys.getenv("RETICULATE_PYTHON"), required=TRUE)

py_run_string("
import numpy as np
import pandas as pd
import torch
from rankwalk import build_temporal_graph_grid, train_gnn, compute_jaccard_fast

def run_rankwalk_gnn(df, epochs=120):

    df = pd.DataFrame(df)

    G,_ = build_temporal_graph_grid(
        df,
        k_similarity=5,
        n_bins=5,
        overlap=0.5
    )

    nodes = list(G.nodes())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = []
    t = []

    for n in nodes:
        x.append(G.nodes[n]['features'])
        t.append(G.nodes[n]['time'])

    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    t = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(1)
    t = (t - t.mean())/(t.std()+1e-8)

    x = torch.cat([x,t],dim=1)

    edges=[]
    et=[]

    for u,v,a in G.edges(data=True):
        edges.append([u,v]); et.append(a.get('edge_type',0))
        edges.append([v,u]); et.append(a.get('edge_type',0))

    edge_index = torch.tensor(edges,dtype=torch.long,device=device).t().contiguous()
    edge_type = torch.tensor(et,dtype=torch.long,device=device)

    J = compute_jaccard_fast(edge_index, G.number_of_nodes(), device=device)
    if J is None:
        J = torch.eye(G.number_of_nodes(), device=device)

    emb = train_gnn(
        x, edge_index, edge_type, J,
        epochs=epochs,
        lr=1e-3,
        walk_length=20,
        top_k=10,
        device=device
    )

    return {
        'emb': emb.detach().cpu().numpy(),
        'sub': np.array([G.nodes[n]['subject'] for n in nodes])
    }
")

run_gnn <- function() {
  res <- py$run_rankwalk_gnn(Longdat2, 100L)
  list(emb=res$emb, sub=as.numeric(res$sub))
}

# =========================================================
# LONG FORMAT
# =========================================================
Longdat_list <- list()

for (v in biomarkers) {

  tmp <- df[, c("RID","time",v)]
  tmp <- tmp[complete.cases(tmp), ]

  Longdat_list[[v]] <- data.frame(
    subject=tmp$RID,
    time=tmp$time,
    outcome=v,
    y=tmp[[v]]
  )
}

Longdat2 <- do.call(rbind, Longdat_list)
Longdat2 <- Longdat2[complete.cases(Longdat2), ]

# =========================================================
# METRICS HELPERS
# =========================================================
get_metrics <- function(cluster, true_labels, subject_time) {

  dfm <- merge(true_labels, subject_time, by="RID")
  dfm$cluster <- cluster[match(dfm$RID, names(cluster))]

  # ARI
  ari <- ARI(dfm$dem, dfm$cluster)

  # C-index
  cidx <- concordance(Surv(time, dem) ~ cluster, data=dfm)$concordance

  # log-rank statistic (NO p-value)
  lr <- survdiff(Surv(time, dem) ~ cluster, data=dfm)
  lr_stat <- sum((lr$obs - lr$exp)^2 / lr$exp)

  list(ARI=ari, C=cidx, LR=lr_stat)
}

# =========================================================
# 30 RUNS
# =========================================================
n_runs <- 30
results <- matrix(NA, n_runs, 6)

colnames(results) <- c(
  "FPCA_ARI","GNN_ARI",
  "FPCA_C","GNN_C",
  "FPCA_LR","GNN_LR"
)

for (r in 1:n_runs) {

  cat("\nRUN", r, "\n")

  # ---------------- FPCA ----------------
  cl_fpca <- kmeans(X_fpca, 2, nstart=50)$cluster
  names(cl_fpca) <- as.character(ids)

  m_fpca <- get_metrics(cl_fpca, true_labels, subject_time)

  # ---------------- GNN ----------------
  gnn_out <- run_gnn()

  # =========================================================
  # SAFE GNN SUBJECT POOLING (FIXED)
  # =========================================================

  emb <- gnn_out$emb
  sub <- gnn_out$sub

  if (is.null(emb) || length(emb) == 0) {
    stop("GNN returned empty embeddings")
  }

  subjects <- unique(sub)

  feat_list <- list()

  for (s in subjects) {

    idx <- which(sub == s)

    if (length(idx) == 0) next

    E <- emb[idx, , drop = FALSE]

    if (nrow(E) == 0) next

    mu <- colMeans(E, na.rm = TRUE)

    if (nrow(E) > 1) {
      sdv <- apply(E, 2, sd, na.rm = TRUE)
    } else {
      sdv <- rep(0, ncol(E))
    }

    feat_list[[as.character(s)]] <- c(mu, sdv)
  }

  feat <- do.call(rbind, feat_list)

  # safety checks
  if (is.null(feat) || nrow(feat) < 2) {
    stop("Not enough valid subjects after pooling GNN embeddings")
  }

  feat[!is.finite(feat)] <- 0
  feat <- scale(feat)

  if (any(is.na(feat))) {
    stop("NA values in feat after scaling")
  }

  cl_gnn <- kmeans(feat, centers = 2, nstart = 50)$cluster
  names(cl_gnn) <- rownames(feat)

  m_gnn <- get_metrics(cl_gnn, true_labels, subject_time)

  results[r,] <- c(
    m_fpca$ARI, m_gnn$ARI,
    m_fpca$C, m_gnn$C,
    m_fpca$LR, m_gnn$LR
  )
  print(results)
  cat("FPCA ARI:", m_fpca$ARI, " GNN ARI:", m_gnn$ARI, "\n")
}

# =========================================================
# FINAL
# =========================================================
cat("\n===== FINAL =====\n")
print(colMeans(results, na.rm=TRUE))
print(apply(results,2,sd,na.rm=TRUE))