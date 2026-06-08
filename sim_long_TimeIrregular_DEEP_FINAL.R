# =========================================================
# PYTHON SETUP (MUST BE FIRST)
# =========================================================
Sys.setenv(RETICULATE_PYTHON = path.expand("~/rankwalk-venv/bin/python"))

library(reticulate)

use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
py_config()

# =========================================================
# R LIBRARIES
# =========================================================
library(TAPIO)
library(clusterMLD)
library(MASS)
library(aricode)
library(reshape)
library(ggplot2)
library(reshape2)
library(kml3d)
library(fda)
library(dtwclust)

# =========================================================
# PYTHON: RankWalk + TS2Vec OFFICIAL
# =========================================================
py_run_string("
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rankwalk import (
    build_temporal_graph_aligned,
    train_gnn,
    compute_jaccard_fast,
    build_temporal_graph,
    build_temporal_graph_grid,
    build_temporal_graph_grid2
)

# =========================================================
# RankWalk GNN (UNCHANGED)
# =========================================================
def run_rankwalk_gnn(df, epochs=300, lr=1e-3, top_k=10, walk_length=20):

    df = pd.DataFrame(df)

    #G, _ = build_temporal_graph_aligned(
    #    df,
    #    k_similarity=10,
    #    k_align=10,
    #    knn_mode='knn'
    #)

    G, labels_df = build_temporal_graph_grid(
        df,
        k_similarity=5,
        n_bins=5,
        overlap=0.5  
    )

    print(len(G.nodes))
    print(len(G.edges))

    node_list = list(G.nodes())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_list, t_list = [], []

    for n in node_list:
        x_list.append(G.nodes[n]['features'])
        t_list.append(G.nodes[n]['time'])

    x = torch.tensor(np.array(x_list), dtype=torch.float32, device=device)

    t = torch.tensor(t_list, dtype=torch.float32, device=device).unsqueeze(1)
    t = (t - t.mean()) / (t.std() + 1e-8)

    # thats not good for irregular time
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
        lr=lr,
        walk_length=walk_length,
        top_k=top_k,
        device=device
    )

    return {
        'embeddings': emb.detach().cpu().numpy(),
        'subjects': np.array([G.nodes[n]['subject'] for n in node_list])
    }

# =========================================================
# TS2VEC OFFICIAL
# =========================================================
import sys
#sys.path.append('/home/bpfeif/GitHub/ts2vec')
sys.path.append('/home/bastian/GitHub/ts2vec')

from ts2vec import TS2Vec

def run_ts2vec_official(df, epochs=100):

    df = pd.DataFrame(df)
    df = df.sort_values(['subject', 'time'])

    subjects = np.unique(df['subject'])

    feat_cols = [c for c in df.columns if c not in ['subject','time','cluster']]

    seqs = []

    for s in subjects:
        seq = df[df['subject'] == s][feat_cols].values.astype(np.float32)
        seqs.append(seq)

    X = np.stack(seqs)

    print('TS2Vec official training start')

    device = 0 if torch.cuda.is_available() else 'cpu'

    model = TS2Vec(
        input_dims=X.shape[2],
        device=device,
        output_dims=64
    )

    model.fit(
        X,
        verbose=True,
        n_epochs=epochs
    )

    emb = model.encode(
        X,
        encoding_window='full_series'
    )

    return {
        'embeddings': emb,
        'subjects': subjects
    }
")

# =========================================================
# EXPERIMENT SETTINGS
# =========================================================
n_iter <- 50

RES <- matrix(NaN, n_iter, 3)

colnames(RES) <- c(
  "fPCA_KMeans",
  "mFPCA_splines",
  "RankWalk_GNN"
)

# =========================================================
# MAIN LOOP
# =========================================================
source("simLongData_GNN.R")

for (ii in 1:n_iter) {

  cat("\n================ ITER", ii, "================\n")

  r_eta = 3
  r_sigma_diag = rep(3, 5)
  id = sample(1:5, 1)
  r_sigma_diag[id] = sample(5:20, 1)

  #print(r_sigma_diag)

  #Longdat2 <- simLongData(
  #  ranTimes = TRUE,
  #  n_i = 10,
  #  eta = r_eta,
  #  sigma_diag = r_sigma_diag
  #)

  Longdat2 = simLongData_hard()

  # =====================================================
  # WIDE FORMAT
  # =====================================================
  Longdat2_wide <- reshape(
    Longdat2,
    idvar = c("subject", "time", "cluster"),
    timevar = "outcome",
    direction = "wide"
  )

  trueClusIDs <- aggregate(
    Longdat2_wide$cluster,
    by = list(Longdat2_wide$subject),
    FUN = function(x) x[1]
  )[,2]

  set_levels <- 4

  DD <- as.matrix(Longdat2_wide[, 4:ncol(Longdat2_wide)])

  n_samples <- length(unique(Longdat2_wide$subject))

# =====================================================
# fPCA + KMeans (ROBUST fdapace VERSION)
# =====================================================
cat("fPCA + KMeans (robust fdapace)\n")

library(fdapace)

subjects <- sort(unique(Longdat2$subject))
outcomes <- length(unique(Longdat2$outcome))

fpca_features <- NULL
valid_subjects <- subjects  # will be updated per outcome

for (m in 1:outcomes) {

  # ---------------------------------------------
  # collect trajectories per subject
  # ---------------------------------------------
  Ly <- list()
  Lt <- list()
  id_map <- c()

  for (s in seq_along(subjects)) {

    sub <- Longdat2[
      Longdat2$subject == subjects[s] &
      Longdat2$outcome == m, ]

    if (nrow(sub) < 3){print("Äh");next}

    Ly[[length(Ly) + 1]] <- sub$y
    Lt[[length(Lt) + 1]] <- sub$time
    id_map <- c(id_map, subjects[s])
  }

  # skip if too sparse
  if (length(Ly) < 2){print("Äh");next}

  # ---------------------------------------------
  # FPCA (PACE)
  # ---------------------------------------------
  fpca_res <- fdapace::FPCA(
    Ly = Ly,
    Lt = Lt,
    optns = list(dataType = "Sparse")#, nRegGrid = 51)
  )
  
  if(fpca_res$selectK == 1){
    scores <- fpca_res$xiEst[, 1, drop = FALSE]  
  }else{
    scores <- fpca_res$xiEst[, 1:2, drop = FALSE]
  }

  # IMPORTANT: scores rows correspond to Ly order
  colnames(scores) <- paste0("PC", 1:dim(scores)[2], "_m", m)

  # ---------------------------------------------
  # initialize / align feature matrix
  # ---------------------------------------------
  if (is.null(fpca_features)) {

    fpca_features <- matrix(
      NA,
      nrow = length(subjects),
      ncol = ncol(scores) * outcomes
    )

    rownames(fpca_features) <- subjects
  }

  # fill only valid subjects
  idx <- match(id_map, subjects)

  start_col <- (m - 1) * 2 + 1
  end_col   <- m * 2

  fpca_features[idx, start_col:end_col] <- scores
}

# ---------------------------------------------
# remove incomplete subjects safely
# ---------------------------------------------
keep <- complete.cases(fpca_features)

fpca_features <- fpca_features[keep, , drop = FALSE]
trueClusIDs_clean <- trueClusIDs[keep]

# ---------------------------------------------
# clustering
# ---------------------------------------------
print(dim(fpca_features))
ari_fpca <- ARI(
  trueClusIDs_clean,
  kmeans(scale(fpca_features), 4, nstart = 25)$cluster
)

cat("ARI fPCA:", ari_fpca, "\n")

# =====================================================
# mFPCA + KMeans (SPLINES FIXED VERSION)
# =====================================================
cat("mFPCA + KMeans (splines)\n")

library(funData)
library(MFPCA)

global_grid <- seq(min(Longdat2$time), max(Longdat2$time), length.out = 51)

multi_list <- lapply(1:outcomes, function(m) {

  X_mat <- matrix(NA, nrow = length(subjects), ncol = length(global_grid))

  for (s in seq_along(subjects)) {

    sub <- Longdat2[
      Longdat2$subject == subjects[s] &
      Longdat2$outcome == m, ]

    if (nrow(sub) < 2) next

    X_mat[s, ] <- approx(
      sub$time,
      sub$y,
      xout = global_grid,
      rule = 2
    )$y
  }

  funData(argvals = global_grid, X = X_mat)
})

mFD <- multiFunData(multi_list)

# -----------------------------------------------------
# IMPORTANT FIX: use splines1D (NOT PACE)
# -----------------------------------------------------
uniExp <- replicate(
  length(mFD),
  list(type = "splines1D", k = 10),
  simplify = FALSE
)

# -----------------------------------------------------
# Multivariate FPCA
# -----------------------------------------------------
mfpca_res <- MFPCA(
  mFD,
  M = 2,
  uniExpansions = uniExp
)

# -----------------------------------------------------
# Scores + clustering
# -----------------------------------------------------
mfpca_scores <- mfpca_res$scores

ari_mfpca <- ARI(
  trueClusIDs,
  kmeans(scale(mfpca_scores), 4, nstart = 25)$cluster
)

cat("ARI mFPCA:", ari_mfpca, "\n")
  

  # =====================================================
  # RankWalk GNN
  # =====================================================
  cat("RankWalk GNN\n")

  res_gnn <- py$run_rankwalk_gnn(Longdat2, 100L, 1e-3, 10L, 20L)

  emb <- res_gnn$embeddings
  sub <- as.numeric(res_gnn$subjects)

  feat <- lapply(sort(unique(sub)), function(g) {
    idx <- which(sub == g)
    as.vector(t(emb[idx, , drop = FALSE]))
  })

  ari_gnn <- ARI(
    trueClusIDs,
    kmeans(scale(do.call(rbind, feat)), 4, nstart = 50)$cluster
  )

  cat("ARI RankWalk:", ari_gnn, "\n")

  # =====================================================
  # STORE RESULTS
  # =====================================================
  RES[ii, ] <- c(ari_fpca, ari_mfpca, ari_gnn)

  print(RES)
}


# =========================================================
# PLOT RESULTS
# =========================================================
RES_df <- as.data.frame(RES)
RES_m <- melt(RES_df)
colnames(RES_m) = c("Method", "value")

ggplot(RES_m, aes(x = Method, y = value, fill = Method)) +
  geom_boxplot() +
  ylim(0,1) +
  theme_minimal() +
  xlab("Method") +
  ylab("Adjusted Rand Index (ARI)") +
  theme(
    text = element_text(size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# =========================================================
# PAIRWISE DOMINANCE HEATMAP (TIES EXCLUDED)
# =========================================================

library(ggplot2)
library(reshape2)

methods <- colnames(RES)
n_methods <- length(methods)

DOM <- matrix(
  0,
  nrow = n_methods,
  ncol = n_methods
)

rownames(DOM) <- methods
colnames(DOM) <- methods

# =========================================================
# COMPUTE PAIRWISE WIN RATES (EXCLUDING TIES)
# =========================================================

for(i in 1:n_methods){

  for(j in 1:n_methods){

    if(i == j){

      DOM[i,j] <- NA

    } else {

      wins <- sum(RES[,i] > RES[,j], na.rm = TRUE)
      losses <- sum(RES[,i] < RES[,j], na.rm = TRUE)
      total <- wins + losses   # ties excluded

      if(total == 0){
        DOM[i,j] <- NA
      } else {
        DOM[i,j] <- wins / total
      }
    }
  }
}

# =========================================================
# MELT FOR GGPLOT
# =========================================================

DOM_m <- melt(DOM)

colnames(DOM_m) <- c(
  "Method1",
  "Method2",
  "WinRate"
)

# =========================================================
# HEATMAP
# =========================================================

p <- ggplot(
  DOM_m,
  aes(
    x = Method1,
    y = Method2,
    fill = WinRate
  )
) +

  geom_tile(color = "white") +

  geom_text(
    aes(label = ifelse(is.na(WinRate), "", round(WinRate, 2))),
    size = 4
  ) +

  scale_fill_gradient2(
    low = "firebrick",
    mid = "white",
    high = "darkgreen",
    midpoint = 0.5,
    na.value = "grey90",
    limits = c(0,1)
  ) +

  theme_minimal() +

  xlab("") +
  ylab("") +

  ggtitle(
    "Pairwise dominance matrix \nP(Method1 beats Method2)"
  ) +

  theme(
    text = element_text(size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p)