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
    compute_jaccard_fast
)

# =========================================================
# RankWalk GNN (UNCHANGED)
# =========================================================
def run_rankwalk_gnn(df, epochs=300, lr=1e-3, top_k=10, walk_length=20):

    df = pd.DataFrame(df)

    G, _ = build_temporal_graph_aligned(
        df,
        k_similarity=10,
        k_align=10,
        knn_mode='mutual'
    )

    node_list = list(G.nodes())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_list, t_list = [], []

    for n in node_list:
        x_list.append(G.nodes[n]['features'])
        t_list.append(G.nodes[n]['time'])

    x = torch.tensor(np.array(x_list), dtype=torch.float32, device=device)

    t = torch.tensor(t_list, dtype=torch.float32, device=device).unsqueeze(1)
    t = (t - t.mean()) / (t.std() + 1e-8)

    x = torch.cat([x, t], dim=1)

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
sys.path.append('/home/bpfeif/GitHub/ts2vec')

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

RES <- matrix(NaN, n_iter, 7)

colnames(RES) <- c(
  "TAPIO_PC1",
  "TAPIO_weighted",
  "kml3d",
  "fPCA_KMeans",
  "DTW",
  "TS2Vec",
  "RankWalk_GNN"
)

# =========================================================
# MAIN LOOP
# =========================================================
source("simLongData_GNN.R")

for (ii in 1:n_iter) {

  cat("\n================ ITER", ii, "================\n")

  r_eta = 5
  r_sigma_diag = rep(5, 5)
  print(r_sigma_diag)

  Longdat2 <- simLongData(
    ranTimes = FALSE,
    n_i = 10,
    eta = r_eta,
    sigma_diag = r_sigma_diag
  )

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
  DD <- as.matrix(Longdat2_wide[,4:ncol(Longdat2_wide)])

  # =====================================================
  # TAPIO PC1
  # =====================================================
  cat("TAPIO PC1\n")
  res1 <- longTAPIO_trajectories(
    DD, k = 4,
    user_id = Longdat2_wide$subject,
    levels = set_levels,
    verbose = 0,
    n_trees = 500,
    method = "ward.D2",
    n_features = NA,
    do.leveling = TRUE,
    pca_selection = "first"
  )
  ari1 <- ARI(trueClusIDs, res1$cl)

  # =====================================================
  # TAPIO weighted
  # =====================================================
  cat("TAPIO weighted\n")
  res2 <- longTAPIO_trajectories(
    DD, k = 4,
    user_id = Longdat2_wide$subject,
    levels = set_levels,
    verbose = 0,
    n_trees = 500,
    method = "ward.D2",
    n_features = NA,
    do.leveling = TRUE,
    pca_selection = "random_weighted"
  )
  ari2 <- ARI(trueClusIDs, res2$cl)

  # =====================================================
  # kml3d
  # =====================================================
  cat("kml3d\n")

  n_samples <- length(unique(Longdat2_wide$subject))
  tr1nn <- array(NaN, dim = c(n_samples, 10, 5))
  IN <- as.matrix(Longdat2_wide[,4:ncol(Longdat2_wide)])

  for (xx in 1:5) {
    for (tt in 1:10) {
      tr1nn[, tt, xx] <- IN[seq(tt, nrow(Longdat2_wide), by = 10), xx]
    }
  }

  object_kml <- clusterLongData3d(
    traj = tr1nn,
    idAll = as.character(1:n_samples),
    time = 1:10,
    varNames = paste("Marker", 1:5, sep = "")
  )

  kml3d(
    object_kml,
    nbClusters = 4,
    nbRedrawing = 10,
    toPlot = "none",
    parAlgo = parKml3d(imputationMethod = "copyMean")
  )

  ari3 <- ARI(trueClusIDs, getClusters(object_kml, 4))

  # =====================================================
  # fPCA + KMeans
  # =====================================================
  cat("fPCA + KMeans\n")

  basis <- create.bspline.basis(rangeval = c(1, 10), nbasis = 5)
  fpca_features <- matrix(0, nrow = n_samples, ncol = 0)

  for (m in 1:5) {
    fd_obj <- smooth.basis(1:10, t(tr1nn[,,m]), basis)$fd
    fpca_res <- pca.fd(fd_obj, nharm = 2)
    fpca_features <- cbind(fpca_features, fpca_res$scores)
  }

  ari4 <- ARI(trueClusIDs, kmeans(fpca_features, 4, nstart = 25)$cluster)

  # =====================================================
  # DTW CLUSTERING (NEW BASELINE)
  # =====================================================
  cat("DTW clustering\n")

  dtw_data <- list()

  subj_ids <- unique(Longdat2_wide$subject)

  for (s in seq_along(subj_ids)) {

    sub <- Longdat2_wide[Longdat2_wide$subject == subj_ids[s], ]

    # flatten multivariate into matrix (time x features)
    dtw_data[[s]] <- as.matrix(sub[, 4:ncol(sub)])
  }

  # DTW distance matrix
  dmat <- proxy::dist(dtw_data, method = function(x, y) {
    dtw::dtw(x, y, distance.only = TRUE)$distance
  })

  # k-means via PAM on distance
  km_dtw <- cluster::pam(as.dist(dmat), k = 4)

  ari5 <- ARI(trueClusIDs, km_dtw$clustering)

  cat("ARI DTW:", ari5, "\n")

  # =====================================================
  # TS2VEC OFFICIAL (AFTER DTW)
  # =====================================================
  cat("TS2Vec\n")

  res_ts <- py$run_ts2vec_official(Longdat2, 50L)

  emb_ts <- res_ts$embeddings

  ari6 <- ARI(
    trueClusIDs,
    kmeans(scale(emb_ts), 4, nstart = 25)$cluster
  )

  cat("ARI TS2Vec:", ari6, "\n")

  # =====================================================
  # RankWalk GNN (LAST)
  # =====================================================
  cat("RankWalk GNN\n")

  res_gnn <- py$run_rankwalk_gnn(Longdat2, 100L, 1e-3, 10L, 20L)

  emb <- res_gnn$embeddings
  sub <- as.numeric(res_gnn$subjects)

  feat <- lapply(sort(unique(sub)), function(g) {
    idx <- which(sub == g)
    as.vector(t(emb[idx,,drop=FALSE]))
  })

  ari7 <- ARI(
    trueClusIDs,
    kmeans(scale(do.call(rbind, feat)), 4, nstart = 25)$cluster
  )

  cat("ARI RankWalk:", ari7, "\n")

  # =====================================================
  # STORE RESULTS
  # =====================================================
  RES[ii,] <- c(ari1, ari2, ari3, ari4, ari5, ari6, ari7)

  print(RES[,2:7])
}