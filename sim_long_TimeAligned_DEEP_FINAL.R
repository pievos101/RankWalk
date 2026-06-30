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
    build_temporal_graph
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

    G, labels_df = build_temporal_graph(
        df,
        k_similarity=5
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

  r_eta = 3
  r_sigma_diag = rep(7, 5)
  id = sample(1:5, 1)
  #r_sigma_diag[id] =  sample(5:20, 1)
  #print(r_sigma_diag)

  Longdat2 <- simLongData(
    ranTimes = FALSE,
    n_i = 10,
    eta = r_eta,
    sigma_diag = r_sigma_diag
  )

  #Longdat2 = simLongData_hard(ranTimes = FALSE)


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
    n_features = NaN,
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
    n_features = NaN,
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
    kmeans(scale(do.call(rbind, feat)), 4, nstart = 50)$cluster
  )

  cat("ARI RankWalk:", ari7, "\n")

  # =====================================================
  # STORE RESULTS
  # =====================================================
  RES[ii,] <- c(ari1, ari2, ari3, ari4, ari5, ari6, ari7)

  print(RES[,2:7])
}


# =========================================================
# PLOT RESULTS
# =========================================================
library(ggplot2)
library(reshape)

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

### EVEN NICER

library(ggplot2)
library(reshape)

RES_df <- as.data.frame(RES)

# reshape (base-style via reshape package)
RES_m <- melt(RES_df)
colnames(RES_m) <- c("Method", "ARI")

# order methods by median ARI
RES_m$Method <- reorder(RES_m$Method, RES_m$ARI, FUN = median)

ggplot(RES_m, aes(x = Method, y = ARI, fill = Method)) +
  geom_violin(trim = FALSE, alpha = 0.4, color = NA) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.08, alpha = 0.4, size = 1) +
  coord_cartesian(ylim = c(0, 1)) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 35, hjust = 1),
    panel.grid.major.x = element_blank()
  ) +
  labs(
    x = "Method",
    y = "Adjusted Rand Index (ARI)"
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