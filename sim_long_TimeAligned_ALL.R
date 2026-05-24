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
library(fda) # For the Functional Data Analysis/fPCA ML baseline

# =========================================================
# PYTHON: RANKWALK GNN 
# =========================================================
py_run_string("
import numpy as np
import pandas as pd
import torch

from sklearn.cluster import KMeans

from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_undirected

from rankwalk import (
    build_temporal_graph,
    train_gnn,
    compute_jaccard_fast
)

def run_rankwalk_gnn(
    df,
    epochs=300,
    lr=1e-3,
    top_k=10,
    walk_length=20
):

    df = pd.DataFrame(df)

    G, labels_df = build_temporal_graph(
        df,
        k_similarity=10
    )

    node_list = list(G.nodes())

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    # -----------------------------------------
    # Extract node features + time feature
    # -----------------------------------------
    x_list = []
    time_list = []

    for n in node_list:
        x_list.append(G.nodes[n]['features'])
        time_list.append(G.nodes[n]['time'])

    x = torch.tensor(np.array(x_list), dtype=torch.float, device=device)

    time = torch.tensor(time_list, dtype=torch.float, device=device).unsqueeze(1)

    # normalize time (important)
    time = (time - time.mean()) / (time.std() + 1e-8)

    # concatenate
    x = torch.cat([x, time], dim=1)

    # -----------------------------------------
    # Build edge index + edge types
    # -----------------------------------------
    edges = []
    edge_types = []

    for u, v, attr in G.edges(data=True):

        edges.append([u, v])
        edge_types.append(attr['edge_type'])

        edges.append([v, u])
        edge_types.append(attr['edge_type'])

    edge_index = torch.tensor(
        edges,
        dtype=torch.long,
        device=device
    ).t().contiguous()

    edge_type = torch.tensor(
        edge_types,
        dtype=torch.long,
        device=device
    )

    # -----------------------------------------
    # Jaccard matrix
    # -----------------------------------------
    J = compute_jaccard_fast(
        edge_index,
        G.number_of_nodes(),
        device=device
    )

    # -----------------------------------------
    # Train model
    # -----------------------------------------
    emb = train_gnn(
        x,
        edge_index,
        edge_type,
        J,
        epochs=epochs,
        lr=lr,
        walk_length=walk_length,
        top_k=top_k,
        device=device
    )

    emb = emb.detach().cpu().numpy()

    node2subject = np.array([
        G.nodes[n]['subject']
        for n in node_list
    ])

    return {
        'embeddings': emb,
        'subjects': node2subject
    }
")

# =========================================================
# EXPERIMENT SETTINGS
# =========================================================
n_iter <- 50

RES <- matrix(NaN, n_iter, 5)
colnames(RES) <- c(
  "TAPIO_PC1",
  "TAPIO_weighted",
  "kml3d",
  "fPCA_KMeans", 
  "RankWalk_GNN"
)

# =========================================================
# MAIN LOOP
# =========================================================
for (ii in 1:n_iter) {

  cat("\n================ ITER", ii, "================\n")

  r_eta = 3 
  r_sigma_diag = rep(5, 5) 
  id = sample(1:5, 1)
  #r_sigma_diag[id] =  sample(3:20, 1)

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
  set_n_features <- NaN

  DD <- as.matrix(Longdat2_wide[,4:ncol(Longdat2_wide)])

  # =====================================================
  # METHOD 1: TAPIO PC1
  # =====================================================
  cat("TAPIO PC1\n")
  res1 <- longTAPIO_trajectories(
    DD, k = 4, user_id = Longdat2_wide$subject, levels = set_levels,
    verbose = 0, n_trees = 500, method = "ward.D2",
    n_features = set_n_features, do.leveling = TRUE, pca_selection = "first"
  )
  ari1 <- ARI(trueClusIDs, res1$cl)

  # =====================================================
  # METHOD 2: TAPIO weighted
  # =====================================================
  cat("TAPIO weighted\n")
  res2 <- longTAPIO_trajectories(
    DD, k = 4, user_id = Longdat2_wide$subject, levels = set_levels,
    verbose = 0, n_trees = 500, method = "ward.D2",
    n_features = set_n_features, do.leveling = TRUE, pca_selection = "random_weighted"
  )
  ari2 <- ARI(trueClusIDs, res2$cl)

  # =====================================================
  # METHOD 3: MULTIVARIATE KML3D
  # =====================================================
  cat("kml3d\n")
  
  n_samples <- length(unique(Longdat2_wide$subject))
  tr1nn <- array(NaN, dim = c(n_samples, 10, 5))
  IN <- as.matrix(Longdat2_wide[, 4:ncol(Longdat2_wide)])

  for (xx in 1:5) {
    tr1nn[, 1, xx] <- IN[seq(1, nrow(Longdat2_wide), by = 10), xx]
    tr1nn[, 2, xx] <- IN[seq(2, nrow(Longdat2_wide), by = 10), xx]
    tr1nn[, 3, xx] <- IN[seq(3, nrow(Longdat2_wide), by = 10), xx]
    tr1nn[, 4, xx] <- IN[seq(4, nrow(Longdat2_wide), by = 10), xx]
    tr1nn[, 5, xx] <- IN[seq(5, nrow(Longdat2_wide), by = 10), xx]
    tr1nn[, 6, xx] <- IN[seq(6, nrow(Longdat2_wide), by = 10), xx]
    tr1nn[, 7, xx] <- IN[seq(7, nrow(Longdat2_wide), by = 10), xx]
    tr1nn[, 8, xx] <- IN[seq(8, nrow(Longdat2_wide), by = 10), xx]
    tr1nn[, 9, xx] <- IN[seq(9, nrow(Longdat2_wide), by = 10), xx]
    tr1nn[, 10, xx] <- IN[seq(10, nrow(Longdat2_wide), by = 10), xx]
  }

  idAll_kml <- as.character(1:n_samples)
  time_kml <- 1:10

  object_kml <- clusterLongData3d(
    traj = tr1nn,
    idAll = idAll_kml,
    time = time_kml,
    varNames = paste("Marker", 1:5, sep = "")
  )

  kml3d(
    object_kml, 
    nbClusters = 4, 
    nbRedrawing = 10, 
    toPlot = "none", 
    parAlgo = parKml3d(imputationMethod = "copyMean")
  )

  cl_kml <- getClusters(object_kml, 4)
  ari3 <- ARI(trueClusIDs, cl_kml)

  # =====================================================
  # METHOD 4: MULTIVARIATE fPCA + K-MEANS
  # =====================================================
  cat("fPCA + KMeans\n")
  
  basis <- create.bspline.basis(rangeval = c(1, 10), nbasis = 5)
  fpca_features <- matrix(0, nrow = n_samples, ncol = 0)
  
  for (m in 1:5) {
    marker_data <- t(tr1nn[, , m])
    fd_obj <- smooth.basis(1:10, marker_data, basis)$fd
    fpca_res <- pca.fd(fd_obj, nharm = 2)
    fpca_features <- cbind(fpca_features, fpca_res$scores)
  }
  
  km_fpca <- kmeans(fpca_features, centers = 4, nstart = 25)
  ari4 <- ARI(trueClusIDs, km_fpca$cluster)

  # =====================================================
  # METHOD 5: RankWalk GNN (Switched to K-Means Clustering)
  # =====================================================
  cat("RankWalk GNN\n")
  res_gnn <- py$run_rankwalk_gnn(
    Longdat2, epochs = 100L, lr = 0.001, top_k = 10L, walk_length = 20L
  )

  EMB <- res_gnn$embeddings    
  SUBJ <- as.numeric(res_gnn$subjects)
  subjects_unique <- sort(unique(SUBJ))
  n_subj <- length(subjects_unique)

  # Aggregate time-step node embeddings into a single vector per subject (Mean Pooling)
  emb_dim <- ncol(EMB)
  subject_features <- matrix(0, nrow = n_subj, ncol = emb_dim)

  for (g in seq_len(n_subj)) {
    idx <- which(SUBJ == subjects_unique[g])
    if (length(idx) > 1) {
      subject_features[g, ] <- colMeans(EMB[idx, , drop = FALSE], na.rm = TRUE)
    } else if (length(idx) == 1) {
      subject_features[g, ] <- EMB[idx, ]
    }
  }

  # Execute K-Means clustering on the structural trajectory profiles
  km_gnn <- kmeans(subject_features, centers = 4, nstart = 25)
  ari5 <- ARI(trueClusIDs, km_gnn$cluster)


  # =====================================================
  # STORE RESULTS
  # =====================================================
  RES[ii,1] <- ari1
  RES[ii,2] <- ari2
  RES[ii,3] <- ari3
  RES[ii,4] <- ari4
  RES[ii,5] <- ari5

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