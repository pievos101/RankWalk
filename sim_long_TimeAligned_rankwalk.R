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

# =========================================================
# PYTHON: RANKWALK GNN (FIXED INPUT ASSUMPTION)
# =========================================================
py_run_string("
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch_geometric.utils import from_networkx, to_undirected

from rankwalk import build_temporal_graph, train_gnn, compute_jaccard_fast


def run_rankwalk_gnn(df,
                     epochs=300,
                     lr=1e-3,
                     top_k=10,
                     walk_length=20):

    # =====================================================
    # FIX: enforce pandas DataFrame
    # =====================================================
    df = pd.DataFrame(df)

    # =====================================================
    # IMPORTANT: expects LONG format with 'outcome'
    # =====================================================
    G, labels_df = build_temporal_graph(df, k_similarity=10)

    labels = np.array(labels_df['cluster']).ravel()

    # clean edges
    for u, v in G.edges():
        G[u][v].clear()

    node_list = list(G.nodes())

    # =====================================================
    # PyG conversion
    # =====================================================
    data = from_networkx(G)
    edge_index = to_undirected(data.edge_index)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = edge_index.to(device)

    x = torch.tensor(
        np.array([G.nodes[n]['features'] for n in node_list]),
        dtype=torch.float,
        device=device
    )

    t = torch.tensor(
        np.array([[G.nodes[n]['time']] for n in node_list]),
        dtype=torch.float,
        device=device
    )

    #x = torch.cat([x, t], dim=1)

    # =====================================================
    # RankWalk similarity
    # =====================================================
    J = compute_jaccard_fast(edge_index, G.number_of_nodes())

    # =====================================================
    # Train GNN
    # =====================================================
    emb = train_gnn(
        x,
        edge_index,
        J,
        epochs=epochs,
        lr=lr,
        walk_length=walk_length,
        top_k=top_k,
        device=device
    )

    emb = emb.detach().cpu().numpy()

    node2subject = np.array([G.nodes[n]['subject'] for n in node_list])

    return {
        'embeddings': emb,
        'subjects': node2subject
    }
")

# =========================================================
# EXPERIMENT SETTINGS
# =========================================================
n_iter <- 50

RES <- matrix(NaN, n_iter, 3)
colnames(RES) <- c(
  "TAPIO_PC1",
  "TAPIO_weighted",
  "RankWalk_GNN"
)

#set.seed(123)

# =========================================================
# MAIN LOOP
# =========================================================
for (ii in 1:n_iter) {

  cat("\n================ ITER", ii, "================\n")

  # -----------------------------
  # SIMULATION (LONG FORMAT)
  # -----------------------------
  r_eta = 3 #sample(1:10,1)
  r_sigma_diag = rep(3, 5) #sample(1:6, 5, replace=TRUE)
  #r_sigma_diag = sample(3:10, 5, replace=TRUE)
  id = sample(1:5, 1)
  r_sigma_diag[id] =  sample(3:20, 1)


  print(r_sigma_diag)

  Longdat2 <- simLongData(
    ranTimes = FALSE,
    n_i = 10,
    eta = r_eta,
    sigma_diag = r_sigma_diag
  )

  # -----------------------------
  # WIDE FORMAT FOR TAPIO ONLY
  # -----------------------------
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
  # TAPIO PC1
  # =====================================================
  cat("TAPIO PC1\n")

  res1 <- longTAPIO_trajectories(
    DD,
    k = 4,
    user_id = Longdat2_wide$subject,
    levels = set_levels,
    verbose = 0,
    n_trees = 500,
    method = "ward.D2",
    n_features = set_n_features,
    do.leveling = TRUE,
    pca_selection = "first"
  )

  ari1 <- ARI(trueClusIDs, res1$cl)

  # =====================================================
  # TAPIO weighted
  # =====================================================
  cat("TAPIO weighted\n")

  res2 <- longTAPIO_trajectories(
    DD,
    k = 4,
    user_id = Longdat2_wide$subject,
    levels = set_levels,
    verbose = 0,
    n_trees = 500,
    method = "ward.D2",
    n_features = set_n_features,
    do.leveling = TRUE,
    pca_selection = "random_weighted"
  )

  ari2 <- ARI(trueClusIDs, res2$cl)

  # =====================================================
  # RankWalk GNN (USES LONG FORMAT)
  # =====================================================
  cat("RankWalk GNN\n")

  res_gnn <- py$run_rankwalk_gnn(
    Longdat2,
    epochs = 200L,
    lr = 0.001,
    top_k = 20L,
    walk_length = 20L
  )

  # -------------------------------------------------
  # Extract
  # -------------------------------------------------
  EMB <- res_gnn$embeddings
  SUBJ <- as.numeric(res_gnn$subjects)

  subjects_unique <- sort(unique(SUBJ))

  n_subj <- length(subjects_unique)

  # -------------------------------------------------
  # Node indices by subject
  # -------------------------------------------------
  nodes_by_subject <- lapply(subjects_unique, function(s) {
    which(SUBJ == s)
  })

  # -------------------------------------------------
  # Pairwise node distances
  # -------------------------------------------------
  DIST <- as.matrix(dist(EMB))

  # -------------------------------------------------
  # Subject-subject block distances
  # -------------------------------------------------
  S <- matrix(0, n_subj, n_subj)

  for (a in seq_len(n_subj)) {

    idx_a <- nodes_by_subject[[a]]

    for (b in seq_len(n_subj)) {

      idx_b <- nodes_by_subject[[b]]

      block <- DIST[idx_a, idx_b, drop = FALSE]

      S[a, b] <- mean(block, na.rm = TRUE)
    }
  }

  # ---------------------------------------------------------
  # Optional cleanup
  # ---------------------------------------------------------

  diag(S) <- 0
  S[is.na(S)] <- 1

  # ---------------------------------------------------------
  # Symmetrie
  # ---------------------------------------------------------

  S <- (S + t(S)) / 2
  
  # -------------------------------------------------
  # Ward clustering
  # -------------------------------------------------

  hc <- hclust(as.dist(S), method = "ward.D2")

  clusters <- cutree(hc, k = 4)

  ari3 <- ARI(trueClusIDs, clusters)

  # =====================================================
  # STORE RESULTS
  # =====================================================
  RES[ii,1] <- ari1
  RES[ii,2] <- ari2
  RES[ii,3] <- ari3

  print(RES)
}

stop("All good!")

# =========================================================
# PLOT RESULTS
# =========================================================
library(ggplot2)
library(reshape)

RES_df <- as.data.frame(RES)
#colnames(RES_df) = c("longTAPIO_PCw","TopKGraphs","longTAPIO_PC1")
RES_m <- melt(RES_df)
colnames(RES_m) = c("Method","value")

ggplot(RES_m, aes(x = Method, y = value, fill = Method)) +
  geom_boxplot() +
  ylim(0,1) +
  theme_minimal() +
  xlab("Method") +
  ylab("ARI") +
  theme(
    text = element_text(size = 15),
    axis.text.x = element_blank()
  )