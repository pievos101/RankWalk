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

    # =====================================================
    # Pooling (subject level)
    # =====================================================
    node2subject = np.array([G.nodes[n]['subject'] for n in node_list])

    subjects = np.unique(node2subject)
    subj_map = {s:i for i,s in enumerate(subjects)}

    pooled = np.zeros((len(subjects), emb.shape[1]))
    counts = np.zeros(len(subjects))

    for i, s in enumerate(node2subject):
        j = subj_map[s]
        pooled[j] += emb[i]
        counts[j] += 1

    pooled = pooled / np.maximum(counts[:, None], 1)

    # =====================================================
    # clustering
    # =====================================================
    k = len(np.unique(labels))
    km = KMeans(n_clusters=k, n_init=10)

    return km.fit_predict(pooled)
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
  r_eta <- 3
  r_sigma_diag <- rep(3, 5)
  id <- sample(1:5, 1)
  r_sigma_diag[id] <- sample(3:20, 1)

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
  set_n_features <- 5

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
  # RankWalk GNN (USES LONG FORMAT — FIX)
  # =====================================================
  cat("RankWalk GNN\n")

  clusters <- py$run_rankwalk_gnn(
    Longdat2,   
    epochs = 100L,
    lr = 0.001,
    top_k = 10L,
    walk_length = 20L
  )

  # NOTE: cluster alignment assumes same subject order
  ari3 <- ARI(trueClusIDs, as.numeric(clusters))

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
RES_df <- as.data.frame(RES)
RES_m <- melt(RES_df)

ggplot(RES_m, aes(x = variable, y = value, fill = variable)) +
  geom_boxplot() +
  ylim(0,1) +
  theme_minimal() +
  xlab("Method") +
  ylab("ARI") +
  theme(text = element_text(size = 15))