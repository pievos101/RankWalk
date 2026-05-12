library(TAPIO)
library(clusterMLD)
library(MASS)
library(aricode)
library(reshape)
library(reticulate)
library(ggplot2)

# =========================================================
# PYTHON SETUP
# =========================================================

pd <- import("pandas")
np <- import("numpy")
torch <- import("torch")

builtins <- import_builtins()
nx <- import("networkx")

torch_geometric_utils <- import("torch_geometric.utils")
from_networkx <- torch_geometric_utils$from_networkx
to_undirected <- torch_geometric_utils$to_undirected

sk_cluster <- import("sklearn.cluster")
KMeans <- sk_cluster$KMeans

sk_metrics <- import("sklearn.metrics")
adjusted_rand_score <- sk_metrics$adjusted_rand_score

rankwalk <- import("rankwalk")

# =========================================================
# POOLING
# =========================================================

pool_patient_embeddings <- function(emb, node2patient){

  emb_np <- py_to_r(emb$detach()$cpu()$numpy())

  patients <- unique(node2patient)

  patient_emb <- matrix(
    0,
    nrow = length(patients),
    ncol = ncol(emb_np)
  )

  rownames(patient_emb) <- patients

  for(i in seq_along(patients)){
    idx <- which(node2patient == patients[i])
    patient_emb[i, ] <- colMeans(emb_np[idx, , drop = FALSE])
  }

  patient_emb
}

# =========================================================
# RANKWALK GNN (FULL FIXED PIPELINE)
# =========================================================

run_rankwalk_gnn <- function(df,
                             walk_length = 20,
                             top_k = 10,
                             epochs = 200,
                             lr = 1e-3){

  # -------------------------------------------------------
  # BUILD GRAPH
  # -------------------------------------------------------

  temp <- rankwalk$build_temporal_graph(
    df,
    k_similarity = as.integer(10)
  )

  G <- temp[[1]]
  labels_df <- temp[[2]]

  # -------------------------------------------------------
  # RAW NODE IDS
  # -------------------------------------------------------

  node_list <- py_to_r(builtins$list(G$nodes()))
  node_list <- as.character(node_list)

  # =======================================================
  # 🔥 STEP 1: CREATE CONTIGUOUS NODE INDEX MAP
  # =======================================================

  node_map <- setNames(seq_along(node_list) - 1, node_list)

  # =======================================================
  # 🔥 STEP 2: REBUILD GRAPH WITH CONTIGUOUS IDS
  # =======================================================

  edges <- py_to_r(builtins$list(G$edges()))

  G_clean <- nx$Graph()

  # nodes must be 0..N-1
  G_clean$add_nodes_from(seq_along(node_list) - 1)

  for(e in edges){

    u_old <- as.character(e[[1]])
    v_old <- as.character(e[[2]])

    u <- node_map[[u_old]]
    v <- node_map[[v_old]]

    G_clean$add_edge(as.integer(u), as.integer(v))
  }

  G <- G_clean

  # -------------------------------------------------------
  # PyG conversion
  # -------------------------------------------------------

  data <- from_networkx(G)
  edge_index <- to_undirected(data$edge_index)

  # -------------------------------------------------------
  # FEATURES (SAFE INDEXING)
  # -------------------------------------------------------

  feat_list <- lapply(node_list, function(n_old){
    py_to_r(G$nodes[[node_map[[n_old]]]][["features"]])
  })

  x_mat <- do.call(rbind, feat_list)

  # -------------------------------------------------------
  # TIME FEATURE (SAFE FALLBACK)
  # -------------------------------------------------------

  time_feat <- matrix(
    unlist(lapply(node_list, function(n_old){

      val <- G$nodes[[node_map[[n_old]]]]

      if (!is.null(val) && "time" %in% names(val)) {
        py_to_r(val[["time"]])
      } else {
        0
      }

    })),
    ncol = 1
  )

  # -------------------------------------------------------
  # FINAL FEATURE MATRIX
  # -------------------------------------------------------

  x <- torch$tensor(
    cbind(x_mat, time_feat),
    dtype = torch$float
  )

  # -------------------------------------------------------
  # JACCARD
  # -------------------------------------------------------

  J <- rankwalk$compute_jaccard_fast(
    edge_index,
    as.integer(G$number_of_nodes())
  )

  # -------------------------------------------------------
  # TRAIN GNN
  # -------------------------------------------------------

  emb_node <- rankwalk$train_gnn(
    x,
    edge_index,
    J,
    epochs = as.integer(epochs),
    lr = lr,
    walk_length = as.integer(walk_length),
    top_k = as.integer(top_k),
    device = "cpu"
  )

  # -------------------------------------------------------
  # NODE → PATIENT MAPPING
  # -------------------------------------------------------

  node2patient <- unlist(lapply(node_list, function(n_old){
    py_to_r(G$nodes[[node_map[[n_old]]]][["subject"]])
  }))

  # -------------------------------------------------------
  # POOLING
  # -------------------------------------------------------

  emb_patient <- pool_patient_embeddings(emb_node, node2patient)

  # -------------------------------------------------------
  # CLUSTERING
  # -------------------------------------------------------

  labels_r <- py_to_r(
    labels_df$groupby("subject")[["cluster"]]$first()
  )

  patient_labels <- as.numeric(labels_r$to_numpy())

  k <- length(unique(patient_labels))

  km <- KMeans(
    n_clusters = as.integer(k),
    n_init = as.integer(10)
  )

  pred <- py_to_r(km$fit_predict(emb_patient))

  # -------------------------------------------------------
  # ARI
  # -------------------------------------------------------

  as.numeric(
    adjusted_rand_score(patient_labels, pred)
  )
}

# =========================================================
# SIMULATION
# =========================================================

set.seed(1)

n_iter <- 50

RES <- matrix(NaN, n_iter, 3)

colnames(RES) <- c(
  "longTAPIO_PC1",
  "longTAPIO_PCwr",
  "RankWalk_GNN"
)

# =========================================================
# MAIN LOOP
# =========================================================

for(ii in 1:n_iter){

  cat("\n=============================\n")
  cat("ITER:", ii, "\n")
  cat("=============================\n")

  # -----------------------------
  # SIMULATION
  # -----------------------------

  r_eta <- 3
  r_sigma_diag <- rep(3,5)

  id <- sample(1:5, 1)
  r_sigma_diag[id] <- sample(3:20, 1)

  Longdat2 <- simLongData(
    ranTimes = FALSE,
    n_i = 10,
    eta = r_eta,
    sigma_diag = r_sigma_diag
  )

  # -----------------------------
  # WIDE FORMAT
  # -----------------------------

  Longdat2_wide <- reshape(
    Longdat2,
    idvar = c("subject","time","cluster"),
    timevar = "outcome",
    direction = "wide"
  )

  trueClusDF <- aggregate(
    cluster ~ subject,
    data = Longdat2_wide,
    FUN = function(x) x[1]
  )

  trueClusIDs <- trueClusDF$cluster

  DD <- as.matrix(Longdat2_wide[,4:ncol(Longdat2_wide)])

  # =====================================================
  # TAPIO PC1
  # =====================================================

  res <- longTAPIO_trajectories(
    DD,
    k = 4,
    user_id = Longdat2_wide$subject,
    levels = 4,
    verbose = 0,
    n_trees = 500,
    method = "ward.D2",
    n_features = 5,
    do.leveling = TRUE,
    pca_selection = "first"
  )

  ari1 <- ARI(trueClusIDs, res$cl)

  # =====================================================
  # TAPIO WEIGHTED
  # =====================================================

  res <- longTAPIO_trajectories(
    DD,
    k = 4,
    user_id = Longdat2_wide$subject,
    levels = 4,
    verbose = 0,
    n_trees = 500,
    method = "ward.D2",
    n_features = 5,
    do.leveling = TRUE,
    pca_selection = "random_weighted"
  )

  ari2 <- ARI(trueClusIDs, res$cl)

  # =====================================================
  # RANKWALK GNN
  # =====================================================

  ari3 <- run_rankwalk_gnn(Longdat2)

  # -----------------------------
  # STORE
  # -----------------------------

  RES[ii,] <- c(ari1, ari2, ari3)

  print(round(RES[1:ii,], 3))
}

# =========================================================
# FINAL RESULTS
# =========================================================

print(round(apply(RES,2,mean),3))
print(round(apply(RES,2,sd),3))

# =========================================================
# PLOT
# =========================================================

RES_m <- melt(RES)
colnames(RES_m) <- c("Iter","Method","ARI")

ggplot(RES_m, aes(Method, ARI, fill=Method)) +
  geom_boxplot() +
  theme_minimal() +
  ylim(0,1)