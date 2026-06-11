# =========================================================
# LIBRARIES
# =========================================================
library(joineR)
library(MASS)
library(survival)
library(Hmisc)
library(reshape2)
library(ggplot2)
library(fda)
library(reticulate)
library(fdapace)

# =========================================================
# PYTHON (RANKWALK GNN)
# =========================================================
Sys.setenv(
  RETICULATE_PYTHON = path.expand("~/rankwalk-venv/bin/python")
)

use_python(
  Sys.getenv("RETICULATE_PYTHON"),
  required = TRUE
)

py_run_string("
import numpy as np
import pandas as pd
import torch

from rankwalk import (
    build_temporal_graph_grid,
    compute_jaccard_fast,
    train_gnn
)

def run_rankwalk_gnn(df, epochs=120):

    df = pd.DataFrame(df)

    G, _ = build_temporal_graph_grid(
        df,
        k_similarity=10,
        n_bins=5,
        overlap=0.5
    )

    nodes = list(G.nodes())

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    x, t = [], []

    for n in nodes:
        x.append(G.nodes[n]['features'])
        t.append(G.nodes[n]['time'])

    x = torch.tensor(
        np.array(x),
        dtype=torch.float32,
        device=device
    )

    t = torch.tensor(
        t,
        dtype=torch.float32,
        device=device
    ).unsqueeze(1)

    t = (t - t.mean()) / (t.std() + 1e-8)

    x = torch.cat([x, t], dim=1)

    edges = []
    et = []

    for u, v, a in G.edges(data=True):

        edges.append([u, v])
        et.append(a['edge_type'])

        edges.append([v, u])
        et.append(a['edge_type'])

    edge_index = torch.tensor(
        edges,
        dtype=torch.long,
        device=device
    ).t().contiguous()

    edge_type = torch.tensor(
        et,
        dtype=torch.long,
        device=device
    )

    J = compute_jaccard_fast(
        edge_index,
        G.number_of_nodes(),
        device=device
    )

    emb = train_gnn(
        x,
        edge_index,
        edge_type,
        J,
        epochs=epochs,
        lr=1e-3,
        walk_length=20,
        top_k=10,
        device=device
    )

    return {
        'embeddings': emb.detach().cpu().numpy(),
        'subjects': np.array(
            [G.nodes[n]['subject'] for n in nodes]
        )
    }
")

# =========================================================
# DATA
# =========================================================
data(heart.valve)

# longitudinal variables
feature_cols <- c(
  "log.lvmi",
  "ef",
  "log.grad"
)

# =========================================================
# CLEANING
# =========================================================
heart_clean <- heart.valve

for(v in feature_cols){

  heart_clean[[v]] <- suppressWarnings(
    as.numeric(heart_clean[[v]])
  )

}

heart_clean <- heart_clean[
  is.finite(heart_clean$time),
]

# keep only rows having at least one longitudinal value
keep_row <- apply(
  heart_clean[, feature_cols],
  1,
  function(x) any(is.finite(x))
)

heart_clean <- heart_clean[keep_row, ]

## LOG - TRANSFORM
heart_clean$lvmi <- log1p(as.numeric(heart_clean$lvmi))
heart_clean$grad <- log1p(as.numeric(heart_clean$grad))
heart_clean$ef   <- as.numeric(heart_clean$ef)


# =========================================================
# SURVIVAL DATA
# =========================================================
surv_data <- unique(
  heart_clean[, c(
    "num",
    "fuyrs",
    "status"
  )]
)

colnames(surv_data) <- c(
  "id",
  "time",
  "event"
)

surv_data$id <- as.numeric(surv_data$id)
surv_data$time <- as.numeric(surv_data$time)
surv_data$event <- as.numeric(surv_data$event)

surv_data <- surv_data[
  complete.cases(surv_data),
]

# =========================================================
# SAFE METRICS
# =========================================================
safe_cindex <- function(score, surv){

  if(length(unique(score)) < 2)
    return(NA)

  S <- Surv(
    surv$time,
    surv$event
  )

  out <- try(
    rcorr.cens(score, S),
    silent = TRUE
  )

  if(inherits(out, "try-error"))
    return(NA)

  as.numeric(out["C Index"])
}

safe_logrank <- function(cluster, surv){

  df <- data.frame(
    time = surv$time,
    event = surv$event,
    cluster = as.factor(cluster)
  )

  df <- df[complete.cases(df), ]

  if(length(unique(df$cluster)) < 2)
    return(NA)

  if(any(table(df$cluster) < 5))
    return(NA)

  tryCatch({

    survdiff(
      Surv(time, event) ~ cluster,
      data = df
    )$chisq

  }, error = function(e) NA)
}

# =========================================================
# ALIGN CLUSTERS
# =========================================================
align_clusters <- function(cluster, merged){

  cluster <- as.numeric(cluster)

  risk <- tapply(
    merged$time,
    cluster,
    mean,
    na.rm = TRUE
  )

  risk[is.na(risk)] <-
    max(risk, na.rm = TRUE) + 1

  ord <- order(risk)

  map <- setNames(
    seq_along(ord),
    ord
  )

  aligned <- map[
    as.character(cluster)
  ]

  as.numeric(aligned)
}

# =========================================================
# PARAMETERS
# =========================================================
n_iter <- 20
k_clusters <- 2

RES <- matrix(
  NA,
  n_iter,
  4
)

colnames(RES) <- c(
  "FPCA_C",
  "FPCA_LR",
  "GNN_C",
  "GNN_LR"
)

# =========================================================
# MAIN LOOP
# =========================================================
for(ii in 1:n_iter){

  cat(
    "\n============================\n",
    "ITERATION:", ii,
    "\n============================\n"
  )

  # =====================================================
  # FPCA
  # =====================================================
  fpca_features <- list()

  for(v in feature_cols){

    cat("FPCA:", v, "\n")

    tmp <- data.frame(
      id   = heart_clean$num,
      time = heart_clean$time,
      y    = heart_clean[[v]]
    )

    tmp <- tmp[
      is.finite(tmp$y) &
      is.finite(tmp$time),
    ]

    if(nrow(tmp) < 20)
      next

    Ly <- split(tmp$y, tmp$id)
    Lt <- split(tmp$time, tmp$id)

    fp <- try(
      FPCA(
        Ly = Ly,
        Lt = Lt,
        optns = list(
          dataType = "Sparse"
        )
      ),
      silent = TRUE
    )

    if(inherits(fp, "try-error"))
      next

    scores <- fp$xiEst

    if(is.null(dim(scores)))
      scores <- matrix(
        scores,
        ncol = 1
      )

    rownames(scores) <- names(Ly)

    scores <- scores[
      apply(
        scores,
        1,
        function(x)
          any(is.finite(x))
      ),
      ,
      drop = FALSE
    ]

    fpca_features[[v]] <- scores
  }

  ids <- sort(
    unique(heart_clean$num)
  )

  X_fpca <- NULL

  for(v in names(fpca_features)){

    S <- fpca_features[[v]]

    tmp <- matrix(
      0,
      nrow = length(ids),
      ncol = ncol(S)
    )

    rownames(tmp) <- ids

    common <- intersect(
      rownames(S),
      ids
    )

    tmp[
      match(common, ids),
    ] <- S[
      common,
      ,
      drop = FALSE
    ]

    X_fpca <- if(is.null(X_fpca))
      tmp else cbind(X_fpca, tmp)
  }

  if(is.null(X_fpca))
    next

  X_fpca[!is.finite(X_fpca)] <- 0

  X_fpca <- scale(X_fpca)

  keep <- apply(
    X_fpca,
    1,
    function(x)
      all(is.finite(x))
  )

  X_fpca <- X_fpca[
    keep,
    ,
    drop = FALSE
  ]

  ids_fpca <- as.numeric(
    rownames(X_fpca)
  )

  surv_fpca <- surv_data[
    surv_data$id %in% ids_fpca,
  ]

  cl_fpca <- kmeans(
    X_fpca,
    centers = k_clusters,
    nstart = 100
  )$cluster

  names(cl_fpca) <- rownames(X_fpca)

  merged_fpca <- merge(
    surv_fpca,
    data.frame(
      id = as.numeric(
        names(cl_fpca)
      ),
      cluster = cl_fpca
    ),
    by = "id"
  )

  if(length(unique(
    merged_fpca$cluster
  )) < 2)
    next

  cl_fpca_aligned <- align_clusters(
    merged_fpca$cluster,
    merged_fpca
  )

  c_fpca <- safe_cindex(
    cl_fpca_aligned,
    merged_fpca
  )

  lr_fpca <- safe_logrank(
    cl_fpca_aligned,
    merged_fpca
  )

  cat(
    "FPCA:",
    c_fpca,
    lr_fpca,
    "\n"
  )

  # =====================================================
  # RANKWALK GNN
  # =====================================================
  Longdat_list <- list()

  for(i in seq_along(feature_cols)){

    v <- feature_cols[i]

    tmp <- data.frame(
      subject = heart_clean$num,
      time    = heart_clean$time,
      outcome = i,
      y       = heart_clean[[v]]
    )

    tmp <- tmp[
      is.finite(tmp$y) &
      is.finite(tmp$time),
    ]

    Longdat_list[[i]] <- tmp
  }

  Longdat2 <- do.call(
    rbind,
    Longdat_list
  )

  res <- py$run_rankwalk_gnn(
    Longdat2,
    100L
  )

  emb <- res$embeddings
  sub <- as.numeric(
    res$subjects
  )

  subjects <- sort(
    unique(sub)
  )

  feat <- lapply(
    subjects,
    function(g){

      idx <- which(
        sub == g
      )

      colMeans(
        emb[idx, ,
            drop = FALSE]
      )
    }
  )

  feat <- do.call(
    rbind,
    feat
  )

  rownames(feat) <- subjects

  feat <- scale(feat)

  keep <- apply(
    feat,
    1,
    function(x)
      all(is.finite(x))
  )

  feat <- feat[
    keep,
    ,
    drop = FALSE
  ]

  cl_gnn <- kmeans(
    feat,
    centers = k_clusters,
    nstart = 100
  )$cluster

  names(cl_gnn) <- rownames(feat)

  merged_gnn <- merge(
    surv_data,
    data.frame(
      id = as.numeric(
        names(cl_gnn)
      ),
      cluster = cl_gnn
    ),
    by = "id"
  )

  if(length(unique(
    merged_gnn$cluster
  )) < 2)
    next

  cl_gnn_aligned <- align_clusters(
    merged_gnn$cluster,
    merged_gnn
  )

  c_gnn <- safe_cindex(
    cl_gnn_aligned,
    merged_gnn
  )

  lr_gnn <- safe_logrank(
    cl_gnn_aligned,
    merged_gnn
  )

  cat(
    "GNN:",
    c_gnn,
    lr_gnn,
    "\n"
  )

  RES[ii, ] <- c(
    c_fpca,
    lr_fpca,
    c_gnn,
    lr_gnn
  )

  print(RES)
}

# =========================================================
# RESULTS
# =========================================================
print(
  colMeans(
    RES,
    na.rm = TRUE
  )
)

print(
  apply(
    RES,
    2,
    sd,
    na.rm = TRUE
  )
)

# =========================================================
# BOXPLOT
# =========================================================
library(ggplot2)
library(gridExtra)

# ---- Boxplot data for C ----
df_C <- data.frame(
  value = c(RES$FPCA_C, RES$GNN_C),
  method = rep(c("FPCA", "GNN"), each = 20)
)

# ---- Boxplot data for LR ----
df_LR <- data.frame(
  value = c(RES$FPCA_LR, RES$GNN_LR),
  method = rep(c("FPCA", "GNN"), each = 20)
)

# ---- Plot 1: C ----
p1 <- ggplot(df_C, aes(x = method, y = value, fill = method)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = unique(RES$FPCA_C), linetype = "dashed", color = "red") +
  labs(title = "C: FPCA vs GNN", x = "", y = "C") +
  theme_minimal() +
  theme(legend.position = "none")

# ---- Plot 2: LR ----
p2 <- ggplot(df_LR, aes(x = method, y = value, fill = method)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = unique(RES$FPCA_LR), linetype = "dashed", color = "red") +
  labs(title = "LR: FPCA vs GNN", x = "", y = "LR") +
  theme_minimal() +
  theme(legend.position = "none")

# ---- Side by side ----
grid.arrange(p1, p2, ncol = 2)