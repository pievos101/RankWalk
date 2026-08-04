# =========================================================
# PYTHON SETUP (MUST BE FIRST)
# =========================================================
Sys.setenv(RETICULATE_PYTHON = path.expand("~/vader-venv/bin/python"))

library(reticulate)

use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
py_config()

# =========================================================
# R LIBRARIES
# =========================================================
library(aricode)
library(reshape2)
library(ggplot2)

# =========================================================
# PYTHON: VaDER ONLY
# =========================================================
py_run_string("
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(False)

from VaDER.vader import VADER

def run_vader(X_train, y_train=None, W_train=None,
              epochs_pre=50, epochs=50,
              k=4):

    # init model
    vader = VADER(
        X_train=X_train,
        W_train=W_train,
        y_train=y_train,
        n_hidden=[12, 2],
        k=k,
        learning_rate=1e-3,
        output_activation=None,
        recurrent=True,
        cell_type='LSTM',
        batch_size=64
    )

    vader.pre_fit(n_epoch=epochs_pre, verbose=False)
    vader.fit(n_epoch=epochs, verbose=False)

    emb = vader.cluster(X_train)

    return {
        'clusters': np.array(emb)
    }
")

# =========================================================
# EXPERIMENT SETTINGS
# =========================================================
n_iter <- 50

RES <- matrix(NaN, n_iter, 1)
colnames(RES) <- c("VaDER")

# =========================================================
# MAIN LOOP
# =========================================================
#source("simLongData_GNN.R")

for (ii in 1:n_iter) {

  cat("\n================ ITER", ii, "================\n")

  r_eta = 3
  r_sigma_diag = rep(5, 5)
  id = sample(1:5, 1)
  #r_sigma_diag[id] =  sample(5:20, 1)
  #print(r_sigma_diag)

  Longdat2 <- TAPIO::simLongData(
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

  # reshape into (subjects, time, features)
  n_samples <- length(unique(Longdat2_wide$subject))
  n_time <- length(unique(Longdat2_wide$time))
  n_feat <- 5

  X_arr <- array(NA, dim = c(n_samples, n_time, n_feat))

  for (xx in 1:n_feat) {
    for (tt in 1:n_time) {
      X_arr[, tt, xx] <- DD[seq(tt, nrow(Longdat2_wide), by = n_time), xx]
    }
  }

  # OPTIONAL missingness mask (keep if used before)
  W_arr <- array(1, dim = dim(X_arr))

  # normalize
  for (i in 1:dim(X_arr)[3]) {
    X_arr[,,i] <- scale(X_arr[,,i])
  }

  # =====================================================
  # VaDER ONLY
  # =====================================================
  cat("VaDER\n")

  res_vader <- py$run_vader(X_arr, trueClusIDs, W_arr, 50L, 100L, 4L)

  clusters <- res_vader$clusters

  ari <- ARI(trueClusIDs, as.numeric(clusters))

  cat("ARI VaDER:", ari, "\n")

  RES[ii,1] <- ari
  print(RES)
}

# =========================================================
# PLOT RESULTS
# =========================================================
RES_df <- as.data.frame(RES)
RES_m <- melt(RES_df)

colnames(RES_m) <- c("Method", "value")

ggplot(RES_m, aes(x = Method, y = value)) +
  geom_boxplot(fill = "skyblue") +
  ylim(0,1) +
  theme_minimal() +
  xlab("") +
  ylab("Adjusted Rand Index (ARI)") +
  theme(
    text = element_text(size = 14)
  )