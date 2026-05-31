simLongData_GNN_friendly <- function(
  n_total = 200,
  K = 4,
  outcomes = 5,
  eta = 2,
  n_i = 10
){

  library(MASS)

  sim_data <- list()
  subject_id <- 1

  # ---------------------------------------------
  # GLOBAL latent regime structure (VERY IMPORTANT)
  # ---------------------------------------------
  n_regimes <- 6
  regime_centers <- matrix(
    rnorm(n_regimes * outcomes, 0, 5),
    nrow = n_regimes
  )

  # regime transition probs (random walk in discrete space)
  P <- matrix(runif(n_regimes^2), n_regimes)
  P <- P / rowSums(P)

  for (k in 1:K) {

    for (i in 1:(n_total / K)) {

      # -----------------------------------------
      # subject-specific regime path
      # -----------------------------------------
      state <- sample(1:n_regimes, 1)

      times <- sort(runif(n_i, 0, 10))

      subject_bias <- rnorm(outcomes, 0, 1.5)

      for (t in 1:n_i) {

        # regime transition (Markov chain)
        state <- sample(1:n_regimes, 1, prob = P[state, ])

        for (h in 1:outcomes) {

          mu <- regime_centers[state, h]

          # -------------------------------------
          # KEY: NO smooth trajectory anymore
          # -------------------------------------
          y <- mu +
            subject_bias[h] +
            rnorm(1, 0, eta)

          sim_data[[length(sim_data) + 1]] <- data.frame(
            subject = subject_id,
            time = times[t],
            outcome = h,
            y = y,
            cluster = k
          )
        }
      }

      subject_id <- subject_id + 1
    }
  }

  do.call(rbind, sim_data)
}



simLongData_GNN_strong = function(
  n_total = 200,
  K = 4,
  outcomes = 5,
  ranTimes = TRUE,
  sigma_diag = rep(3, 5),
  eta = 3
){

  library(MASS)

  # -----------------------------
  # base mean structure (weak)
  # -----------------------------
  mean_functions <- list(
    function(t) sin(t/2),
    function(t) cos(t/3),
    function(t) 0.2*t,
    function(t) -0.1*t^2,
    function(t) log(t + 1)
  )

  # -----------------------------
  # hidden regime drift (CRITICAL)
  # -----------------------------
  regime_effect <- function(t, r){
    if (r == 1) return(  3*sin(t) )
    if (r == 2) return( -3*cos(t) )
    if (r == 3) return(  2*t )
    return(-2*t)
  }

  # -----------------------------
  # covariance
  # -----------------------------
  R <- diag(outcomes)
  Sigma_sigma <- diag(sigma_diag) %*% R %*% diag(sigma_diag)

  sim_data <- list()
  subject_id <- 1

  # -----------------------------
  # global latent "environment states"
  # (this is what FPCA CANNOT capture)
  # -----------------------------
  global_env <- function(t){
    sin(t / 2) + 0.5 * sin(3*t)
  }

  for (k in 1:K){

    for (i in 1:(n_total / K)){

      if (ranTimes){
        n_i <- sample(6:15, 1)
        times <- sort(runif(n_i, 0, 10))
      } else {
        n_i <- 12
        times <- seq(0, 10, length.out = n_i)
      }

      u_i <- mvrnorm(1, rep(0, outcomes), Sigma_sigma)

      # subject-specific coupling strength
      gamma_i <- runif(1, 0.5, 2.5)

      for (j in 1:n_i){

        t <- times[j]

        # hidden regime (nonlinear, shared structure)
        r_t <- sample(1:4, 1, prob = c(
          0.4 + 0.3*sin(t),
          0.2,
          0.2,
          0.2
        ))

        env_t <- global_env(t)

        for (h in 1:outcomes){

          # -----------------------------
          # TRUE DATA GENERATING PROCESS
          # -----------------------------
          mu <- mean_functions[[h]](t)

          # interaction term (CRUCIAL)
          coupling <- gamma_i * env_t * sin(t + k)

          # regime distortion (breaks smooth FPCA structure)
          regime <- regime_effect(t, r_t)

          y <- mu +
               u_i[h] +
               coupling +
               regime +
               rnorm(1, 0, eta * (1 + abs(env_t)))

          sim_data[[length(sim_data) + 1]] <- data.frame(
            subject = subject_id,
            time = t,
            outcome = h,
            y = y,
            cluster = k,
            regime = r_t
          )
        }
      }

      subject_id <- subject_id + 1
    }
  }

  do.call(rbind, sim_data)
}

simClinicalGraphData <- function(
    n_total = 200,
    K = 4,
    outcomes = 5,
    eta = 0.6,          # ↓ noise (CRITICAL FIX)
    n_time = 10
){

  library(MASS)

  cluster_sizes <- rep(n_total / K, K)

  sim_data <- list()
  subject_id <- 1

  # =====================================================
  # 1. STRONGER CLUSTER-SEPARATING DISEASE PROCESS
  # =====================================================

  disease_progression <- function(t, severity, k){

    cluster_shift <- c(-1.2, -0.4, 0.6, 1.3)[k]   # <<< KEY FIX

    base <- 1.8 * tanh((t - 5) / 1.8)

    base + cluster_shift + 0.5 * severity * sin(t/2)
  }

  # =====================================================
  # 2. OUTCOMES
  # =====================================================

  outcome_loadings <- list(

    function(d, t) 1.0*d + 0.2*sin(t),
    function(d, t) 1.2*d - 0.1*t,
    function(d, t) 0.8*d + 0.25*cos(t),
    function(d, t) 1.3*d + 0.04*t^2,
    function(d, t) 1.1*d - 0.1*sin(t)

  )

  # =====================================================
  # 3. STRONGER GRAPH STRUCTURE (DIRECTIONAL EFFECTS)
  # =====================================================

  coupling_matrices <- list(

    matrix(c(
      0,3,0,1,1,
      0,0,3,0,1,
      0,0,0,3,1,
      1,0,0,0,3,
      2,1,0,1,0
    ),5,5,byrow=TRUE),

    matrix(c(
      0,1,2,1,0,
      0,0,2,1,1,
      2,0,0,2,1,
      1,1,0,0,2,
      1,1,1,0,0
    ),5,5,byrow=TRUE),

    matrix(c(
      0,3,2,2,2,
      0,0,3,2,2,
      0,0,0,3,2,
      0,0,0,0,3,
      0,0,0,0,0
    ),5,5,byrow=TRUE),

    matrix(1,5,5) - diag(5)

  )

  # =====================================================
  # 4. SIMULATION LOOP
  # =====================================================

  for(k in 1:K){

    A <- coupling_matrices[[k]]

    for(i in 1:cluster_sizes[k]){

      times <- seq(0, 11, length.out = n_time)

      severity_i <- runif(1, 0.8, 1.4)

      phase_i <- runif(1, 0, 2*pi)

      u_i <- MASS::mvrnorm(
        1,
        mu = rep(0, outcomes),
        Sigma = diag(rep(0.7, outcomes))   # ↓ random effect variance
      )

      for(t in times){

        t <- t + rnorm(1, 0, 0.08)

        d <- disease_progression(t, severity_i, k)

        latent_system <- sin(t/2 + phase_i)

        raw_state <- numeric(outcomes)

        for(h in 1:outcomes){

          raw_state[h] <-
            outcome_loadings[[h]](d, t) +
            0.25 * latent_system

        }

        # =================================================
        # GRAPH COUPLING (NOW STRONG + NON-SYMMETRIC)
        # =================================================

        gamma <- 2.8   # ↑ stronger signal

        coupled_state <- raw_state

        denom <- pmax(rowSums(A), 1)

        for(h in 1:outcomes){

          # directional influence (IMPORTANT FIX)
          neighbor_signal <- sum(A[h, ] * raw_state)

          coupled_state[h] <-
            coupled_state[h] +
            gamma * neighbor_signal / denom[h]

        }

        # NO centering (critical fix)
        # → preserves separability

        # =================================================
        # OBSERVATION
        # =================================================

        for(h in 1:outcomes){

          y <- coupled_state[h] +
            u_i[h] +
            rnorm(1, 0, eta)

          sim_data[[length(sim_data) + 1]] <- data.frame(
            subject = subject_id,
            time = t,
            outcome = h,
            y = y,
            cluster = k
          )
        }
      }

      subject_id <- subject_id + 1
    }
  }

  do.call(rbind, sim_data)
}

simRelationalClusters <- function(
    n_total = 200,
    K = 4,
    outcomes = 5,
    eta = 1.5,
    ranTimes = TRUE
){

  library(MASS)

  stopifnot(outcomes == 5)

  cluster_sizes <- rep(n_total / K, K)

  sim_data <- list()
  subject_id <- 1

  # =====================================================
  # EVERYBODY SHARES SAME TRAJECTORY SHAPES
  # =====================================================

  base_fns <- list(

    function(t) 5*sin(t/2) + 0.3*t,

    function(t) 3*cos(t/3),

    function(t) 0.5*(t-5)^2 - 8,

    function(t) 2*sin(t/4) + 0.2*t^2,

    function(t) -0.8*t + 2*cos(t/2)
  )

  # =====================================================
  # CLUSTER-SPECIFIC DEPENDENCY NETWORKS
  # =====================================================

  networks <- list(

    # Cluster 1: chain
    matrix(c(
      0,1,0,0,0,
      0,0,1,0,0,
      0,0,0,1,0,
      0,0,0,0,1,
      0,0,0,0,0
    ),5,5,byrow=TRUE),

    # Cluster 2: ring
    matrix(c(
      0,0,0,0,1,
      1,0,0,0,0,
      0,1,0,0,0,
      0,0,1,0,0,
      0,0,0,1,0
    ),5,5,byrow=TRUE),

    # Cluster 3: star
    matrix(c(
      0,1,1,1,1,
      1,0,0,0,0,
      1,0,0,0,0,
      1,0,0,0,0,
      1,0,0,0,0
    ),5,5,byrow=TRUE),

    # Cluster 4: dense
    matrix(1,5,5) - diag(5)
  )

  # =====================================================
  # LATENT SHARED PROCESS
  # =====================================================

  latent_process <- function(t, phase){

    sin(t/2 + phase) +
      0.5*cos(t/3) +
      0.25*sin(2*t)

  }

  # =====================================================
  # SIMULATION LOOP
  # =====================================================

  for(k in 1:K){

    A <- networks[[k]]

    for(i in 1:cluster_sizes[k]){

      if(ranTimes){

        n_i <- sample(6:12,1)

        times <- sort(
          c(
            0,
            runif(n_i-1,0.5,11)
          )
        )

      } else {

        n_i <- 10

        times <- seq(
          0,
          11,
          length.out = n_i
        )
      }

      phase_i <- runif(
        1,
        0,
        2*pi
      )

      random_intercept <- MASS::mvrnorm(
        1,
        mu = rep(0,outcomes),
        Sigma = diag(rep(1.5,outcomes))
      )

      for(j in seq_along(times)){

        t <- times[j]

        z <- latent_process(
          t,
          phase_i
        )

        # ------------------------------------------
        # latent outcome states
        # ------------------------------------------

        latent_state <- numeric(outcomes)

        for(h in 1:outcomes){

          latent_state[h] <-
            base_fns[[h]](t) +
            z +
            rnorm(1,0,0.3)

        }

        # ------------------------------------------
        # network coupling
        # ------------------------------------------

        coupled_state <- latent_state

        gamma <- 0.8

        for(h in 1:outcomes){

          coupled_state[h] <-
            coupled_state[h] +
            gamma *
            sum(
              A[h,] *
              latent_state
            ) /
            max(
              1,
              sum(A[h,])
            )

        }

        # ------------------------------------------
        # observations
        # ------------------------------------------

        for(h in 1:outcomes){

          y <-
            coupled_state[h] +
            random_intercept[h] +
            rnorm(
              1,
              0,
              eta
            )

          sim_data[[length(sim_data)+1]] <-
            data.frame(
              subject = subject_id,
              time = t,
              outcome = h,
              y = y,
              cluster = k
            )

        }

      }

      subject_id <- subject_id + 1

    }

  }

  do.call(
    rbind,
    sim_data
  )

}


simGraphFriendlyData <- function(
    n_total = 200,
    K = 4,
    outcomes = 5,
    eta = 2,
    ranTimes = TRUE
){

  library(MASS)

  cluster_sizes <- rep(n_total / K, K)

  sim_data <- list()
  subject_id <- 1

  # =====================================================
  # OUTCOME-SPECIFIC DYNAMICS DEFINITIONS
  # =====================================================

  outcome_fns <- list(

    # Outcome 1: smooth drift (easy / FPCA-friendly)
    function(t, k){
      if(k == 1) 8*t - 0.5*t^2
      else if(k == 2) 6*t - 0.3*t^2
      else if(k == 3) 4*t - 0.2*t^2
      else 5*t
    },

    # Outcome 2: abrupt regime shift (hard for smooth models)
    function(t, k){
      base <- ifelse(t < 6, 0, -12)
      base + k * 0.5
    },

    # Outcome 3: oscillatory / cyclic
    function(t, k){
      sin(t + k) * 5 + cos(t/2) * 2
    },

    # Outcome 4: branching + curvature differences
    function(t, k){
      if(k %in% c(1,2)){
        -0.8*t
      } else {
        -2*t + 0.3*t^2
      }
    },

    # Outcome 5: noisy nonlinear interaction
    function(t, k){
      (t - 5)^2 * (ifelse(k == 4, 2, 1)) - 10 + rnorm(1, 0, 0.5)
    }
  )

  # =====================================================
  # SIMULATION LOOP
  # =====================================================

  for(k in 1:K){

    for(i in 1:cluster_sizes[k]){

      if(ranTimes){
        n_i <- sample(5:12, 1)
        times <- sort(c(0, runif(n_i - 1, 0.5, 11)))
      } else {
        n_i <- 10
        times <- seq(0, 11, length.out = n_i)
      }

      # subject warping (IMPORTANT)
      delta_i <- runif(1, -1, 1) # before -2, 2
      warped_time <- times #- delta_i

      # random effect
      u_i <- MASS::mvrnorm(
        1,
        mu = rep(0, outcomes),
        Sigma = diag(rep(2, outcomes))
      )

      for(j in seq_along(times)){
        t <- warped_time[j]

        for(h in 1:outcomes){

          mu <- outcome_fns[[h]](t, k)

          y <- mu +
               u_i[h] +
               rnorm(1, 0, eta)

          sim_data[[length(sim_data) + 1]] <- data.frame(
            subject = subject_id,
            time = times[j],
            outcome = h,
            y = y,
            cluster = k
          )
        }
      }

      subject_id <- subject_id + 1
    }
  }

  do.call(rbind, sim_data)
}

simGraphFriendlyData2 <- function(
    n_total = 200,
    K = 4,
    outcomes = 5,
    eta = 2,
    ranTimes = TRUE
){

  library(MASS)

  cluster_sizes <- rep(n_total / K, K)

  sim_data <- list()
  subject_id <- 1

  # =====================================================
  # BASE OUTCOME DYNAMICS (cluster-dependent skeletons)
  # =====================================================

  base_fns <- list(

    function(t, k){
      if(k == 1) 8*t - 0.5*t^2 else
      if(k == 2) 6*t - 0.3*t^2 else
      if(k == 3) 4*t - 0.2*t^2 else
      5*t
    },

    function(t, k){
      base <- ifelse(t < 6, 0, -12)
      base + 0.3 * k
    },

    function(t, k){
      sin(t + k) * 5 + cos(t/2) * 2
    },

    function(t, k){
      if(k %in% c(1,2)) -0.8*t else -2*t + 0.3*t^2
    },

    function(t, k){
      (t - 5)^2 * (ifelse(k == 4, 2, 1)) - 10
    }
  )

  # =====================================================
  # LATENT SHARED DYNAMICAL PROCESS (KEY ADDITION)
  # =====================================================

  latent_process <- function(t, phase){
    sin(t/2 + phase) + 0.5 * cos(t/3)
  }

  # =====================================================
  # SIMULATION LOOP
  # =====================================================

  for(k in 1:K){

    for(i in 1:cluster_sizes[k]){

      # irregular sampling
      if(ranTimes){
        n_i <- sample(5:12, 1)
        times <- sort(c(0, runif(n_i - 1, 0.5, 11)))
      } else {
        n_i <- 10
        times <- seq(0, 11, length.out = n_i)
      }

      # no global warping (IMPORTANT)
      warped_time <- times

      # subject-level random effects (structured)
      u_i <- MASS::mvrnorm(
        1,
        mu = rep(0, outcomes),
        Sigma = diag(rep(1.5, outcomes))
      )

      # latent phase per subject (critical coupling mechanism)
      phase_i <- runif(1, 0, 2*pi)

      # outcome coupling strength
      lambda <- 0.4

      for(j in seq_along(times)){
        t <- warped_time[j]

        # shared latent signal (same across outcomes)
        z <- latent_process(t, phase_i)

        for(h in 1:outcomes){

          # =================================================
          # CROSS-OUTCOME COUPLED SIGNAL (IMPORTANT CHANGE)
          # =================================================

          h_prev <- ifelse(h == 1, outcomes, h - 1)

          mu <- base_fns[[h]](t, k) +
                lambda * base_fns[[h_prev]](t, k)

          # =================================================
          # OBSERVATION MODEL (NON-SEPARABLE)
          # =================================================

          y <- mu +
               u_i[h] +
               0.8 * z +  # shared latent dynamics
               rnorm(1, 0, eta * runif(1, 0.7, 1.3))

          sim_data[[length(sim_data) + 1]] <- data.frame(
            subject = subject_id,
            time = t,
            outcome = h,
            y = y,
            cluster = k
          )
        }
      }

      subject_id <- subject_id + 1
    }
  }

  do.call(rbind, sim_data)
}

#' Plot Graph-Friendly Multivariate Longitudinal Trajectories (base R only)
#'
#' @param data Data frame from simGraphFriendlyData()
#' @return ggplot object
#'
#' @import ggplot2
#' @export

plot_graph_trajectories <- function(data){

  library(ggplot2)

  # ---------------------------------------------------
  # format outcome labels
  # ---------------------------------------------------

  data$outcome <- factor(
    data$outcome,
    labels = paste0("Y", sort(unique(data$outcome)))
  )

  data$cluster <- factor(data$cluster)

  # ---------------------------------------------------
  # compute cluster means WITHOUT dplyr
  # ---------------------------------------------------

  uniq_keys <- unique(data[, c("cluster", "outcome", "time")])

  avg_list <- vector("list", nrow(uniq_keys))

  for(i in seq_len(nrow(uniq_keys))){

    cl <- uniq_keys$cluster[i]
    ou <- uniq_keys$outcome[i]
    tm <- uniq_keys$time[i]

    idx <- which(
      data$cluster == cl &
      data$outcome == ou &
      data$time == tm
    )

    avg_list[[i]] <- data.frame(
      cluster = cl,
      outcome = ou,
      time = tm,
      mean_y = mean(data$y[idx])
    )
  }

  avg_data <- do.call(rbind, avg_list)

  # ---------------------------------------------------
  # plot
  # ---------------------------------------------------

  ggplot(data, aes(x = time, y = y, color = cluster)) +

    geom_line(
      aes(group = subject),
      alpha = 0.2,
      linewidth = 0.3
    ) +

    geom_smooth(
      method = "loess",
      se = FALSE,
      linewidth = 1.5
    ) +

    facet_wrap(~ outcome, scales = "free_y") +

    theme_minimal(base_size = 14) +

    labs(
      title = "Graph-Friendly Nonlinear Longitudinal Trajectories",
      subtitle = "Irregular + asynchronous + branching dynamics",
      x = "Time",
      y = "Outcome Value",
      color = "Cluster"
    ) +

    scale_color_brewer(palette = "Dark2") +

    theme(legend.position = "bottom")
}