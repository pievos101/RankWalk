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