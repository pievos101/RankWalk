# Also good for GNN
simLongData_hard <- function(
  n_total = 200,
  K = 4,
  outcomes = 5,
  eta = 0.8,          # easier: was 1.2
  ranTimes = TRUE,
  max_time = 10,
  missing = FALSE,
  seed = 1
) {

  #set.seed(seed)
  library(MASS)

  if (K != 4) {
    stop("This version currently assumes K = 4 clusters.")
  }

  # --------------------------------------------------
  # Cluster sizes
  # --------------------------------------------------
  cluster_sizes <- rep(floor(n_total / K), K)
  cluster_sizes[1] <- cluster_sizes[1] +
    (n_total - sum(cluster_sizes))

  # --------------------------------------------------
  # More stable regime dynamics
  # --------------------------------------------------
  trans_mats <- list(

    # Stable
    matrix(c(
      0.93, 0.05, 0.02,
      0.05, 0.90, 0.05,
      0.02, 0.05, 0.93
    ), 3, 3, byrow = TRUE),

    # Progressive
    matrix(c(
      0.82, 0.15, 0.03,
      0.10, 0.80, 0.10,
      0.03, 0.15, 0.82
    ), 3, 3, byrow = TRUE),

    # Unstable but not chaotic
    matrix(c(
      0.75, 0.15, 0.10,
      0.15, 0.70, 0.15,
      0.10, 0.15, 0.75
    ), 3, 3, byrow = TRUE),

    # Relapsing-remitting
    matrix(c(
      0.75, 0.20, 0.05,
      0.20, 0.60, 0.20,
      0.05, 0.20, 0.75
    ), 3, 3, byrow = TRUE)
  )

  # --------------------------------------------------
  # Stronger cluster separation
  # --------------------------------------------------
  regime_effects <- array(
    0,
    dim = c(K, 3, outcomes)
  )

  for (k in 1:K) {

    for (z in 1:3) {

      if (k == 1) base_mean <- -5
      if (k == 2) base_mean <- -1
      if (k == 3) base_mean <- 3
      if (k == 4) base_mean <- c(-6, 0, 6)[z]

      regime_effects[k, z, ] <- rnorm(
        outcomes,
        mean = base_mean,
        sd = 1.2
      )
    }
  }

  # --------------------------------------------------
  # Nonlinear dynamics (still present but weaker)
  # --------------------------------------------------
  nonlinear_map <- function(y_prev) {

    y_prev <- pmax(
      pmin(y_prev, 8),
      -8
    )

    out <- numeric(length(y_prev))

    for (h in seq_along(y_prev)) {

      prev <- y_prev[max(1, h - 1)]

      prev <- pmax(
        pmin(prev, 8),
        -8
      )

      out[h] <-
        sin(prev) +
        0.10 * (prev^2) / (1 + abs(prev)) -   # was 0.25
        0.10 * tanh(sum(y_prev))
    }

    pmax(
      pmin(out, 4),
      -4
    )
  }

  # --------------------------------------------------
  # Less severe missingness
  # --------------------------------------------------
  sampling_prob <- function(severity) {

    p <- plogis(
      0.5 + 0.5 * severity      # easier than before
    )

    if (is.na(p) ||
        is.nan(p) ||
        is.infinite(p)) {
      p <- 0.7
    }

    pmax(
      pmin(p, 0.98),
      0.20
    )
  }

  # --------------------------------------------------
  # Fewer shocks
  # --------------------------------------------------
  generate_shocks <- function() {

    n_shocks <- rpois(
      1,
      lambda = 0.3     # was 1
    )

    if (n_shocks == 0) {
      return(numeric(0))
    }

    sort(
      runif(
        n_shocks,
        min = 2,
        max = max_time
      )
    )
  }

  sim_data <- list()
  row_counter <- 1
  subject_id <- 1

  for (k in 1:K) {

    for (i in 1:cluster_sizes[k]) {

      if (ranTimes) {

        n_i <- sample(8:16, 1)   # slightly more observations

        times <- cumsum(
          rexp(n_i, rate = 0.45)
        )

        times <- times[
          times < max_time
        ]

        if (length(times) < 5) {
          times <- seq(
            0,
            max_time,
            length.out = 8
          )
        }

      } else {

        times <- seq(
          1,
          max_time,
          length.out = 10
        )
      }

      z <- 1

      y_prev <- rnorm(
        outcomes,
        mean = 0,
        sd = 0.5
      )

      shocks <- generate_shocks()

      for (t in times) {

        z <- sample(
          1:3,
          size = 1,
          prob = trans_mats[[k]][z, ]
        )

        shock_effect <- rep(
          0,
          outcomes
        )

        if (length(shocks) > 0) {

          for (s in shocks) {

            if (t >= s) {

              shock_effect <-
                shock_effect +
                0.5 * regime_effects[k, z, ] *
                exp(-(t - s))
            }
          }
        }

        dyn <- nonlinear_map(y_prev)

        mu <-
          regime_effects[k, z, ] +
          dyn +
          shock_effect

        y_t <- mu +
          rnorm(
            outcomes,
            mean = 0,
            sd = eta
          )

        y_t <- pmax(
          pmin(y_t, 10),
          -10
        )

        y_t[is.na(y_t)] <- 0

        severity <- mean(
          abs(y_t),
          na.rm = TRUE
        )

        if (is.na(severity) ||
            is.infinite(severity)) {
          severity <- 0
        }

        if(missing){
          obs_prob <- sampling_prob(
            severity
          )
        }else{
          obs_prob = 1
        }

        for (h in 1:outcomes) {

          if (!is.na(obs_prob) &&
              runif(1) < obs_prob) {

            sim_data[[row_counter]] <-
              data.frame(
                subject = subject_id,
                time = t,
                outcome = h,
                y = y_t[h],
                cluster = k,
                regime = z
              )

            row_counter <- row_counter + 1
          }
        }

        y_prev <- y_t
      }

      subject_id <- subject_id + 1
    }
  }

  do.call(rbind, sim_data)
}


plot_raw_vs_smooth <- function(df, max_subjects = 30){

  library(dplyr)
  library(ggplot2)

  keep <- sample(unique(df$subject),
                 min(max_subjects, length(unique(df$subject))))

  df_sub <- df %>% filter(subject %in% keep)

  ggplot(df_sub,
         aes(x = time, y = y,
             group = interaction(subject, outcome))) +

    geom_line(alpha = 0.15, color = "grey60") +

    stat_smooth(aes(group = interaction(cluster, outcome),
                    color = factor(cluster)),
                method = "loess",
                se = FALSE,
                size = 1.2) +

    facet_wrap(~outcome, scales = "free_y") +

    theme_minimal(base_size = 12) +
    labs(
      title = "Raw Trajectories + Cluster-Level Smooth Structure",
      x = "Time",
      y = "Value"
    )
}


# This works great for GNN
simLongData_CoupledTrajectories <- function(
  n_total = 200,
  K = 4,
  outcomes = 5,
  eta = 3,
  cluster_sizes = rep(n_total / K, K),
  ranTimes = TRUE,
  n_i = 10,
  sigma_diag = rep(3, outcomes)
){

  library(MASS)

  # =====================================================
  # GLOBAL DISEASE PROGRESSION
  # =====================================================

  base_fun <- function(t){
    8*t - 0.5*t^2
  }

  # =====================================================
  # SUBJECT RANDOM EFFECTS
  # =====================================================

  R <- matrix(c(
    1,0.5,0.3,0.1,0,
    0.5,1,0.2,0.1,0,
    0.3,0.2,1,0.1,0,
    0.1,0.1,0.1,1,0,
    0,0,0,0,1
  ), outcomes, outcomes)

  Sigma_sigma <- diag(sigma_diag) %*%
    R %*%
    diag(sigma_diag)

  sim_data <- list()
  subject_id <- 1

  # =====================================================
  # CLUSTER-SPECIFIC COUPLING PATTERNS
  # =====================================================

  for(k in 1:K){

    for(i in 1:cluster_sizes[k]){

      if(ranTimes){

        n_i_sub <- sample(4:12,1)

        times <- sort(
          c(
            0,
            runif(
              n_i_sub-1,
              min = 0.5,
              max = 11
            )
          )
        )

      } else {

        times <- seq(
          0,
          11,
          length.out = n_i
        )

      }

      # -----------------------------------------
      # Subject-level random outcome shifts
      # -----------------------------------------

      u_i <- MASS::mvrnorm(
        1,
        mu = rep(0, outcomes),
        Sigma = Sigma_sigma
      )

      # -----------------------------------------
      # Subject-specific latent disease severity
      # -----------------------------------------

      severity <- rnorm(
        1,
        mean = 0,
        sd = 1
      )

      for(t in times){

        # smooth subject curve
        random_curve <- generate_random_curve(t)

        latent_state <-
          base_fun(t) +
          2*severity +
          random_curve

        # =========================================
        # CLUSTER DEFINITIONS
        # =========================================

        if(k == 1){

          # all outcomes increase together

          mu <- c(
            latent_state,
            latent_state,
            latent_state,
            latent_state,
            latent_state
          )

        }

        if(k == 2){

          # outcome 1 & 2 increase
          # outcome 3 & 4 decrease

          mu <- c(
            latent_state,
            latent_state,
            -latent_state,
            -latent_state,
            latent_state
          )

        }

        if(k == 3){

          # alternating pattern

          mu <- c(
            latent_state,
            -latent_state,
            latent_state,
            -latent_state,
            latent_state
          )

        }

        if(k == 4){

          # delayed coupling

          delayed <-
            base_fun(max(t-2,0)) +
            2*severity +
            random_curve

          mu <- c(
            latent_state,
            latent_state,
            delayed,
            delayed,
            -latent_state
          )

        }

        # =========================================
        # OBSERVATIONS
        # =========================================

        for(h in 1:outcomes){

          y <-

            mu[h] +

            u_i[h] +

            rnorm(
              1,
              mean = 0,
              sd = eta
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

  sim_df <- do.call(rbind, sim_data)

  return(sim_df)
}

