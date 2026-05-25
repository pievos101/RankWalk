simLongData_GNN <- function(
  n_total = 200,
  K = 4,
  outcomes = 5,
  eta = 4,
  n_i = 10
){

  library(MASS)

  # -----------------------------------------------------
  # TIME GRID
  # -----------------------------------------------------
  time_grid <- seq(0, 11, length.out = n_i)

  # -----------------------------------------------------
  # CLUSTER-SPECIFIC LATENT TRAJECTORY SHAPES
  # (IMPORTANT: weak marginal separation)
  # -----------------------------------------------------
  base_shape <- list(
    function(t) sin(t/2),
    function(t) cos(t/3),
    function(t) 0.3 * t,
    function(t) -0.2 * t + sin(t)
  )

  # -----------------------------------------------------
  # CROSS-OUTCOME MIXING MATRIX (KEY INNOVATION)
  # -----------------------------------------------------
  A_list <- list(
    matrix(c(
      1, 0.2, 0.1, 0, 0,
      0.2, 1, 0.2, 0.1, 0,
      0.1, 0.2, 1, 0.2, 0.1,
      0, 0.1, 0.2, 1, 0.2,
      0, 0, 0.1, 0.2, 1
    ), 5, 5, byrow = TRUE),

    matrix(c(
      1, 0.4, 0, 0.2, 0,
      0.4, 1, 0.3, 0, 0.1,
      0, 0.3, 1, 0.2, 0,
      0.2, 0, 0.2, 1, 0.3,
      0, 0.1, 0, 0.3, 1
    ), 5, 5, byrow = TRUE),

    matrix(c(
      1, 0, 0.3, 0.2, 0.1,
      0, 1, 0.2, 0.3, 0,
      0.3, 0.2, 1, 0.1, 0.2,
      0.2, 0.3, 0.1, 1, 0,
      0.1, 0, 0.2, 0, 1
    ), 5, 5, byrow = TRUE),

    matrix(c(
      1, 0.1, 0.2, 0.3, 0,
      0.1, 1, 0.2, 0, 0.3,
      0.2, 0.2, 1, 0.1, 0,
      0.3, 0, 0.1, 1, 0.2,
      0, 0.3, 0, 0.2, 1
    ), 5, 5, byrow = TRUE)
  )

  # -----------------------------------------------------
  # STORAGE
  # -----------------------------------------------------
  out <- list()
  id <- 1

  for (k in 1:K) {

    A <- A_list[[k]]

    for (i in 1:(n_total / K)) {

      u_i <- rnorm(outcomes, sd = 1)

      for (j in 1:n_i) {

        t <- time_grid[j]

        latent <- sapply(base_shape, function(f) f(t))

        # cluster deformation via matrix coupling
        mu <- A %*% latent

        # subject random trajectory distortion
        subject_traj <- 0.5 * sin(t + rnorm(1))

        for (h in 1:outcomes) {

          y <- mu[h] +
               subject_traj +
               u_i[h] +
               rnorm(1, sd = eta)

          out[[length(out) + 1]] <- data.frame(
            subject = id,
            time = t,
            outcome = h,
            y = y,
            cluster = k
          )
        }
      }

      id <- id + 1
    }
  }

  do.call(rbind, out)
}