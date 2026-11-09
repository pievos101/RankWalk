#' Simulate Longitudinal Time Series Data for Clustering
#'
#' Generates multivariate longitudinal data with a specified number of clusters,
#' noise level, and time structure. Can simulate either fixed or random observation times.
#' A default run will create sth. VERY close to \code{\link{LongDat}}
#'
#' @param n_total Integer. Total number of subjects to simulate. Default is 200.
#' @param K Integer. Number of underlying clusters to simulate. Default is 4.
#' @param outcomes Integer. Dimensionality of the time series (number of outcomes per time point). Default is 5.
#' @param eta Numeric. Noise level in the data. Default is 3.
#' @param cluster_sizes Numeric vector. Number of subjects in each cluster. Should sum to \code{n_total}.
#'        Default is equal cluster sizes.
#' @param ranTimes Logical. If \code{TRUE}, simulate random time points per subject; otherwise, use fixed time grid.
#'        Default is \code{TRUE}.
#' @param n_i Integer. Number of fixed time points per subject (only used if \code{ranTimes = FALSE}). Default is 12.
#'
#' @return A data frame with the following columns:
#' \describe{
#'   \item{\code{subject}}{Integer. Subject ID (1 to \code{n_total}).}
#'   \item{\code{time}}{Numeric. Time point at which the observation was recorded.}
#'   \item{\code{outcome}}{Integer. Outcome dimension (1 to \code{outcomes}).}
#'   \item{\code{y}}{Numeric. Observed value of the time series at given time and outcome dimension.}
#'   \item{\code{cluster}}{Integer. True cluster assignment for the subject.}
#' }
#'
#' @details
#' For each subject, a smooth subject-specific random curve is generated based on a second-order
#' random effects model using \code{\link{generate_random_curve}}. This curve introduces structured
#' within-subject variation by modeling deviations from the cluster mean as:
#' \eqn{r_i(t) = b_{i0} + b_{i1} t + b_{i2} t^2}, where the random coefficients follow a multivariate
#' normal distribution with user-specified variance and correlation matrices.

#' @examples
#' sim_data <- simLongData()
#' head(sim_data)
#'
#' @export


simLongData = function(n_total = 200, # total subjects
                       K = 4,     # number of clusters
                       outcomes = 5, #dimensionality of time series
                       eta = 3,   # noise level; also set to 6 in some settings
                       cluster_sizes = rep(n_total / K, K),  # Balanced setting (S0)
                       ranTimes = TRUE, #random times ?
                       n_i = 10, #number of fixed time samples (only relevant if ranTimes == FALSE)
                       sigma_diag = rep(3,5)
){
  # ---- Fixed effect mean functions ----
  # Each cluster has its own mean trajectory per outcome
  # For simplicity, using some simple functional forms
  mean_functions <- list(
    # Cluster 1
    list(
      function(t) 8*t - 0.6*t^2,
      function(t) t,
      function(t) -10 + 6*t - 0.4*t^2,
      function(t) -1 + t,
      function(t) -2*t + 0.1*t^2
    ),
    # Cluster 2
    list(
      function(t) 20 - 6*t + 0.3*t^2,
      function(t) -t,
      function(t) -10 + 6*t - 0.4*t^2,
      function(t) -1 + t,
      function(t) -2*t + 0.1*t^2
    ),
    # Cluster 3
    list(
      function(t) 0,
      function(t) -7*t + 0.5*t^2,
      function(t) 0.2*t,
      function(t) -1 + t,
      function(t) -2*t + 0.1*t^2
    ),
    # Cluster 4
    list(
      function(t) 20,
      function(t) -20 + t,
      function(t) 0.2*t,
      function(t) 10 + 2*t - 0.2*t^2,
      function(t) -2*t + 0.1*t^2
    )
  )
  
  # ---- Covariance structure for random subject-level outcome effects ----
  #sigma_diag <- sample(1:10, 5, replace=TRUE) #rep(3, outcomes)
  R <- matrix(c(
    1,   0.5, 0.3, -0.1, 0,
    0.5, 1,   0.2,  0.1, 0,
    0.3, 0.2, 1,    0.1, 0,
    -0.1, 0.1, 0.1,  1,   0,
    0,   0,   0,    0,   1
  ), nrow = outcomes)
  Sigma_sigma <- diag(sigma_diag) %*% R %*% diag(sigma_diag)
  
  # ---- Simulate data ----
  sim_data <- list()
  subject_id <- 1
  
  for (k in 1:K) {
    for (i in 1:cluster_sizes[k]) {
      if (ranTimes){
        n_i <- sample(4:12, 1)
        times <- sort(c(0, sort(runif(n_i - 1, min = 0.5, max = 11))))
      } else {
        #n_i <- 12
        times <- 1:n_i #seq(0,11,length = n_i)
      }
      
      
      # Subject-specific outcome effects
      u_i <- mvrnorm(1, mu = rep(0, outcomes), Sigma = Sigma_sigma)
      
      for (j in 1:n_i) {
        t_ij <- times[j]
        random_effect_t <- generate_random_curve(t_ij)
        #browser()
        
        for (h in 1:outcomes) {
          mu <- mean_functions[[k]][[h]](t_ij)
          y_ijh <- mu + random_effect_t + u_i[h] +
            rnorm(1, mean = 0, sd = eta)  # You can change this to runif if needed
            #runif(1, min = -eta, max = eta)
            
          sim_data[[length(sim_data) + 1]] <- data.frame(
            subject = subject_id,
            time = t_ij,
            outcome = h,
            y = y_ijh,
            cluster = k
          )
        }
      }
      subject_id <- subject_id + 1
    }
  }
  
  # ---- Combine into a data frame ----
  sim_df <- do.call(rbind, sim_data)
  
  return(sim_df)
}