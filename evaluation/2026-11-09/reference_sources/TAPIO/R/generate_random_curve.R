#' Generate Subject-Specific Random Curves
#'
#' Computes random subject-level time effects based on a quadratic random effects model:
#' \eqn{r_i(t) = b_{i0} + b_{i1} t + b_{i2} t^2}, as defined in Equation (9) of Zhou et al. (2022).
#'
#' This function is used internally by \code{\link{simLongData}} to generate smooth, individual-specific
#' deviations from cluster-level means.
#'
#' @param times A numeric vector of time points at which to evaluate the subject-specific random curve.
#' @param sigma_b A 3x3 diagonal variance matrix (default: \code{diag(c(2, 0.3, 0.06))}) specifying the marginal variances of the random coefficients \eqn{b_{i0}}, \eqn{b_{i1}}, and \eqn{b_{i2}}.
#' @param rho_b A 3x3 correlation matrix specifying the correlation structure among the random coefficients, as defined in Equation (10) of Zhou et al. (2022).
#'
#' @return A numeric vector of the same length as \code{times}, representing the subject-specific time effect evaluated at each time point.
#'
#' @references
#' Zhou, J., Zhang, Y., & Tu, W. (2022). clusterMLD: An Efficient Hierarchical Clustering Method for Multivariate Longitudinal Data. *Journal of Computational and Graphical Statistics*, DOI: \doi{10.1080/10618600.2022.2149540}
#'
#' @examples
#' times <- seq(0, 1, length.out = 10)
#' rho_b <- diag(3) # identity correlation matrix for simplicity
#' curve <- generate_random_curve(times, rho_b = rho_b)
#'
#' @export



generate_random_curve <- function(times, # time to evaluate at
                                  sigma_b = diag(c(2, 0.3, 0.06)), #diagonal variance matrix
                                  rho_b # correlation coefficient matrix, defined in Eq. (10) from the same paper
) {
  # Random time effect function: ri(t) = bi0 + bi1*t + bi2*t^2 

  if (missing(rho_b)) {
    # As per Section 3.2
    rho_b <- matrix(c(
      1.0,  0.4, -0.3,
      0.4,  1.0, -0.2,
      -0.3, -0.2, 1.0
    ), nrow = 3, byrow = TRUE)
  }
  # Full covariance matrix
  Sigma_b <- sigma_b %*% rho_b %*% sigma_b
  b <- mvrnorm(1, mu = rep(0, 3), Sigma = Sigma_b)
  b[1] + b[2]*times + b[3]*times^2
}
