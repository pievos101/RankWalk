#' Plot Multivariate Longitudinal Trajectories
#'
#' Creates a faceted \code{ggplot2} line plot of the multivariate longitudinal data
#' returned by \code{\link{simLongData}}. Each facet corresponds to a different outcome dimension,
#' and each line represents a subject's trajectory over time.
#'
#' @param data A data frame as returned by \code{\link{simLongData}}, containing columns:
#'   \code{subject}, \code{time}, \code{outcome}, \code{y}, and \code{cluster}.
#'
#' @return No return value; this function is called for its side effect: generating a plot.
#'
#' @details The plot shows individual subject trajectories, faceted by outcome variable.
#' Optionally, users can modify the function to include color coding by cluster or other aesthetics.
#'
#' @examples
#' sim_data <- simLongData()
#' plot_multivariate_trajectories(sim_data)
#'
#' @import ggplot2
#' @export


plot_multivariate_trajectories <- function(data) {
  # Assumes: columns = subject, time, outcome, y, cluster
  data$outcome <- factor(data$outcome, labels = paste0("Y", 1:5))
  data$cluster <- as.factor(data$cluster)
  
  # Compute average trajectory per cluster/outcome/time
  avg_data <- data %>%
    group_by(cluster, outcome, time) %>%
    summarise(mean_y = mean(y), .groups = 'drop')
  if (0)
    ggplot(data, aes(x = time, y = y, group = subject, color = cluster)) +
    geom_line(alpha = 0.3, linewidth = 0.3) +  # Individual trajectories
    geom_line(data = avg_data, aes(x = time, y = mean_y, group = cluster),
              linewidth = 1.2, alpha = 0.9) +  # Cluster mean
    facet_wrap(~ outcome, scales = "free_y") +
    theme_minimal(base_size = 14) +
    labs(title = "Simulated Multivariate Longitudinal Data",
         x = "Time", y = "Outcome Value") +
    scale_color_brewer(palette = "Dark2") +
    theme(legend.position = "bottom")
  
  ggplot(data, aes(x = time, y = y, color = cluster)) +
    geom_line(aes(group = subject), alpha = 0.3, linewidth = 0.3) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 1.2) +
    facet_wrap(~ factor(outcome, labels = paste0("Y", 1:5)), scales = "free_y") +
    theme_minimal(base_size = 14) +
    labs(title = "Smoothed Multivariate Longitudinal Trajectories",
         x = "Time", y = "Outcome Value") +
    scale_color_brewer(palette = "Dark2") +
    theme(legend.position = "bottom")
  
}