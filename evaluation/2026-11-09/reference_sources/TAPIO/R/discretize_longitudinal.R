#' Discretize Irregular Longitudinal Data into Time Windows
#'
#' Converts irregularly sampled longitudinal observations into a common
#' discrete time representation by aggregating observations within fixed
#' time windows. Empty windows can optionally be filled by interpolation.
#'
#' The function can either construct window boundaries from the supplied
#' data or use previously defined boundaries. The latter is useful for
#' inductive applications, where window boundaries estimated from a
#' reference cohort must be kept fixed when processing new subjects.
#'
#' @param DATA A data.frame or matrix containing longitudinal measurements.
#'   Rows correspond to observations and columns to variables.
#'
#' @param user_id Vector identifying the subject corresponding to each row
#'   of \code{DATA}. Must have length \code{nrow(DATA)}.
#'
#' @param time Numeric vector containing the observation time corresponding
#'   to each row of \code{DATA}. Must have length \code{nrow(DATA)}.
#'
#' @param n_windows Number of equally spaced time windows. Ignored if
#'   \code{breaks} is supplied.
#'
#' @param breaks Optional numeric vector defining the window boundaries.
#'   If supplied, these boundaries are used instead of estimating them from
#'   the data. This is recommended when applying a fitted preprocessing
#'   scheme to new subjects.
#'
#' @param FUN Function used to aggregate multiple observations from the
#'   same subject within the same time window. Default is \code{mean}.
#'
#' @param fill_empty Logical. If \code{TRUE}, empty subject-specific windows
#'   are filled by linear interpolation. Default is \code{TRUE}.
#'
#' @param extrapolate Logical. If \code{TRUE}, values before the first and
#'   after the last observed window are filled using the nearest available
#'   value. If \code{FALSE}, they remain \code{NA}. Default is \code{TRUE}.
#'
#' @return An object of class \code{"TAPIO_discretized"} containing:
#' \describe{
#'   \item{DATA}{Discretized longitudinal data.}
#'   \item{user_id}{Subject identifier corresponding to each output row.}
#'   \item{time}{Window centers corresponding to each output row.}
#'   \item{window}{Integer window index.}
#'   \item{breaks}{Window boundaries used for discretization.}
#'   \item{centers}{Window centers.}
#'   \item{n_windows}{Number of windows.}
#'   \item{empty_fraction}{Fraction of subject-windows that were empty
#'     before interpolation.}
#' }
#'
#' @examples
#' \dontrun{
#'
#' # Reference cohort
#' disc_train <- discretize_longitudinal(
#'   DATA = X_train,
#'   user_id = id_train,
#'   time = time_train,
#'   n_windows = 10
#' )
#'
#' fit <- longTAPIO_inductive(
#'   DATA = disc_train$DATA,
#'   user_id = disc_train$user_id,
#'   k = 4
#' )
#'
#' # New patients: use exactly the same windows
#' disc_test <- discretize_longitudinal(
#'   DATA = X_test,
#'   user_id = id_test,
#'   time = time_test,
#'   breaks = disc_train$breaks
#' )
#' }
#'
#' @export
discretize_longitudinal <- function(
  DATA,
  user_id,
  time,
  n_windows = 10,
  breaks = NULL,
  FUN = mean,
  fill_empty = TRUE,
  extrapolate = TRUE
) {

  # ==========================================================================
  # 1. INPUT CHECKS
  # ==========================================================================

  DATA <- as.data.frame(DATA)

  if (nrow(DATA) != length(user_id)) {
    stop(
      "length(user_id) must equal nrow(DATA)."
    )
  }

  if (nrow(DATA) != length(time)) {
    stop(
      "length(time) must equal nrow(DATA)."
    )
  }

  if (!is.numeric(time)) {
    stop(
      "'time' must be numeric."
    )
  }

  if (any(!is.finite(time))) {
    stop(
      "'time' must contain only finite values."
    )
  }

  if (ncol(DATA) < 1L) {
    stop(
      "'DATA' must contain at least one measurement variable."
    )
  }

  # Require numeric longitudinal variables
  is_num <- vapply(
    DATA,
    is.numeric,
    logical(1)
  )

  if (!all(is_num)) {
    stop(
      "All columns of 'DATA' must be numeric."
    )
  }

  if (anyDuplicated(names(DATA))) {
    stop(
      "Column names in 'DATA' must be unique."
    )
  }

  # ==========================================================================
  # 2. CONSTRUCT OR VALIDATE WINDOW BOUNDARIES
  # ==========================================================================

  if (is.null(breaks)) {

    if (
      length(n_windows) != 1L ||
      !is.numeric(n_windows) ||
      !is.finite(n_windows) ||
      n_windows < 1 ||
      n_windows != as.integer(n_windows)
    ) {
      stop(
        "'n_windows' must be a positive integer."
      )
    }

    n_windows <- as.integer(n_windows)

    time_min <- min(time)
    time_max <- max(time)

    if (time_min == time_max) {
      stop(
        "At least two distinct time values are required."
      )
    }

    breaks <- seq(
      time_min,
      time_max,
      length.out = n_windows + 1L
    )

  } else {

    if (!is.numeric(breaks)) {
      stop(
        "'breaks' must be numeric."
      )
    }

    if (length(breaks) < 2L) {
      stop(
        "'breaks' must contain at least two values."
      )
    }

    if (any(!is.finite(breaks))) {
      stop(
        "'breaks' must contain only finite values."
      )
    }

    if (is.unsorted(breaks, strictly = TRUE)) {
      stop(
        "'breaks' must be strictly increasing."
      )
    }

    n_windows <- length(breaks) - 1L
  }

  centers <- (
    breaks[-1L] +
    breaks[-length(breaks)]
  ) / 2


  # ==========================================================================
  # 3. CHECK WHETHER OBSERVATIONS ARE INSIDE THE WINDOW RANGE
  # ==========================================================================

  outside <- (
    time < breaks[1L] |
    time > breaks[length(breaks)]
  )

  if (any(outside)) {

    warning(
      sum(outside),
      " observation(s) fall outside the supplied time-window range ",
      "and will be ignored."
    )
  }

  keep <- !outside

  DATA_use <- DATA[keep, , drop = FALSE]
  id_use <- user_id[keep]
  time_use <- time[keep]


  # ==========================================================================
  # 4. SUBJECT INFORMATION
  # ==========================================================================

  ids <- unique(user_id)

  n_subjects <- length(ids)
  n_features <- ncol(DATA)

  if (n_subjects < 1L) {
    stop(
      "No subjects found."
    )
  }


  # ==========================================================================
  # 5. ASSIGN OBSERVATIONS TO WINDOWS
  # ==========================================================================

  window_id <- cut(
    time_use,
    breaks = breaks,
    include.lowest = TRUE,
    right = FALSE,
    labels = FALSE
  )

  # cut(..., right = FALSE) excludes the final right boundary.
  # Explicitly assign observations at the maximum boundary to the final window.
  at_upper_boundary <- (
    time_use == breaks[length(breaks)]
  )

  window_id[at_upper_boundary] <- n_windows


  # ==========================================================================
  # 6. CREATE SUBJECT x WINDOW x FEATURE ARRAY
  # ==========================================================================

  X <- array(
    NA_real_,
    dim = c(
      n_subjects,
      n_windows,
      n_features
    ),
    dimnames = list(
      as.character(ids),
      paste0(
        "window_",
        seq_len(n_windows)
      ),
      names(DATA)
    )
  )


  # ==========================================================================
  # 7. AGGREGATE OBSERVATIONS WITHIN WINDOWS
  # ==========================================================================

  for (i in seq_len(n_subjects)) {

    subject_rows <- which(
      id_use == ids[i]
    )

    if (length(subject_rows) == 0L) {
      next
    }

    subject_windows <- window_id[
      subject_rows
    ]

    for (w in seq_len(n_windows)) {

      idx <- subject_rows[
        subject_windows == w
      ]

      if (length(idx) == 0L) {
        next
      }

      for (j in seq_len(n_features)) {

        values <- DATA_use[
          idx,
          j,
          drop = TRUE
        ]

        values <- values[
          is.finite(values)
        ]

        if (length(values) > 0L) {

          X[i, w, j] <- FUN(
            values,
            na.rm = TRUE
          )
        }
      }
    }
  }


  # ==========================================================================
  # 8. RECORD EMPTY WINDOWS BEFORE IMPUTATION
  # ==========================================================================

  empty_windows <- matrix(
    FALSE,
    nrow = n_subjects,
    ncol = n_windows
  )

  for (i in seq_len(n_subjects)) {

    for (w in seq_len(n_windows)) {

      empty_windows[i, w] <- all(
        !is.finite(
          X[i, w, ]
        )
      )
    }
  }

  empty_fraction <- mean(
    empty_windows
  )


  # ==========================================================================
  # 9. INTERPOLATE EMPTY VALUES
  # ==========================================================================

  if (fill_empty) {

    approx_rule <- if (
      extrapolate
    ) {
      2
    } else {
      1
    }

    for (i in seq_len(n_subjects)) {

      for (j in seq_len(n_features)) {

        y <- X[i, , j]

        ok <- is.finite(y)

        n_available <- sum(ok)

        # --------------------------------------------------------------------
        # At least two observations:
        # linear interpolation
        # --------------------------------------------------------------------

        if (n_available >= 2L) {

          X[i, , j] <- approx(
            x = centers[ok],
            y = y[ok],
            xout = centers,
            rule = approx_rule,
            ties = "ordered"
          )$y

        # --------------------------------------------------------------------
        # Exactly one observation:
        # only propagate if extrapolation was explicitly requested
        # --------------------------------------------------------------------

        } else if (
          n_available == 1L &&
          extrapolate
        ) {

          X[i, , j] <- rep(
            y[ok][1L],
            n_windows
          )
        }
      }
    }
  }


  # ==========================================================================
  # 10. CONVERT BACK TO LONG FORMAT
  # ==========================================================================

  n_rows <- n_subjects * n_windows

  DATA_out <- matrix(
    NA_real_,
    nrow = n_rows,
    ncol = n_features
  )

  colnames(DATA_out) <- names(DATA)

  id_out <- rep(
    ids,
    each = n_windows
  )

  time_out <- rep(
    centers,
    times = n_subjects
  )

  window_out <- rep(
    seq_len(n_windows),
    times = n_subjects
  )

  row_counter <- 1L

  for (i in seq_len(n_subjects)) {

    rows <- row_counter:(
      row_counter +
      n_windows -
      1L
    )

    DATA_out[rows, ] <- X[i, , ]

    row_counter <- row_counter +
      n_windows
  }

  DATA_out <- as.data.frame(
    DATA_out
  )


  # ==========================================================================
  # 11. OUTPUT
  # ==========================================================================

  out <- list(

    DATA = DATA_out,

    user_id = id_out,

    time = time_out,

    window = window_out,

    breaks = breaks,

    centers = centers,

    n_windows = n_windows,

    empty_fraction = empty_fraction,

    empty_windows = empty_windows,

    array = X,

    call = match.call()
  )

  class(out) <- "TAPIO_discretized"

  out
}