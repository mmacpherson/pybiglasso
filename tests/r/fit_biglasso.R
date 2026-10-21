# Fit R biglasso for one or more comparison cases.
#
# Usage: Rscript fit_biglasso.R CASE_DIR [CASE_DIR ...]
#
# Each CASE_DIR holds params.json, X.bin (nrow x ncol float64, column-major) and
# y.bin (ny float64; R biglasso itself subsets it by row.idx), as written by
# tests/rcompare.py. Results go to CASE_DIR/out.json.

suppressPackageStartupMessages({
  library(biglasso)
  library(jsonlite)
})

read_design <- function(case_dir, n, p, filebacked) {
  path <- file.path(case_dir, "X.bin")
  if (!filebacked) {
    X <- matrix(readBin(path, "double", n * p), n, p)
    return(as.big.matrix(X))
  }
  # -- fill a file-backed big.matrix a block of columns at a time, so that
  #    designs larger than memory never have to be materialized
  bm <- filebacked.big.matrix(
    n, p, type = "double", backingpath = case_dir,
    backingfile = "X.bm", descriptorfile = "X.desc"
  )
  con <- file(path, "rb")
  on.exit(close(con))
  block <- max(1L, floor(2^27 / (8 * n)))
  for (j in seq(1L, p, by = block)) {
    k <- min(block, p - j + 1L)
    bm[, j:(j + k - 1L)] <- matrix(readBin(con, "double", n * k), n, k)
  }
  bm
}

fit_case <- function(case_dir) {
  prm <- fromJSON(file.path(case_dir, "params.json"), simplifyVector = TRUE)
  n <- prm$nrow
  p <- prm$ncol
  X <- read_design(case_dir, n, p, isTRUE(prm$filebacked))
  y <- readBin(file.path(case_dir, "y.bin"), "double", prm$ny)

  args <- list(
    X = X, y = y, family = "binomial", screen = "SSR",
    penalty = if (prm$alpha < 1) "enet" else "lasso", alpha = prm$alpha,
    nlambda = prm$nlambda, lambda.min = prm$lambda_min,
    lambda.log.scale = prm$lambda_log_scale, eps = prm$eps,
    max.iter = prm$max_iter, ncores = 1
  )
  if (!is.null(prm$lambdas)) args$lambda <- prm$lambdas
  if (!is.null(prm$penalty_factor)) args$penalty.factor <- prm$penalty_factor
  if (!is.null(prm$row_idx)) args$row.idx <- prm$row_idx
  if (!is.null(prm$dfmax)) args$dfmax <- prm$dfmax

  warns <- character()
  elapsed <- system.time(
    fit <- withCallingHandlers(
      do.call(biglasso, args),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
  )[["elapsed"]]

  beta <- as.matrix(fit$beta)
  out <- list(
    biglasso_version = as.character(packageVersion("biglasso")),
    r_version = R.version.string,
    center = fit$center,
    scale = fit$scale,
    lambda = fit$lambda,
    loss = fit$loss,
    iter = fit$iter,
    rejections = fit$rejections,
    beta = as.vector(beta),
    beta_dim = dim(beta),
    warnings = warns,
    seconds = elapsed
  )
  writeLines(
    toJSON(out, digits = NA, auto_unbox = TRUE, na = "null"),
    file.path(case_dir, "out.json")
  )
}

# -- record failures per case rather than abandoning the whole batch
for (case_dir in commandArgs(trailingOnly = TRUE)) {
  tryCatch(fit_case(case_dir), error = function(e) {
    writeLines(
      toJSON(list(error = conditionMessage(e)), auto_unbox = TRUE),
      file.path(case_dir, "out.json")
    )
  })
}
