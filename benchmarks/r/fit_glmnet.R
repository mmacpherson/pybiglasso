# Fit glmnet's logistic path for benchmarks/path.py.
#
# Usage: Rscript fit_glmnet.R CASE_DIR [CASE_DIR ...]
#
# Each CASE_DIR holds params.json (nrow, ncol, alpha, lambdas), X.bin (the
# standardized design, nrow x ncol float64, column-major) and y.bin. glmnet
# minimizes -loglik / n + lambda * ((1 - alpha) / 2 * |b|_2^2 + alpha * |b|_1),
# the benchmark's objective, so X goes in as is (standardize = FALSE) with the
# benchmark's lambda grid. Results go to CASE_DIR/out.json.

suppressPackageStartupMessages({
  library(glmnet)
  library(jsonlite)
})

fit_case <- function(case_dir) {
  prm <- fromJSON(file.path(case_dir, "params.json"), simplifyVector = TRUE)
  n <- prm$nrow
  p <- prm$ncol
  X <- matrix(readBin(file.path(case_dir, "X.bin"), "double", n * p), n, p)
  y <- readBin(file.path(case_dir, "y.bin"), "double", n)

  elapsed <- system.time(
    fit <- glmnet(
      X, y, family = "binomial", alpha = prm$alpha, lambda = prm$lambdas,
      standardize = FALSE, intercept = TRUE
    )
  )[["elapsed"]]

  beta <- as.matrix(fit$beta)
  writeLines(
    toJSON(list(
      glmnet_version = as.character(packageVersion("glmnet")),
      r_version = R.version.string,
      lambda = fit$lambda,
      intercept = as.vector(fit$a0),
      beta = as.vector(beta),
      beta_dim = dim(beta),
      seconds = elapsed
    ), digits = NA, auto_unbox = TRUE),
    file.path(case_dir, "out.json")
  )
}

for (case_dir in commandArgs(trailingOnly = TRUE)) {
  tryCatch(fit_case(case_dir), error = function(e) {
    writeLines(
      toJSON(list(error = conditionMessage(e)), auto_unbox = TRUE),
      file.path(case_dir, "out.json")
    )
  })
}
