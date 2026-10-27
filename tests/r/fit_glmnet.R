# Fit glmnet's logistic path for the default-mode comparison.
#
# Usage: Rscript fit_glmnet.R CASE_DIR [CASE_DIR ...]
#
# Each CASE_DIR holds params.json, X.bin (nrow x ncol float64, column-major)
# and y.bin, as written by tests/support/rglmnet.py. glmnet standardizes X
# itself (population sd) and minimizes
#   -loglik / n + lambda * sum(pf * ((1 - alpha) / 2 * b^2 + alpha * |b|))
# over coefficients on standardized columns: pybiglasso's default objective.
# glmnet rescales penalty factors to sum to ncol; the caller passes them
# already so. glmnet chooses the lambda grid. Results go to CASE_DIR/out.json.

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
  pf <- if (is.null(prm$penalty_factor)) rep(1, p) else prm$penalty_factor

  # fit the whole grid: no stop on small deviance change (fdev) or on the
  # fraction of deviance explained (devmax)
  glmnet.control(fdev = 0, devmax = 1)
  on.exit(glmnet.control(factory = TRUE))
  warns <- character()
  # The generic Gaussian family keeps y in its original units. The fast
  # character-family path standardizes y, which changes the effective ridge
  # penalty. Check the documented RSS/(2n) objective without that rescaling.
  family <- if (identical(prm$family, "gaussian")) gaussian() else "binomial"
  fit <- withCallingHandlers(
    glmnet(
      X, y, family = family, alpha = prm$alpha, nlambda = prm$nlambda,
      lambda.min.ratio = prm$lambda_min_ratio, penalty.factor = pf,
      standardize = TRUE, intercept = TRUE,
      control = list(thresh = prm$thresh, maxit = prm$maxit)
    ),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  beta <- as.matrix(fit$beta)
  writeLines(
    toJSON(list(
      glmnet_version = as.character(packageVersion("glmnet")),
      r_version = R.version.string,
      lambda = fit$lambda,
      intercept = as.vector(fit$a0),
      beta = as.vector(beta),
      beta_dim = dim(beta),
      jerr = fit$jerr,
      warnings = warns
    ), digits = NA, auto_unbox = TRUE),
    file.path(case_dir, "out.json")
  )
}

# record failures per case rather than abandoning the whole batch
for (case_dir in commandArgs(trailingOnly = TRUE)) {
  tryCatch(fit_case(case_dir), error = function(e) {
    writeLines(
      toJSON(list(error = conditionMessage(e)), auto_unbox = TRUE),
      file.path(case_dir, "out.json")
    )
  })
}
