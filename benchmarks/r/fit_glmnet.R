# Fit glmnet's binomial or native Gaussian path for benchmarks/path.py.
#
# Usage: Rscript fit_glmnet.R CASE_DIR [CASE_DIR ...]
#
# Each CASE_DIR holds params.json (nrow, ncol, alpha, lambdas), X.bin (the raw
# design, nrow x ncol float64, column-major) and y.bin. glmnet standardizes X
# (standardize = TRUE, its default) and minimizes -loglik / n + lambda *
# ((1 - alpha) / 2 * |b|_2^2 + alpha * |b|_1) on that scale, the benchmark's
# objective, along the benchmark's lambda grid; it reports coefficients on X's
# own scale. Results go to CASE_DIR/out.json.

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

  warns <- character()
  elapsed <- system.time({
    family <- if (identical(prm$family, "gaussian")) "gaussian" else "binomial"
    alpha <- prm$alpha
    factor <- 1
    response_scale <- NULL
    if (identical(family, "gaussian")) {
      response_scale <- sqrt(mean((y - mean(y))^2))
      if (isTRUE(prm$generic)) {
        # Correctness reference only; never selected by the timing runner.
        family <- gaussian()
      } else {
        # Native glmnet fits y/s and returns coefficients in original units.
        # Its raw-scale L1 and L2 weights are lambda'*alpha' and
        # lambda'*(1-alpha')/s. Match lambda*alpha and lambda*(1-alpha).
        factor <- alpha + (1 - alpha) * response_scale
        alpha <- alpha / factor
      }
    }
    options <- list()
    if (!is.null(prm$thresh)) {
      options <- list(control = list(thresh = prm$thresh))
    }
    fit <- withCallingHandlers(do.call(glmnet, c(list(
      x = X, y = y, family = family,
      alpha = alpha, lambda = prm$lambdas * factor,
      standardize = TRUE, intercept = TRUE
    ), options)), warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  })[["elapsed"]]

  beta <- as.matrix(fit$beta)
  writeLines(
    toJSON(list(
      glmnet_version = as.character(packageVersion("glmnet")),
      r_version = R.version.string,
      lambda = fit$lambda / factor,
      glmnet_alpha = alpha,
      glmnet_lambda_factor = factor,
      response_scale = response_scale,
      glmnet_family = if (is.list(family)) "gaussian()" else family,
      intercept = as.vector(fit$a0),
      beta = as.vector(beta),
      beta_dim = dim(beta),
      warnings = warns,
      jerr = fit$jerr,
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
