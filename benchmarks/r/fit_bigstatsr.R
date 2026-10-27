# End-to-end CMSA fit for benchmarks/cv.py. Attaching the shared FBM and
# exporting its averaged model are outside the clock.
suppressPackageStartupMessages({ library(bigstatsr); library(jsonlite) })

fit_case <- function(case_dir) {
  prm <- fromJSON(file.path(case_dir, "params.json"))
  stopifnot(prm$family == "binomial")
  X <- FBM(prm$nrow, prm$ncol, type = "double",
           backingfile = file.path(case_dir, "X_train"), create_bk = FALSE)
  y <- readBin(file.path(case_dir, "y_train.bin"), "double", prm$nrow)
  sets <- scan(file.path(case_dir, "folds.txt"), quiet = TRUE, what = integer())
  stopifnot(length(sets) == length(y), setequal(unique(sets), seq_len(prm$folds)))
  # Fold workers inherit these limits; BLAS must not compete with ncores.
  Sys.setenv(OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1", OMP_NUM_THREADS = "1")
  bigparallelr::set_blas_ncores(1)
  set.seed(prm$seed)
  observed <- character()
  elapsed <- system.time(
    fit <- withCallingHandlers(
      big_spLogReg(X, y, K = prm$folds, ind.sets = sets,
                   alphas = prm$alpha, ncores = prm$ncores,
                   nlambda = prm$nlambda, nlam.min = prm$nlam_min,
                   n.abort = if (prm$early_stopping) 10L else .Machine$integer.max,
                   lambda.min.ratio = prm$lambda_min, eps = 1e-5, max.iter = 1000L),
      warning = function(w) {
        observed <<- c(observed, conditionMessage(w))
        invokeRestart("muffleWarning")
      })
  )[["elapsed"]]
  best <- summary(fit)
  stopifnot(nrow(best) == 1L)
  out <- list(
    bigstatsr_version = as.character(packageVersion("bigstatsr")),
    r_version = R.version.string,
    seconds = elapsed, intercept = best$intercept[[1]], beta = best$beta[[1]],
    ind_col = attr(fit, "ind.col"),
    # summary's all_conv ONLY means all messages are "No more improvement".
    # Finishing the requested grid is not an optimization failure.
    all_early_stopped = best$all_conv[[1]],
    fold_lambdas = sapply(fit[[1]], function(f) length(f$lambda)),
    fold_max_iterations = sapply(fit[[1]], function(f) max(f$iter)),
    fold_messages = sapply(fit[[1]], function(f) f$message), max_iter = 1000L,
    probe_logits = as.numeric(predict(fit, X, ind.row = seq_len(min(8L, prm$nrow)), proba = FALSE)),
    warnings = observed, model_kind = "average of selected fold models",
    n_abort = if (prm$early_stopping) 10L else .Machine$integer.max,
    nlam_min = prm$nlam_min, nlambda = prm$nlambda,
    lambda_min = prm$lambda_min, eps = 1e-5
  )
  writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(case_dir, "out.json"))
}

for (case_dir in commandArgs(trailingOnly = TRUE)) {
  tryCatch(fit_case(case_dir), error = function(e) {
    writeLines(toJSON(list(error = conditionMessage(e)), auto_unbox = TRUE),
               file.path(case_dir, "out.json"))
  })
}
