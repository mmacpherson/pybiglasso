# Fit snpnet's logistic path for benchmarks/path.py.
#
# Usage: Rscript fit_snpnet.R CASE_DIR [CASE_DIR ...]
#
# Each CASE_DIR holds params.json (nrow, ncol, lambdas, sds, ncores), X.bed,
# X.bim and X.fam (the dosages, as plink 1 files) and pheno.tsv. snpnet reads
# genotypes through plink2, so the .bed is first converted to its .pgen format
# (untimed, as R biglasso's big.matrix is built untimed). snpnet fits glmnet
# (its glmnetPlus fork) on batches of variants, on the dosages as they are:
# its standardize.variant option stops in computeProduct (apply() on a
# vector). Penalizing each variant by its standard deviation instead gives
# the lasso on standardized variants, the benchmark's objective, exactly; the
# elastic net's ridge term would not scale the same way, so this is lasso
# only. snpnet reads penalty factors only alongside a covariate (without one,
# it drops the first variant's), so pheno.tsv carries a covariate of zeros:
# unpenalized, and left out by glmnet as constant. Results go to
# CASE_DIR/out.json.

suppressPackageStartupMessages({
  library(snpnet)
  library(jsonlite)
})

fit_case <- function(case_dir) {
  prm <- fromJSON(file.path(case_dir, "params.json"), simplifyVector = TRUE)
  stopifnot(prm$family == "binomial")
  pfile <- file.path(case_dir, "X")
  status <- system2(
    "plink2", c("--bfile", pfile, "--make-pgen", "vzs", "--out", pfile,
                "--threads", prm$ncores, "--silent")
  )
  if (status != 0) stop("plink2 --make-pgen failed")
  results_dir <- file.path(case_dir, "snpnet")
  # snpnet names each variant's column <ID>_<ALT>
  ids <- sprintf("v%d", seq_len(prm$ncol))
  columns <- paste0(ids, "_A")

  observed <- character()
  elapsed <- system.time(
    fit <- withCallingHandlers(snpnet(
      genotype.pfile = pfile,
      phenotype.file = file.path(case_dir, "pheno.tsv"),
      phenotype = "y",
      covariates = "zero",
      family = "binomial",
      alpha = 1,
      lambda = prm$lambdas,
      p.factor = setNames(prm$sds, columns),
      configs = list(
        standardize.variant = FALSE, glmnet.thresh = prm$eps,
        # keep every variant, as the other solvers do
        MAF.thresh = 0, missing.rate = 1,
        nCores = prm$ncores,
        # the default 50 rounds of adding variants can stop short of the end
        # of the path
        niter = 10000,
        results.dir = results_dir
      )
    ), warning = function(w) {
      observed <<- c(observed, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  )[["elapsed"]]

  # one coefficient vector per fitted lambda, of the variants it has added
  beta <- sapply(fit$beta, function(b) {
    out <- setNames(numeric(prm$ncol), columns)
    b <- b[names(b) %in% columns]
    out[names(b)] <- b
    out
  })
  writeLines(
    toJSON(list(
      snpnet_version = as.character(packageVersion("snpnet")),
      r_version = R.version.string,
      intercept = unname(sapply(fit$a0, function(a) a[[1]])),
      beta = as.vector(beta),
      beta_dim = dim(beta),
      # Seventeen significant digits round-trip the supplied binary64 grid;
      # jsonlite's numeric formatting can otherwise drop last-bit precision.
      lambdas = format(fit$full.lams, digits = 17, scientific = TRUE, trim = TRUE),
      warnings = observed,
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
