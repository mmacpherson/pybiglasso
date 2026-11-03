# Gaussian out-of-core benchmark: prepare storage once, then attach it cold.
suppressPackageStartupMessages({
  library(biglasso)
  library(bigmemory)
  library(jsonlite)
})
stopifnot(as.character(packageVersion("biglasso")) == "1.7.2")
args <- commandArgs(trailingOnly = TRUE)
case_dir <- args[[1]]
mode <- if (length(args) > 1) args[[2]] else "fit"
prm <- fromJSON(file.path(case_dir, "params.json"))

if (mode == "prepare") {
  X <- filebacked.big.matrix(
    prm$nrow, prm$ncol, type = "double", backingpath = case_dir,
    backingfile = "X.bm", descriptorfile = "X.desc"
  )
  con <- file(file.path(case_dir, "X.bin"), "rb")
  block <- max(1L, floor(2^25 / (8 * prm$nrow)))
  for (j in seq(1L, prm$ncol, by = block)) {
    width <- min(block, prm$ncol - j + 1L)
    X[, j:(j + width - 1L)] <- matrix(
      readBin(con, "double", prm$nrow * width), prm$nrow, width
    )
  }
  close(con)
  quit(status = 0)
}

X <- attach.big.matrix(file.path(case_dir, "X.desc"))
stopifnot(all(dim(X) == c(prm$nrow, prm$ncol)))
y <- readBin(file.path(case_dir, "y.bin"), "double", prm$nrow)

cgroup <- function() {
  base <- "/sys/fs/cgroup"
  stat <- read.table(file.path(base, "memory.stat"), col.names = c("key", "value"))
  value <- function(key) as.numeric(stat$value[stat$key == key])
  io <- readLines("/proc/self/io")
  list(
    peak_gb = as.numeric(readLines(file.path(base, "memory.peak"))) / 1e9,
    cap_gb = as.numeric(readLines(file.path(base, "memory.max"))) / 1e9,
    current_gb = as.numeric(readLines(file.path(base, "memory.current"))) / 1e9,
    anon_gb = value("anon") / 1e9,
    swap_bytes = as.numeric(readLines(file.path(base, "memory.swap.current"))),
    refaults = value("workingset_refault_file"),
    major_faults = value("pgmajfault"),
    read_bytes = as.numeric(sub("read_bytes: +", "", grep("^read_bytes:", io, value = TRUE)))
  )
}

warns <- character()
gc()
cpu_use <- function() {
  counters <- function() {
    fields <- strsplit(trimws(readLines("/proc/stat", n = 1)), " +")[[1]]
    values <- as.numeric(fields[2:9])
    c(sum(values), sum(values[1:3]))
  }
  before <- counters()
  Sys.sleep(1)
  change <- counters() - before
  change[[2]] / change[[1]]
}
pre <- if (isTRUE(prm$recording)) cpu_use() else NULL
if (!is.null(pre) && pre > 0.15) stop("Busy pre-fit CPU bookend")
before <- cgroup()
elapsed <- system.time(
  fit <- withCallingHandlers(
    biglasso(
      X, y, family = "gaussian", screen = "SSR", penalty = "lasso", alpha = 1,
      nlambda = prm$nlambda, lambda.min = prm$lambda_min,
      lambda.log.scale = TRUE, eps = prm$eps, max.iter = prm$max_iter, ncores = 1
    ), warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  ), gcFirst = FALSE
)[["elapsed"]]
after <- cgroup()
post <- if (isTRUE(prm$recording)) cpu_use() else NULL
if (!is.null(post) && post > 0.15) stop("Busy post-fit CPU bookend")
page <- as.numeric(system("getconf PAGESIZE", intern = TRUE))
beta <- as.matrix(fit$beta)
out <- list(
  biglasso_version = as.character(packageVersion("biglasso")),
  r_version = R.version.string, center = fit$center, scale = fit$scale,
  lambda = fit$lambda, loss = fit$loss, iter = fit$iter,
  rejections = fit$rejections, beta = as.vector(beta), beta_dim = dim(beta),
  warnings = warns, seconds = elapsed,
  cpu_before = pre, cpu_after = post,
  cgroup = list(
    peak_gb = after$peak_gb, cap_gb = after$cap_gb,
    current_gb = after$current_gb, anon_gb = after$anon_gb,
    swap_bytes = after$swap_bytes,
    refaulted_gb = (after$refaults - before$refaults) * page / 1e9,
    major_faults = after$major_faults - before$major_faults,
    read_gb = (after$read_bytes - before$read_bytes) / 1e9
  )
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(case_dir, "out.json"))
