library(biglasso)
library(glue)
library(microbenchmark)
library(tidyverse)

parse_infile <- function(fn) {
  parts <- str_split(fn %>% str_replace(".feather", ""), "_")[[1]]
  out <- list()
  for (e in parts[2:length(parts)]) {
    pts <- str_split(e, "-")[[1]]
    out[[pts[1]]] <- as.numeric(pts[2])
  }
  out
}

args <- commandArgs()
args <- args[8:length(args)]

X_file <- args[1]
y_file <- args[2]
nreps <- as.integer(args[3])
nlambda <- as.integer(args[4])
alpha <- as.numeric(args[5])
mmap <- as.integer(args[6])

working_dir <- dirname(X_file)
X_file_name <- basename(X_file)

if (mmap) {
  backingfile <- paste0(unlist(strsplit(X_file_name, split = "\\."))[1], ".bin")
  descriptorfile <- paste0(unlist(strsplit(X_file_name, split = "\\."))[1], ".desc")
  X <- read.big.matrix(
    filename = X_file, sep = ",", type = "double",
    backingfile = backingfile,
    backingpath = working_dir,
    descriptorfile = descriptorfile,
  )
} else {
  X <- as.big.matrix(as.matrix(read_csv(X_file, col_names = F)))
}

y <- read_csv(y_file, col_names = F)$X1

if (alpha < 1) {
  mbm <- microbenchmark(bl = biglasso(X, y,
    family = "binomial", screen = "SSR",
    penalty = "enet", alpha = alpha, nlambda = nlambda
  ), times = nreps, control = list(warmup = 0))
} else {
  mbm <- microbenchmark(bl = biglasso(X, y,
    family = "binomial", screen = "SSR",
    penalty = "lasso", nlambda = nlambda
  ), times = nreps, control = list(warmup = 0))
}

tm <- round((as_tibble(mbm) %>% summarize(mean(time)))[[1]] / 1000000, 0)

print(glue("biglasso {tm} {nrow(X)} {ncol(X)} {nlambda} {alpha} {mmap} float64"))
