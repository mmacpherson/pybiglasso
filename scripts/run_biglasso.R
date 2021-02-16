library(tidyverse)
library(glue)
library(arrow)
library(biglasso)

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

input_file <- args[1]
if_base <- str_replace(input_file, ".feather$", "")

message(glue("Processing {input_file}"))

params <- parse_infile(input_file)
message(glue("params: {params}"))

df <- arrow::read_feather(input_file)
X <- df %>%
  select(-y) %>%
  as.matrix() %>%
  as.big.matrix()
y <- df$y

if (params[["alpha"]] < 1) {
  fit <- biglasso(X, y, family = "binomial", screen = "SSR", penalty = "enet", alpha = params[["alpha"]], nlambda = params[["nlambda"]])
} else {
  fit <- biglasso(X, y, family = "binomial", screen = "SSR", penalty = "lasso", nlambda = params[["nlambda"]])
}

df.ncol <- tibble(
  center = fit$center,
  scale = fit$scale
)

df.nlam <- tibble(
  lambda = fit$lambda,
  loss = fit$loss,
  iter = fit$iter
)

df.beta <- fit$beta %>%
  as.matrix() %>%
  as_data_frame()


arrow::write_feather(df.ncol, glue("{if_base}.r-biglasso.ncol.feather"))
arrow::write_feather(df.nlam, glue("{if_base}.r-biglasso.nlam.feather"))
arrow::write_feather(df.beta, glue("{if_base}.r-biglasso.beta.feather"))
