# Export input data, not solver expectations, from the pinned R package.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript scripts/export_browser_colon.R OUTPUT.csv")
suppressPackageStartupMessages(library(biglasso))
if (as.character(packageVersion("biglasso")) != "1.7.2") {
  stop("Use the pinned biglasso 1.7.2 image to export this example.")
}
data(colon, package = "biglasso")
stopifnot(identical(dim(colon$X), c(62L, 2000L)))
stopifnot(sum(colon$y == 1) == 40, sum(colon$y == 0) == 22)
table <- data.frame(response = colon$y, colon$X, check.names = FALSE)
write.csv(table, args[[1]], row.names = FALSE, na = "")
