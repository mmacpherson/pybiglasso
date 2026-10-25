# R + biglasso for the comparison tests in tests/support/rcompare.py, and glmnet for
# the benchmarks (benchmarks/path.py).
#
# rocker/r-ver images install packages from a Posit Package Manager snapshot
# dated to their R release, so R_VERSION also pins biglasso:
#   4.6.1  -> biglasso 1.7.2 (the stored test fixtures)
#   latest -> current CRAN (the weekly live check)
ARG R_VERSION=4.6.1
FROM rocker/r-ver:${R_VERSION}

RUN install2.r --error biglasso glmnet jsonlite
