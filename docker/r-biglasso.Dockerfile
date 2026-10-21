# R + biglasso for the comparison tests in tests/rcompare.py.
#
# rocker/r-ver images install packages from a Posit Package Manager snapshot
# dated to their R release, so R_VERSION also pins biglasso:
#   4.0.3  -> biglasso 1.4.1 (the version pybiglasso was ported from)
#   latest -> current CRAN
ARG R_VERSION=4.0.3
FROM rocker/r-ver:${R_VERSION}

RUN install2.r --error biglasso jsonlite
