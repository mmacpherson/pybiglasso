# Optional competitors only; never add these to the lean fixture image.
# Use rocker/P3M as the fixture image does, but pin the snapshot explicitly.
ARG R_VERSION=4.6.1
FROM rocker/r-ver:${R_VERSION} AS bigstatsr
ARG BUILD_JOBS=2
ARG CRAN_SNAPSHOT=2026-06-24
ENV MAKEFLAGS=-j${BUILD_JOBS} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    RSPM=https://p3m.dev/cran/__linux__/noble/${CRAN_SNAPSHOT} \
    R_PROFILE_USER=/opt/r-benchmarks.Rprofile
COPY r-benchmarks.Rprofile /opt/r-benchmarks.Rprofile
RUN install2.r --repos ${RSPM} --error --ncpus ${BUILD_JOBS} bigstatsr jsonlite
RUN Rscript -e 'library(bigstatsr); cat(R.version.string, " / bigstatsr ", as.character(packageVersion("bigstatsr")), "\n")'

# The optional snpnet stage can be time-boxed independently of bigstatsr.
# GitHub-only dependencies are pinned to the original benchmark's commits.
# cindex needs the contemporary plink-ng headers, not the current ones.
FROM bigstatsr AS snpnet
ARG BUILD_JOBS=2
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl unzip zstd libzstd-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*
RUN install2.r --repos ${RSPM} --error --ncpus ${BUILD_JOBS} remotes ROCR stringr tibble tidyr data.table dplyr magrittr pgenlibr
RUN Rscript -e 'remotes::install_github("junyangq/glmnetPlus@755b0e58ee166f6d94c705c5b54ecba357eae03c", upgrade = "never")' \
 && Rscript -e 'remotes::install_github("chrchang/plink-ng@5a44c663ae5ddbe5d7ba2e5c09ff00a7534c7310", subdir = "2.0/cindex", upgrade = "never")' \
 && Rscript -e 'remotes::install_github("rivas-lab/snpnet@f7fca34569cd5c4cd83fd5c62ad8ab5b742591d9", upgrade = "never")' \
 && Rscript -e 'library(snpnet)'
# This checksum-pinned PLINK release is Linux x86-64, the benchmark platform.
RUN test "$(uname -m)" = x86_64 \
 && curl -fsSLo /tmp/plink2.zip https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_x86_64_20260919.zip \
 && echo "160f4758d54a96ee281a813eda1d0a488296fa774605b6c0587ae25ebd00557d  /tmp/plink2.zip" | sha256sum -c - \
 && unzip -q /tmp/plink2.zip plink2 -d /usr/local/bin \
 && rm /tmp/plink2.zip \
 && plink2 --version
