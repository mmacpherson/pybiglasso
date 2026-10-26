Browser example datasets
========================

colon.csv
  Source: colon, shipped with R biglasso 1.7.2 (GPL-3.0-or-later).
  https://pbreheny.github.io/biglasso/reference/colon.html
  62 samples, 2,000 gene-expression predictors; response 1=tumor, 0=normal.
  40 tumor samples and 22 normal samples. Original study:
  Alon et al. (1999), Broad patterns of gene expression revealed by clustering
  analysis of tumor and normal colon tissues probed by oligonucleotide arrays.
  PNAS 96:6745-6750. https://doi.org/10.1073/pnas.96.12.6745
  Exported by scripts/export_browser_colon.R from the pinned R image.
  Gene identifiers and expression values are retained; response is the first
  column. No fitting, feature selection or rescaling was performed.

breast-cancer.csv
  Wisconsin Diagnostic Breast Cancer, UCI (CC BY 4.0).
  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
  Wolberg, Mangasarian, Street and Street (1995).
  https://doi.org/10.24432/C5DW2B
  569 samples, 30 predictors. Exported from scikit-learn's bundled copy by
  scripts/export_browser_breast_cancer.py. The ID column is not included.
  response 1=malignant, 0=benign (the inverse of sklearn's target encoding).

Correlated features
  Generated locally using the fixed integer generator in web/src/data.mjs.
  160 observations and 48 predictors in six correlated groups. The binary
  outcome depends on the first two groups with added noise.

These examples are for exploring a statistical algorithm, not medical use.
Their training fits do not establish performance on unseen observations.

Diabetes (442x10) and riboflavin (71x4088) are intended future examples for
a Gaussian-family solver. This demo currently fits binary logistic models.
