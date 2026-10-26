"""Export scikit-learn's bundled Wisconsin input data for the browser demo."""

import csv
from pathlib import Path

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
destination = (
    Path(__file__).resolve().parent.parent / "web/public/examples/breast-cancer.csv"
)
with destination.open("w", newline="") as output:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["response", *data.feature_names])
    for row, target in zip(data.data, data.target, strict=True):
        writer.writerow([int(target == 0), *(format(v, ".17g") for v in row)])
