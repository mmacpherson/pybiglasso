"""Fit a path from the installed wheel with each backend, and check no optional
extra loaded."""

import sys

import pybiglasso as bl

d = bl.common.generate_test_data(100, 5)
for backend in ("numba", "rust"):
    print(backend, bl.biglasso(d["X"], d["y"], nlambda=5, backend=backend))
heavy = {"pandas", "polars", "sklearn", "seaborn", "matplotlib"} & set(sys.modules)
assert not heavy, heavy
