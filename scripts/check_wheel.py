"""A bare wheel fits both families and imports no optional extra."""

import sys

import pybiglasso as bl

d = bl.common.generate_test_data(100, 5)
print(bl.biglasso(d["X"], d["y"], nlambda=5, family="binomial"))
default = bl.biglasso(d["X"], d["X"][:, 0] + 1.25, nlambda=5)
assert default.family == "gaussian"
heavy = {"numba", "pandas", "polars", "sklearn", "seaborn", "matplotlib"}
assert not heavy & set(sys.modules), heavy & set(sys.modules)
