"""Fit a path from the installed wheel with no extras: the Rust backend fits,
the numba backend (an optional extra) explains how to get it, and no optional
extra loaded."""

import sys

import pybiglasso as bl

d = bl.common.generate_test_data(100, 5)
print(bl.biglasso(d["X"], d["y"], nlambda=5, family="binomial"))
default = bl.biglasso(d["X"], d["X"][:, 0] + 1.25, nlambda=5)
assert default.family == "gaussian"
try:
    bl.biglasso(d["X"], d["y"], nlambda=5, backend="numba", family="binomial")
except ImportError as e:
    assert "pybiglasso[numba]" in str(e), e
else:
    raise AssertionError('backend="numba" fit without the numba extra')
heavy = {"numba", "pandas", "polars", "sklearn", "seaborn", "matplotlib"}
assert not heavy & set(sys.modules), heavy & set(sys.modules)
