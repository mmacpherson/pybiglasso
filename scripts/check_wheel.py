"""A bare wheel fits both families, diagnoses the removed numba backend,
and imports no optional extra."""

import sys

import pybiglasso as bl

d = bl.common.generate_test_data(100, 5)
print(bl.biglasso(d["X"], d["y"], nlambda=5, family="binomial"))
default = bl.biglasso(d["X"], d["X"][:, 0] + 1.25, nlambda=5)
assert default.family == "gaussian"
try:
    bl.biglasso(d["X"], d["y"], nlambda=5, backend="numba", family="binomial")
except ValueError as e:
    assert "removed" in str(e) and "0.4.0" in str(e), e
    assert 'backend="rust"' in str(e), e
else:
    raise AssertionError('backend="numba" fit after removal')
heavy = {"numba", "pandas", "polars", "sklearn", "seaborn", "matplotlib"}
assert not heavy & set(sys.modules), heavy & set(sys.modules)
