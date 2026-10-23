"""Fit a path from the installed wheel, and check no optional extra loaded."""

import sys

import pybiglasso as bl

d = bl.common.generate_test_data(100, 5)
print(bl.biglasso(d["X"], d["y"], nlambda=5))
heavy = {"pandas", "polars", "sklearn", "seaborn", "matplotlib"} & set(sys.modules)
assert not heavy, heavy
