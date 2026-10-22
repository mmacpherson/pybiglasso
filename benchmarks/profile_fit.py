#!/usr/bin/env python3
"""A standard fit to profile: an elastic-net path on 5,000 x 1,000.

    make profile    # Python 3.15's sampling profiler, run on this script

The solver's driver is plain Python calling one numba kernel per step, so a
sampling profiler attributes time to each kernel by the driver line that calls
it (the kernels themselves are opaque to it). The profile covers the whole
script, imports and warm-up included; on a first run after the kernels change
it also shows numba compiling them, so run it twice.
"""

import pybiglasso as bl

data = bl.common.make_biglasso_dataset(nrow=5_000, ncol=1_000, seed=1)
bl.biglasso(data["X"], data["y"], alpha=0.5, nlambda=10)  # warm numba's cache
for _ in range(3):
    bl.biglasso(data["X"], data["y"], alpha=0.5)
