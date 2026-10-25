"""The Rust extension's own input checks, for callers that skip biglasso()."""

import numpy as np
import pytest

import pybiglasso as bl

_rust = pytest.importorskip("pybiglasso._rust")


def misaligned(a):
    """A copy of `a` whose elements sit one byte off their dtype's alignment."""
    buffer = bytearray(a.nbytes + 1)
    out = np.ndarray(a.shape, dtype=a.dtype, buffer=buffer, offset=1, order="F")
    out[...] = a
    assert not out.flags.aligned
    return out


def arguments(rng):
    X = np.asfortranarray(rng.normal(size=(40, 5)))
    y = (X[:, 0] + rng.normal(size=40) > 0).astype(np.float64)
    ybar = y.mean()
    return dict(
        x=X,
        y=y,
        row_indices=np.arange(0, 40, 2, dtype=np.int64),
        alpha=1.0,
        lambdas=np.array([0.1, 0.05]),
        nlambda=2,
        eps=1e-7,
        lambda_min=0.05,
        lambda_log_scale=True,
        penalty_factor=np.ones(5),
        dfmax=6,
        max_iter=1000,
        stop_on_saturation=True,
        r_compatible=False,
        ybar=ybar,
        null_deviance=-(y * np.log(ybar) + (1 - y) * np.log(1 - ybar)).sum(),
        n_threads=1,
    )


@pytest.mark.parametrize("boundscheck", [True, False])
def test_direct_call_fits(boundscheck):
    path = _rust.fit_path(
        **arguments(np.random.default_rng(0)), boundscheck=boundscheck
    )
    assert len(path["iters"]) == 2 and np.isfinite(path["deviance"]).all()


@pytest.mark.parametrize("boundscheck", [True, False])
@pytest.mark.parametrize("name", ["x", "y", "row_indices", "lambdas", "penalty_factor"])
def test_direct_call_rejects_misaligned_arrays(name, boundscheck):
    args = arguments(np.random.default_rng(0))
    args[name] = misaligned(args[name])
    with pytest.raises(ValueError, match="aligned"):
        _rust.fit_path(**args, boundscheck=boundscheck)


def test_biglasso_rejects_misaligned_x():
    args = arguments(np.random.default_rng(0))
    with pytest.raises(ValueError, match="aligned"):
        bl.biglasso(misaligned(args["x"]), args["y"], backend="rust")
