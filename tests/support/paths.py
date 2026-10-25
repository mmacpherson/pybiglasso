"""Two distinct path contracts: identical bytes, or a rounded same trajectory.

Neither is an optimum-only comparison. Permuted coordinates and independently
converged solvers instead need the residual/curvature checks in certificates.
"""

import numpy as np

PATH_FIELDS = ("lambdas", "intercept", "deviance", "iterations", "n_rejected")
# Existing fixture policy, not a theorem or an error bound. Iterations and
# rejections remain exact; the known fixtures stay away from discrete ties.
SAME_TRAJECTORY = dict(rtol=1e-9, atol=1e-12)


def _assert_array_bits(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def assert_bitwise(a, b, *, length=None, rows=None, coef_b=None):
    """Compare arrays or path trajectories by shape, dtype and bytes.

    For paths, length explicitly requests a prefix of b; otherwise both paths
    must have the same length. rows projects b's coefficient rows, and coef_b
    supplies an algebraically transformed target (e.g. power-of-two scaling).
    Coefficients are compared densely so sparse storage layout is immaterial.
    Center/scale and other metadata are checked separately when relevant.
    """
    if isinstance(a, np.ndarray):
        assert length is None and rows is None and coef_b is None
        _assert_array_bits(a, b)
        return
    length = len(b) if length is None else length
    assert len(a) == length and len(b) >= length
    for field in PATH_FIELDS:
        _assert_array_bits(getattr(a, field), getattr(b, field)[:length])
    coef = b.coef.toarray()[:, :length] if coef_b is None else coef_b
    _assert_array_bits(a.coef.toarray(), coef if rows is None else coef[rows])


def assert_same_trajectory(a, b, *, coef_b=None, intercept_b=None, deviance_b=None):
    """Exact discrete trajectory, with the existing rounded-value policy."""
    assert len(a) == len(b)
    np.testing.assert_array_equal(a.iterations, b.iterations)
    np.testing.assert_array_equal(a.n_rejected, b.n_rejected)
    # Preserve the original lambda comparison, including numpy's default atol.
    np.testing.assert_allclose(a.lambdas, b.lambdas, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(
        a.coef.toarray(),
        b.coef.toarray() if coef_b is None else coef_b,
        **SAME_TRAJECTORY,
    )
    np.testing.assert_allclose(
        a.intercept,
        b.intercept if intercept_b is None else intercept_b,
        **SAME_TRAJECTORY,
    )
    np.testing.assert_allclose(
        a.deviance,
        b.deviance if deviance_b is None else deviance_b,
        **SAME_TRAJECTORY,
    )
