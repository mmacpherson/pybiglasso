"""Adversarial boundaries and invalid inputs, separate from smooth optima.

Expected scores and updates are constructed algebraically; a failure here
cannot be dismissed by applying a smooth-domain metamorphism at a boundary.
"""

import tempfile
import warnings
from fractions import Fraction
from pathlib import Path

import numba
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from scipy.linalg import hadamard
from scipy.special import expit
from support.paths import assert_bitwise
from test_hypothesis import (
    BACKENDS,
    FIT_OPTIONS,
    SIGNAL,
    assert_certified_solutions,
    smooth_problems,
    theta_and_residuals,
)

import pybiglasso as bl
from pybiglasso import binomial as solver


# Boundary strategies are kept separate: they intentionally land ON strict
# comparisons, where exact-real metamorphisms do not imply identical iterates.
@st.composite
def boundary_scores(draw):
    direction = draw(st.sampled_from([-1, 0, 1]))
    power = draw(st.integers(-5, -1))
    threshold = 2.0**power
    score = (
        threshold
        if direction == 0
        else np.nextafter(threshold, -np.inf if direction < 0 else np.inf)
    )
    return threshold, draw(st.sampled_from([-1, 1])) * score, direction


@given(case=boundary_scores())
def test_kkt_scans_at_and_next_to_the_boundary(case):
    threshold, score, direction = case
    # Exact dyadic products and n=8: score = X'second*s/n with no reduction
    # ambiguity. The unused constant column exercises raw/used indexing.
    x = np.tile([-1.0, 1.0], 4)
    X = np.asfortranarray(np.column_stack([np.ones(8), x]))
    s = score * x
    for scan, initial_strong in [
        (solver.check_strong_set, 1),
        (solver.check_rest_set, 0),
    ]:
        active, strong = np.array([0]), np.array([initial_strong])
        count = scan(
            X,
            None,
            threshold,
            1.0,
            np.ones(2),
            np.array([1]),
            np.zeros(2),
            np.ones(2),
            active,
            strong,
            s,
            np.zeros(1),
            np.zeros(1),
        )
        assert count == int(direction > 0)
        assert active[0] == int(direction > 0)
        assert strong[0] == (initial_strong or direction > 0)


@pytest.mark.parametrize("backend", BACKENDS)
@given(direction=st.sampled_from([-1, 0, 1]))
def test_strong_rule_at_and_next_to_the_boundary(direction, backend):
    # Orthogonal +-1 columns and dyadic counts give EXACT initial scores
    # (1/4, 1/8). With lambda_max=1/4 and lambda=3/16, cutoff=1/8: the
    # second feature is rejected at equality (strict >), and included just
    # below. None of these numbers is obtained from a solver's output.
    X = np.asfortranarray(np.repeat(hadamard(4)[:, 1:3].astype(float), 4, axis=0))
    y = np.concatenate([np.arange(4) < count for count in (3, 1, 2, 0)]).astype(float)
    lam = 3 / 16 if direction == 0 else np.nextafter(3 / 16, direction * np.inf)
    path = bl.biglasso(X, y, lambdas=[lam], backend=backend, **FIT_OPTIONS)
    assert len(path) == 1
    assert path.n_rejected[0] == int(direction >= 0)


def reentry_problem():
    """A rejected suppressor enters only after fitting the signal column.

    x1 and q are orthogonal +-1 columns; x2=x1+2*q, scale=sqrt(5).
    The cell proportions .5 + x1/4 - q/8 give initial scores (1/4, 0).
    At lambda=3/16, pf=(1,1/16), the strong cutoff is 1/8: x1 is strong,
    x2 is rejected. Fitting x1 alone reduces its score by 1/16, so x2's
    score becomes -1/(16*sqrt(5)), beyond its L1 threshold 3/256.
    Thus the first strong scan must activate x1, and the rest scan must
    later activate x2. Both assertions follow from the constructed scores.
    """
    walsh = hadamard(4)[:, 1:3].astype(float)
    x1, q = walsh.T
    X = np.repeat(np.column_stack([x1, x1 + 2 * q]), 8, axis=0)
    counts = (4 + 2 * x1 - q).astype(int)
    y = np.concatenate([np.arange(8) < count for count in counts]).astype(float)
    return np.asfortranarray(X), y


@pytest.mark.parametrize("backend", BACKENDS)
def test_both_kkt_reentry_branches_are_needed(backend):
    X, y = reentry_problem()
    path = bl.biglasso(
        X,
        y,
        penalty_factor=[1.0, 1 / 16],
        lambdas=[3 / 16, 1 / 8],
        backend=backend,
        **FIT_OPTIONS,
    )
    assert len(path) == 2
    theta_and_residuals(path, X, y)
    assert path.coef[0, 0] > 0
    assert path.coef[1, 0] < 0


@pytest.mark.parametrize("scan", ["check_strong_set", "check_rest_set"])
def test_independent_kkt_rejects_a_disabled_reentry_scan(monkeypatch, scan):
    X, y = reentry_problem()
    monkeypatch.setattr(solver, scan, lambda *args: 0)
    path = bl.biglasso(
        X,
        y,
        penalty_factor=[1.0, 1 / 16],
        lambdas=[3 / 16],
        backend="numba",
        **FIT_OPTIONS,
    )
    with pytest.raises(AssertionError, match="KKT residual"):
        theta_and_residuals(path, X, y)


@given(side=st.sampled_from([-1.0, 1.0]), direction=st.sampled_from([-1, 0, 1]))
def test_irls_clamp_is_strict_at_ten(side, direction):
    edge = side * 10.0
    eta = np.array([edge if direction == 0 else np.nextafter(edge, direction * np.inf)])
    # The matching label keeps the loss finite even outside the clamp.
    y = np.array([float(side > 0)])
    p, w, s, r = (np.zeros(1) for _ in range(4))
    solver.update_irls(y, None, eta, p, w, s, r)
    if abs(eta[0]) > 10:
        assert p[0] == y[0]
        assert w[0] == 0.0001
        assert s[0] == r[0] == 0
    else:
        expected = expit(eta[0])
        # p=e^eta/(1+e^eta) vs expit: a few operations, not iterative error.
        np.testing.assert_allclose(p, expected, rtol=8 * np.finfo(float).eps)
        assert 0 < p[0] < 1
        assert w[0] == p[0] * (1 - p[0])


def signal_arrays(n=8):
    X = np.asfortranarray(
        np.column_stack(
            [np.tile([-1.0, -1.0, 1.0, 1.0], n // 4), np.tile([-1.0, 1.0], n // 2)]
        )
    )
    return X, (X[:, 1] > 0).astype(float)


@pytest.mark.parametrize("backend", BACKENDS)
@given(direction=st.sampled_from([-1, 0, 1]))
def test_scale_cutoff_and_neighbors(direction, backend):
    X, y = signal_arrays()
    cutoff = 1e-6
    scale = cutoff if direction == 0 else np.nextafter(cutoff, direction * np.inf)
    X[:, 1] *= scale
    fit = bl.biglasso(X, y, lambdas=[0.125, 0.0625], backend=backend, **FIT_OPTIONS)
    assert_bitwise(fit.scale[1:], np.array([scale]))
    if direction <= 0:
        assert fit.coef.nnz == 0
    else:
        assert np.all(fit.coef.toarray()[1] > 0)


@pytest.mark.parametrize("backend", BACKENDS)
@given(delta=st.sampled_from([-1, 0, 1]))
def test_default_lambda_min_switches_when_n_crosses_p(delta, backend):
    p, n = 9, 9 + delta
    x = np.resize([-1.0, 1.0], n)
    X = np.asfortranarray(np.tile(x[:, None], (1, p)))
    y = np.resize([0.0, 0.0, 0.0, 1.0], n)
    options = dict(nlambda=2, backend=backend, **FIT_OPTIONS)
    auto = bl.biglasso(X, y, **options)
    ratio = 0.001 if n > p else 0.05
    explicit = bl.biglasso(X, y, lambda_min=ratio, **options)
    assert len(auto) == 2
    # Independent of the explicit-parameter comparison: log/exp and division,
    # with a gamma_(16*n) relative allowance for their rounding.
    np.testing.assert_allclose(
        auto.lambdas[-1] / auto.lambdas[0],
        ratio,
        rtol=16 * n * np.finfo(float).eps,
        atol=0,
    )
    assert_bitwise(auto, explicit)


@pytest.mark.parametrize("backend", BACKENDS)
@given(
    dtype=st.sampled_from(["int8", "uint8", "float32"]),
    layout=st.sampled_from(["fortran", "c", "strided", "memmap"]),
)
def test_storage_and_promotion_do_not_change_the_fit(dtype, layout, backend):
    # Each dtype's edge values convert EXACTLY to float64. A float32 extreme's
    # square is finite in float64; integer extremes expose arithmetic in the
    # storage dtype (e.g. subtracting -128 from 127 before promotion).
    X, _ = signal_arrays()
    low, high = (-128, 127) if dtype == "int8" else (0, 255)
    if dtype == "float32":
        high = float(np.finfo(np.float32).max)
        low = -high
    stored = np.asfortranarray(np.where(X > 0, high, low).astype(dtype))
    y = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    promoted = bl.biglasso(
        stored.astype(float), y, lambdas=[0.125, 0.0625], backend=backend, **FIT_OPTIONS
    )
    with tempfile.TemporaryDirectory() as directory:
        if layout == "c":
            view = np.ascontiguousarray(stored)
        elif layout == "strided":
            backing = np.repeat(stored, 2, axis=0)
            view = backing[::2]
        elif layout == "memmap":
            filename = Path(directory) / "X.npy"
            np.save(filename, stored)
            view = np.load(filename, mmap_mode="r")
        else:
            view = stored
        before = view.tobytes()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The design matrix is not in Fortran"
            )
            actual = bl.biglasso(
                view, y, lambdas=[0.125, 0.0625], backend=backend, **FIT_OPTIONS
            )
        assert view.tobytes() == before
        if backend == "rust":
            # Rust's explicit LANES order also agrees across storage types.
            assert_bitwise(actual, promoted)
        else:
            # LLVM can vectorize int8, f32 and f64 reductions differently.
            np.testing.assert_array_equal(actual.iterations, promoted.iterations)
            assert_certified_solutions(actual, promoted, stored.astype(float), y)
        # Release the read-only mapping before TemporaryDirectory removes it,
        # including on Windows where an open mapping prevents unlinking.
        del view


@st.composite
def invalid_inputs(draw):
    name = draw(
        st.sampled_from(
            [
                "alpha",
                "eps",
                "lambda_min",
                "max_iter",
                "nlambda",
                "lambdas",
                "penalty_factor",
                "response",
                "design",
            ]
        )
    )
    choices = {
        "alpha": st.sampled_from([-1.0, 1.01, float("nan"), float("inf")]),
        "eps": st.sampled_from([0.0, -1.0, float("nan"), float("inf")]),
        "lambda_min": st.sampled_from([0.0, -1.0, 1.01, float("nan")]),
        "max_iter": st.sampled_from([0, -1, 1.5, True]),
        "nlambda": st.sampled_from([0, 1, 1.5, True]),
        "lambdas": st.sampled_from(
            [[], [0.1, 0.2], [-1.0], [float("nan")], [float("inf")]]
        ),
        "penalty_factor": st.sampled_from(
            [[0.0, 0.0], [-1.0, 1.0], [1.0], [1.0, float("nan")]]
        ),
        "response": st.sampled_from(["zero", "one", "nonbinary", "matrix"]),
        "design": st.sampled_from(["constant", "nonfinite", "float16", "complex"]),
    }
    return name, draw(choices[name])


@pytest.mark.parametrize("backend", BACKENDS)
@given(case=invalid_inputs())
def test_invalid_input_raises_value_error(case, backend):
    name, value = case
    X, y = signal_arrays()
    options = dict(nlambda=2, n_threads=1, backend=backend)
    if name == "response":
        y = {
            "zero": np.zeros(8),
            "one": np.ones(8),
            "nonbinary": np.full(8, 0.5),
            "matrix": y[:, None],
        }[value]
    elif name == "design":
        if value == "constant":
            X[:] = 1
        elif value == "nonfinite":
            X[0, 0] = np.inf
        else:
            X = X.astype(np.float16 if value == "float16" else complex)
    else:
        options[name] = value
    with pytest.raises(ValueError):
        bl.biglasso(X, y, **options)


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_alpha_has_its_documented_exception(backend):
    with pytest.raises(NotImplementedError, match="zero alpha"):
        bl.biglasso(*signal_arrays(), alpha=0, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dfmax_equality_vs_one_above(backend):
    X, y = signal_arrays()
    options = dict(lambdas=[0.125, 0.0625], backend=backend, **FIT_OPTIONS)
    equal = bl.biglasso(X, y, dfmax=1, **options)
    assert len(equal) == 2
    np.testing.assert_array_equal(equal.n_nonzero, [1, 1])
    with pytest.warns(UserWarning, match="Exceeded dfmax"):
        above = bl.biglasso(X, y, dfmax=0, **options)
    assert_bitwise(above, equal, length=1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("prefit", [False, True])
def test_saturation_can_return_an_empty_exportable_path(backend, prefit):
    X, y = signal_arrays()
    options = (
        dict(penalty_factor=[1.0, 0.0], nlambda=2) if prefit else dict(lambdas=[0.0])
    )
    with pytest.warns(UserWarning, match="Model saturated"):
        path = bl.biglasso(X, y, backend=backend, **options, **FIT_OPTIONS)
    assert len(path) == 0
    assert path.coef.shape == (2, 0)
    assert path.decision_function(X).shape == (8, 0)
    for kind in ("coef", "path"):
        assert len(path.to_pandas(kind)) == 0


def test_numba_restores_threads_when_standardization_fails():
    prior = numba.get_num_threads()
    requested = 2 if prior == 1 and numba.config.NUMBA_NUM_THREADS >= 2 else 1
    with pytest.raises(ValueError, match="no varying features"):
        bl.biglasso(
            np.ones((8, 2), order="F"),
            np.tile([0.0, 1.0], 4),
            n_threads=requested,
            backend="numba",
        )
    assert numba.get_num_threads() == prior


@given(delta=st.sampled_from([-1, 0, 1]))
def test_rust_row_chunk_parallelism_preserves_bits(delta):
    """The small strategies cannot reach Rust's 16,384-row parallel dispatch.

    Test both sides of that switch, including a tail, with two active columns
    so both the shift-only and fused shift/moments passes are used.
    """
    n = 16_384 + delta
    X = np.asfortranarray(
        np.column_stack(
            [
                np.resize([1.0] * 4 + [-1.0] * 4, n),
                np.resize([-1.0, 1.0], n),
            ]
        )
    )
    y = np.resize([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], n)
    options = dict(alpha=0.5, lambdas=[0.125, 0.0625], backend="rust")
    one = bl.biglasso(X, y, n_threads=1, **options)
    many = bl.biglasso(X, y, n_threads=2, **options)
    assert len(one) == 2
    assert np.all(one.n_nonzero == 2)
    assert_bitwise(one, many)
    assert_bitwise(one.center, many.center)
    assert_bitwise(one.scale, many.scale)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("compatible", [False, True])
@given(
    alpha=st.sampled_from([0.25, 0.5, 0.75, 1.0]),
    lam=st.sampled_from([0.0625, 0.125, 0.25]),
)
def test_one_active_sweep_uses_the_documented_enet_denominator(
    alpha, lam, compatible, backend
):
    """A hand calculation, including the driver's initial KKT activation.

    With x=+-1, y=(x>0), the null intercept is 0, v=1/4 and u=1/2.
    Sweep one has no active features and activates x; sweep two updates it.
    Stopping there tests the update, not agreement between converged fits.
    """
    X = np.tile([[-1.0], [1.0]], (4, 1))
    y = (X[:, 0] > 0).astype(float)
    with warnings.catch_warnings():
        with pytest.warns(UserWarning, match="Failed to converge"):
            path = bl.biglasso(
                X,
                y,
                alpha=alpha,
                lambdas=[lam],
                max_iter=2,
                r_compatible=compatible,
                n_threads=1,
                backend=backend,
            )
    l1 = Fraction(lam) * Fraction(alpha)
    l2 = Fraction(lam) * (1 - Fraction(alpha))
    v = Fraction(1, 4)
    denominator = v * (1 + l2) if compatible else v + l2
    expected = float((Fraction(1, 2) - l1) / denominator)
    assert path.iterations[0] == 2
    assert path.intercept[0] == 0
    # Dyadic sums are exact here; two possible divisions and a subtraction
    # need <=gamma_3 relative error, with room for converting Fraction to f64.
    np.testing.assert_allclose(
        path.coef.toarray(), expected, rtol=8 * np.finfo(float).eps, atol=0
    )


@pytest.mark.parametrize("backend", BACKENDS)
@given(
    problem=smooth_problems(),
    grid=st.lists(
        st.sampled_from([0.25, 0.125, 0.0625, 0.0]), min_size=1, max_size=5
    ).map(lambda values: sorted(values, reverse=True)),
)
@example(problem=SIGNAL, grid=[0.0])
@example(problem=SIGNAL, grid=[0.125, 0.125, 0.0])
def test_supplied_grids_keep_duplicates_and_singleton_zero(problem, grid, backend):
    X, y = problem.arrays()
    path = problem.fit(backend, lambdas=grid)
    assert len(path) == len(grid)
    assert_bitwise(path.lambdas, np.array(grid))
    theta_and_residuals(path, X, y)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tiny_positive_alpha_fits_finitely_or_raises_value_error(backend):
    """alpha>0 is accepted, but an unrepresentable grid must fail cleanly.

    The score is 1/4. Dividing by the smallest positive float cannot produce
    a finite lambda_max. A finite supplied grid remains a separate valid case.
    """
    X = np.array([[-1.0], [-1.0], [1.0], [1.0]] * 2, order="F")
    y = np.tile([0.0, 0.0, 0.0, 1.0], 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            path = bl.biglasso(
                X,
                y,
                alpha=np.nextafter(0.0, 1.0),
                nlambda=2,
                n_threads=1,
                backend=backend,
            )
        except ValueError:
            return
    assert np.isfinite(path.lambdas).all()
    assert np.isfinite(path.intercept).all()
    assert np.isfinite(path.coef.data).all()
