"""Independent oracles for the partial large-offset centering repair."""

import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy.special import expit
from support.paths import assert_bitwise

import pybiglasso as bl
from pybiglasso import _rust

BASE = 2**50


def alternating(n=4096):
    return np.asfortranarray((BASE + 2 * (np.arange(n) % 2))[:, None], dtype=float)


def capture_native(monkeypatch):
    original = _rust.fit_path
    results = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(_rust, "fit_path", capture)
    return results


@pytest.mark.parametrize("compatible", [False, True])
def test_fraction_center_and_actual_returned_center_rms(compatible):
    X = alternating()
    values = [Fraction(BASE), Fraction(BASE + 2)]
    mean = sum(values) / 2
    variance = sum((v - mean) ** 2 for v in values) / 2
    assert mean == BASE + 1 and variance == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = bl.biglasso(
            X,
            np.tile([-1.0, 1.0], 2048),
            family="gaussian",
            lambdas=[0.25],
            r_compatible=compatible,
        )
    if compatible:
        # Explicit sequential binary64 arithmetic, not pybiglasso expectations.
        total = 0.0
        for value in X[:, 0]:
            total += float(value)
        center = total / len(X)
        moment = 0.0
        for value in X[:, 0]:
            moment += (float(value) - center) ** 2
        assert path.center[0] == center == BASE + 0.25
        assert path.scale[0] == np.sqrt(moment / len(X)) == 1.25
    else:
        assert path.center[0] == float(mean)
        assert path.scale[0] == float(variance)
        z = [
            (v - Fraction(float(path.center[0]))) / Fraction(float(path.scale[0]))
            for v in values
        ]
        assert z == [-1, 1] and sum(z) == 0
    assert any("too far from zero" in str(w.message) for w in caught) == compatible


def test_gaussian_analytic_optimum_and_intercept_kkt(monkeypatch):
    X = np.column_stack(
        [
            alternating()[:, 0],
            np.tile([-1.0, -1.0, 1.0, 1.0], 1024),
            np.full(4096, 1e12 + 0.1),
        ]
    )
    X = np.asfortranarray(X)
    y = np.tile([-1.0, 1.0], 2048)
    native = capture_native(monkeypatch)
    path = bl.biglasso(X, y, family="gaussian", lambdas=[0.25], eps=1e-14)
    # Orthogonal standardized columns; soft_threshold(1, 1/4) = 3/4.
    slope = Fraction(3, 4)
    np.testing.assert_array_equal(path.coef.toarray()[:, 0], [float(slope), 0.0, 0.0])
    assert path.intercept[0] == float(-Fraction(BASE + 1) * slope)
    np.testing.assert_array_equal(native[0]["col_map"], [0, 1])
    assert path.scale[2] == 0.0  # corrected constant column is dropped
    residual = y - (X @ path.coef.toarray()[:, 0] + path.intercept[0])
    z = np.column_stack([y, X[:, 1]])
    assert np.mean(residual) == 0  # intercept KKT, independently of slopes
    np.testing.assert_array_equal(z.T @ residual / len(y), [0.25, 0.0])


def test_binomial_nonseparable_stationarity(monkeypatch):
    X = alternating()
    # Among x=-1 rows, 1/4 successes; among x=+1 rows, 3/4.
    y = np.tile([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], 512)
    native = capture_native(monkeypatch)
    path = bl.biglasso(
        X,
        y,
        family="binomial",
        lambdas=[1 / 16],
        eps=1e-14,
        max_iter=10_000,
        stop_on_saturation=False,
    )
    beta = path.coef.toarray()[0, 0]
    intercept = native[0]["intercept"][0]
    # Stationarity: p(+1)=3/4-lambda=11/16, hence beta=log(11/5), a=0.
    np.testing.assert_allclose(beta, np.log(11 / 5), atol=1e-6, rtol=0)
    assert abs(intercept) < 1e-7
    z = np.tile([-1.0, 1.0], 2048)
    residual = y - expit(intercept + z * beta)
    assert abs(np.mean(residual)) < 1e-7
    assert abs(z @ residual / len(y) - 1 / 16) < 1e-7
    # The raw intercept has its own representability limit; do not use a
    # cancellation-prone raw X @ beta + a as this stationarity oracle.


@pytest.mark.parametrize(
    "offsets,changed", [([0, 4, 8, 12] * 4, True), ([0, 0, 4], False)]
)
@pytest.mark.parametrize("compatible", [False, True])
@pytest.mark.parametrize("family", ["gaussian", "binomial"])
def test_unrepresentable_center_keeps_warning(offsets, changed, compatible, family):
    base = 2**54
    X = np.array([base + x for x in offsets], dtype=float)[:, None]
    exact_values = [Fraction(float(x)) for x in X[:, 0]]
    exact_mean = sum(exact_values) / len(X)
    center = 0.0
    for x in X[:, 0]:
        center += float(x)
    center /= len(X)
    expected = center if compatible else float(exact_mean)
    with pytest.warns(UserWarning, match="too far from zero"):
        path = bl.biglasso(
            X,
            (np.arange(len(X)) % 2).astype(float),
            family=family,
            lambdas=[1.0],
            r_compatible=compatible,
        )
    assert path.center[0] == expected
    if not compatible:
        assert (path.center[0] != center) == changed
    residuals = [v - Fraction(float(path.center[0])) for v in exact_values]
    assert sum(residuals) != 0
    rms_squared = sum(r * r for r in residuals) / len(X)
    np.testing.assert_allclose(path.scale[0] ** 2, float(rms_squared), rtol=2e-15)


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("layout", ["F", "C", "memmap", "arrow"])
@pytest.mark.parametrize("n", [4096, 16384])
def test_corrected_storage_and_threads_are_bit_identical(tmp_path, family, layout, n):
    X = np.asfortranarray(
        np.column_stack([alternating(n)[:, 0], np.tile([-1.0, -1.0, 1.0, 1.0], n // 4)])
    )
    y = (
        np.tile([-1.0, 1.0], n // 2)
        if family == "gaussian"
        else np.tile([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], n // 8)
    )
    expected = bl.biglasso(X, y, family=family, lambdas=[0.125, 0.0625], n_threads=1)
    assert expected.center[0] == BASE + 1 and expected.scale[0] == 1
    if layout == "memmap":
        mapped = np.memmap(
            tmp_path / "X.bin", dtype="float64", shape=X.shape, order="F", mode="w+"
        )
        mapped[:] = X
        mapped.flush()
        X = np.memmap(
            tmp_path / "X.bin", dtype="float64", shape=X.shape, order="F", mode="r"
        )
    elif layout == "arrow":
        pa = pytest.importorskip("pyarrow")
        columns = {}
        for j in range(X.shape[1]):
            chunks = [pa.array([], type=pa.float64())]
            chunks.extend(pa.array(X[i : i + 3, j]) for i in range(0, len(X), 3))
            columns[str(j)] = pa.chunked_array(chunks)
        X = pa.table(columns)
    else:
        X = np.array(X, order=layout)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The design matrix is not in Fortran")
        for threads in (1, 2, 6):
            path = bl.biglasso(
                X, y, family=family, lambdas=[0.125, 0.0625], n_threads=threads
            )
            assert_bitwise(path, expected)
            assert_bitwise(path.center, expected.center)
            assert_bitwise(path.scale, expected.scale)


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("selection", ["mask", "unsafe-mask", "repeated"])
@pytest.mark.parametrize("compatible", [False, True])
def test_selected_row_order_and_duplicates(monkeypatch, family, selection, compatible):
    X = np.full((4098, 1), float(BASE), order="F")
    X[:4096] = alternating()
    X[-2:] = float(BASE) if selection == "mask" else 1e200
    rows = np.tile(np.arange(8), 512) if selection == "repeated" else np.arange(4096)
    y = np.resize(
        [-1.0, 1.0]
        if family == "gaussian"
        else [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        len(X),
    )
    direct = _rust.fit_path
    observed = []

    def capture(*args, **kwargs):
        result = direct(
            *args, **kwargs, row_path="gather" if selection == "repeated" else "mask"
        )
        observed.append(result["row_path"])
        return result

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Column.*too far from zero")
        expected = bl.biglasso(
            np.asfortranarray(X[rows]),
            y[rows],
            family=family,
            lambdas=[0.125],
            r_compatible=compatible,
            n_threads=1,
        )
        monkeypatch.setattr(_rust, "fit_path", capture)
        path = bl.biglasso(
            X,
            y,
            family=family,
            lambdas=[0.125],
            row_indices=rows,
            r_compatible=compatible,
            n_threads=2,
        )
    assert observed == ["mask" if selection == "mask" else "gather"]
    assert_bitwise(path, expected)
    assert_bitwise(path.center, expected.center)
    assert_bitwise(path.scale, expected.scale)
    if not compatible:
        assert path.center[0] == BASE + 1
        assert path.scale[0] == 1


@pytest.mark.r
@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("compatible", [False, True])
def test_ordinary_centering_modes_match_live_r(tmp_path, family, compatible):
    from support import rcompare

    if not rcompare.r_available():
        pytest.skip("R not available")
    # Lasso has the same objective in both modes. Large-offset fits instead
    # use independent oracles: R's raw-sum scale can fail on those inputs.
    case = rcompare.Case(100, 10, family=family, alpha=1.0, nlambda=5)
    py, _, _ = rcompare.prepare_case(case, tmp_path, r_compatible=compatible)
    (r,) = rcompare.run_r([tmp_path])
    if family == "binomial" and not compatible:
        # Default mode's documented loss is at the returned model; R stores
        # its pre-sweep loss. Evaluate R's returned coefficients independently
        # instead of expecting those two different bookkeeping values to agree.
        X = np.memmap(
            tmp_path / "X.bin",
            dtype="float64",
            shape=(case.nrow, case.ncol),
            order="F",
            mode="r",
        )
        y = np.fromfile(tmp_path / "y.bin", dtype="float64")
        beta = np.asarray(r["beta"])
        eta = X @ beta[1:] + beta[0]
        signed = (2 * y[:, None] - 1) * eta
        r = dict(r, loss=np.logaddexp(0, -signed).sum(axis=0))
    assert not (problems := rcompare.compare(py, r)), "\n".join(problems)
