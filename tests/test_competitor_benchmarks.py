"""Benchmark protocols, serialization and negative controls; no R required."""

import importlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    loaded = tuple(
        importlib.import_module(name) for name in ("cv", "snpnet", "competitors")
    )
    # Exercise POSIX container command construction on Windows CI too.
    monkeypatch.setattr(loaded[2].os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(loaded[2].os, "getgid", lambda: 1000, raising=False)
    return loaded


def test_plink_dosages_are_encoded_exactly_including_tail_padding(modules, tmp_path):
    _, snpnet, _ = modules
    X = np.array([[0, 2], [1, 1], [2, 0], [0, 2], [2, 0]])
    y = np.array([0, 1, 0, 1, 1])
    snpnet.write_plink1(X, y, tmp_path / "X")
    # BED 00=two A, 10=one A, 11=zero A; little-endian sample quads.
    assert (tmp_path / "X.bed").read_bytes() == bytes.fromhex("6c1b01 cb00 3803")
    assert (tmp_path / "X.bim").read_text().splitlines() == [
        "1\tv1\t0\t1\tA\tC",
        "1\tv2\t0\t2\tA\tC",
    ]
    assert (tmp_path / "pheno.tsv").read_text().splitlines()[-1] == "s5\ts5\t2\t0"


@pytest.mark.parametrize("invalid", [-1.0, 0.5, 3.0, np.nan])
def test_plink_writer_does_not_round_or_reinterpret_invalid_dosages(
    modules, tmp_path, invalid
):
    _, snpnet, _ = modules
    X = np.array([[0, invalid], [1, 2]])
    with pytest.raises(ValueError, match="dosages"):
        snpnet.write_plink1(X, np.array([0, 1]), tmp_path / "X")
    assert not (tmp_path / "X.bed").exists()


def test_training_file_and_folds_have_one_common_nonconstant_column_filter(
    modules, monkeypatch, tmp_path
):
    cv, _, _ = modules
    indices = np.arange(60)
    X = np.column_stack([indices % 2, indices % 3, np.ones(60), indices == 0]).astype(
        float
    )
    y = (indices % 2).astype(float)
    monkeypatch.setattr(cv, "data", lambda *args: (X, y))
    X_test, y_test = cv.prepare_case("smoke", tmp_path, 3, 0)
    training, response, sets = cv.open_train(tmp_path)
    assert training.shape == (48, 2) and X_test.shape == (12, 2)
    assert len(y_test) == 12 and len(response) == 48
    assert training.flags.f_contiguous and not training.flags.writeable
    held_out = np.concatenate([test for _, test in cv.fold_indices(sets)])
    np.testing.assert_array_equal(np.sort(held_out), np.arange(48))
    assert all(
        np.all(training[train].std(axis=0) > 1e-6) for train, _ in cv.fold_indices(sets)
    )


def test_optional_image_and_blas_limits_are_independent_of_fixture_image(
    modules, monkeypatch, tmp_path
):
    _, _, competitors = modules
    monkeypatch.setenv("CONTAINER", "docker")
    monkeypatch.setenv("R_IMAGE", "fixture-image")
    monkeypatch.setenv("R_BENCH_IMAGE", "competitor-image")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "64")
    calls = []
    monkeypatch.setattr(
        competitors.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    (tmp_path / "out.json").write_text(json.dumps({"warnings": []}))
    competitors.run_r(tmp_path, tmp_path / "fit.R", 2)
    command, options = calls[0]
    assert "competitor-image" in command and "fixture-image" not in command
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        assert f"{key}=1" in command and options["env"][key] == "1"
    assert options["timeout"] == 2


def test_r_timeout_cleans_up_only_its_own_named_container(
    modules, monkeypatch, tmp_path
):
    _, _, competitors = modules
    monkeypatch.setenv("CONTAINER", "docker")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(competitors.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        competitors.run_r(tmp_path, tmp_path / "fit.R", 1)
    name = calls[0][calls[0].index("--name") + 1]
    assert name.startswith("pybiglasso-bench-")
    assert calls[1] == ["docker", "rm", "-f", name]


@pytest.mark.parametrize(
    "result", [{"error": "broken"}, {"warnings": ["did not converge"]}]
)
def test_r_errors_and_warnings_cannot_be_reported_as_valid_results(
    modules, monkeypatch, tmp_path, result
):
    _, _, competitors = modules
    monkeypatch.setattr(competitors.subprocess, "run", lambda *args, **kwargs: None)
    (tmp_path / "out.json").write_text(json.dumps(result))
    with pytest.raises(RuntimeError):
        competitors.run_r(tmp_path, tmp_path / "fit.R", 1)


def test_cv_records_median_not_best_and_marks_smokes_nonreportable(
    modules, monkeypatch, tmp_path
):
    cv, _, _ = modules
    out = tmp_path / "cv.jsonl"
    monkeypatch.setattr(
        cv.sys,
        "argv",
        [
            "cv.py",
            "--solver",
            "pybiglasso",
            "--alpha",
            "1",
            "--threads",
            "2",
            "--early-stopping",
            "off",
            "--repeats",
            "3",
            "--out",
            str(out),
        ],
    )
    monkeypatch.setattr(cv.bench, "environment", lambda: {"cpu": "test CPU"})
    times = iter([2.0, 9.0, 4.0])

    def child(command, **kwargs):
        assert kwargs["env"]["RAYON_NUM_THREADS"] == "2"
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        shape = json.loads(
            (Path(command[command.index("--case") + 1]) / "shape.json").read_text()
        )
        np.savez(
            command[-1],
            coef=np.zeros(shape["ncol"]),
            intercept=0.0,
            seconds=next(times),
            metadata=json.dumps({"family": "binomial"}),
        )

    monkeypatch.setattr(cv.subprocess, "run", child)
    cv.main()
    row = json.loads(out.read_text())
    assert row["seconds"] == 4.0 and row["times"] == [2.0, 9.0, 4.0]
    assert row["phase"] == "smoke" and row["reportable"] is False
    assert row["family"] == "binomial"


@pytest.mark.parametrize(
    "failure", [None, "iterations", "short_path", "column", "export"]
)
def test_bigstatsr_full_grid_is_valid_but_real_failures_are_not(
    modules, monkeypatch, tmp_path, failure
):
    cv, _, _ = modules
    cv.prepare_case("smoke", tmp_path, 3, 0)
    X, _, _ = cv.open_train(tmp_path)
    options = SimpleNamespace(
        alpha=1,
        threads=1,
        seed=0,
        folds=3,
        nlambda=60,
        lambda_min=0.05,
        early="off",
        timeout=1,
    )
    result = dict(
        beta=[0.0] * X.shape[1],
        ind_col=list(range(1, X.shape[1] + 1)),
        intercept=0.0,
        seconds=1.0,
        probe_logits=[0.0] * 8,
        fold_lambdas=[60] * 3,
        fold_max_iterations=[10] * 3,
        fold_messages=["All lambda values have been used"] * 3,
        all_early_stopped=False,
        max_iter=1000,
        bigstatsr_version="test",
        r_version="test",
        model_kind="CMSA",
        n_abort=2**31 - 1,
        nlam_min=50,
        nlambda=60,
        lambda_min=0.05,
        eps=1e-5,
        warnings=[
            "Some models may not have reached a minimum; check with summary() and plot()."
        ],
    )
    if failure == "iterations":
        result["fold_max_iterations"][0] = 1000
    elif failure == "short_path":
        result["fold_lambdas"][0] = 59
    elif failure == "column":
        result["ind_col"][0] = X.shape[1] + 1
    elif failure == "export":
        result["probe_logits"][0] = 1
    monkeypatch.setattr(cv, "run_r", lambda *args, **kwargs: result)
    if failure is None:
        coef, intercept, _, metadata = cv.fit_bigstatsr(tmp_path, options)
        assert not metadata["all_early_stopped"]
        assert np.all(coef == 0) and intercept == 0
    else:
        with pytest.raises(ValueError):
            cv.fit_bigstatsr(tmp_path, options)


@pytest.mark.parametrize("failure", ["truncated", "nonfinite", "grid"])
def test_snpnet_cannot_report_an_incomplete_or_different_problem(
    modules, monkeypatch, failure
):
    _, snpnet, _ = modules
    X = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]], order="F")
    lambdas = np.array([0.25, 0.125])
    result = dict(
        beta=[0.0] * 4,
        beta_dim=[2, 2],
        intercept=[0.0, 0.0],
        lambdas=[format(value, ".17g") for value in lambdas],
        seconds=1,
        warnings=[],
        snpnet_version="test",
        r_version="test",
    )
    if failure == "truncated":
        result.update(beta=[0.0] * 2, beta_dim=[2, 1], intercept=[0.0])
    elif failure == "nonfinite":
        result["beta"][0] = float("nan")
    else:
        result["lambdas"][0] = "0.225"
    monkeypatch.setattr(snpnet, "run_r", lambda *args, **kwargs: result)
    with pytest.raises((ValueError, AssertionError)):
        snpnet.fit_r_snpnet(X, X - 1, np.array([0.0, 0.0, 1.0, 1.0]), lambdas, 1, 1)


def test_specific_advisory_does_not_hide_a_convergence_warning(
    modules, monkeypatch, tmp_path
):
    _, _, competitors = modules
    monkeypatch.setattr(competitors.subprocess, "run", lambda *args, **kwargs: None)
    result = tmp_path / "out.json"
    result.write_text(json.dumps({"warnings": "a known advisory"}))
    assert competitors.run_r(
        tmp_path, tmp_path / "fit.R", 1, advisory_warnings=("a known advisory",)
    )["warnings"] == ["a known advisory"]
    result.write_text(
        json.dumps({"warnings": ["a known advisory", "failed to converge"]})
    )
    with pytest.raises(RuntimeError, match="converge"):
        competitors.run_r(
            tmp_path, tmp_path / "fit.R", 1, advisory_warnings=("a known advisory",)
        )


@pytest.mark.parametrize("interrupted", [False, True])
def test_path_cap_drains_logs_and_terminates_the_child_group(
    modules, monkeypatch, interrupted
):
    cv, _, _ = modules
    bench = cv.bench
    signals = []
    calls = []

    class Child:
        pid = 999
        stdout = io.StringIO(bench.TIMING_START + "\n")

        def communicate(self, timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                if interrupted:
                    raise KeyboardInterrupt
                raise subprocess.TimeoutExpired("fit", timeout)
            return "", None

        def terminate(self):
            signals.append("terminate")

    def spawn(*args, **kwargs):
        assert kwargs["start_new_session"] == (bench.os.name == "posix")
        return Child()

    monkeypatch.setattr(bench.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        bench.os, "killpg", lambda *args: signals.append(args), raising=False
    )
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            bench.run_capped(["fit"], {}, 0.1)
    else:
        assert bench.run_capped(["fit"], {}, 0.1) is False
    assert calls == [0.1, 10]
    assert signals == (
        [(999, bench.signal.SIGTERM)] if bench.os.name == "posix" else ["terminate"]
    )


def test_snpnet_smoke_kkt_has_an_analytic_positive_twin_and_negative_controls(modules):
    _, snpnet, _ = modules
    Xs = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    # With lambda=1/8, expit(beta)=1-lambda=7/8, so beta=log(7).
    beta = np.array([[np.log(7), 0.0]])
    grid = np.array([0.125])
    assert snpnet.max_kkt(Xs, y, grid, beta, np.zeros(1)) <= 32 * np.finfo(float).eps
    assert snpnet.max_kkt(Xs, y, grid, beta * 0.9, np.zeros(1)) > 1e-6
    assert snpnet.max_kkt(Xs, y, grid, beta, np.array([0.1])) > 1e-6


@pytest.mark.parametrize(
    "arguments", [["--repeats", "1"], ["--repeats", "3", "--dataset", "smoke"]]
)
def test_reportable_phase_rejects_single_runs_or_smoke_inputs(
    modules, monkeypatch, tmp_path, arguments
):
    cv, _, _ = modules
    monkeypatch.setattr(
        cv.sys,
        "argv",
        ["cv.py", "--phase", "reportable", "--out", str(tmp_path / "out"), *arguments],
    )
    with pytest.raises(SystemExit):
        cv.main()
