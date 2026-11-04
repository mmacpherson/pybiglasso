"""A load failure rejects a record, without spending or changing fit time."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from support import rcompare


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("path")


def test_quiet_gate_requires_180_consecutive_seconds(benchmark, monkeypatch):
    samples = [0.01] * 35 + [0.2] + [0.02] * 36
    durations = []
    values = iter(samples)

    def cpu(seconds):
        durations.append(seconds)
        return next(values)

    monkeypatch.setattr(benchmark, "cpu_use", cpu)
    result = benchmark.wait_for_quiet()
    assert result == dict(samples=samples, threshold=0.1, consecutive_seconds=180)
    assert durations == [5] * 72


def test_repeated_fit_bookends_do_not_spend_native_clock(
    benchmark, monkeypatch, capsys
):
    monkeypatch.setenv("PYBIGLASSO_BENCH_RECORD", "1")
    ticks = iter([10.0, 11.0, 20.0, 22.0])
    seen = []

    def tick():
        value = next(ticks)
        seen.append(value)
        return value

    def cpu(seconds):
        assert seconds == 1
        assert len(seen) in (0, 4)
        return 0.01

    monkeypatch.setattr(benchmark.time, "perf_counter", tick)
    monkeypatch.setattr(benchmark, "cpu_use", cpu)
    clock = benchmark.FitClock(signals=True)
    clock.intervals_total = 2
    with clock:
        pass
    with clock:
        pass
    assert clock.seconds == 3.0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(benchmark.LOAD_PRE)
    assert lines[-1].startswith(benchmark.LOAD_POST)
    assert sum(line.startswith(benchmark.LOAD_PRE) for line in lines) == 1
    assert sum(line.startswith(benchmark.LOAD_POST) for line in lines) == 1


def test_busy_pre_bookend_never_starts_fit_clock(benchmark, monkeypatch, capsys):
    monkeypatch.setenv("PYBIGLASSO_BENCH_RECORD", "1")
    monkeypatch.setattr(benchmark, "cpu_use", lambda _: 0.2)
    with pytest.raises(RuntimeError, match="busy recording"):
        with benchmark.FitClock(signals=True):
            pytest.fail("fit must not run")
    assert benchmark.TIMING_START not in capsys.readouterr().out


@pytest.mark.parametrize("post", [None, 0.2])
def test_missing_or_busy_post_is_not_a_timeout(benchmark, tmp_path, post):
    script = tmp_path / "child.py"
    source = "print('--load-pre-- 0.01',flush=True)\nprint('--timing-start--',flush=True)\nprint('--timing-stop-- 0.01',flush=True)\n"
    if post is not None:
        source += f"print('--load-post-- {post}',flush=True)\n"
    script.write_text(source)
    audit = {}
    with pytest.raises(RuntimeError, match="recording bookend"):
        benchmark.run_capped([sys.executable, str(script)], None, 2, audit=audit)
    assert audit["fit_seconds"] == 0.01


@pytest.mark.parametrize("post,busy", [(0.01, False), (0.2, True)])
def test_capped_fit_requires_quiet_post_after_termination(
    benchmark, monkeypatch, tmp_path, post, busy
):
    script = tmp_path / "child.py"
    script.write_text(
        "import time\nprint('--load-pre-- 0.01',flush=True)\n"
        "print('--timing-start--',flush=True)\ntime.sleep(5)\n"
    )
    monkeypatch.setattr(benchmark, "cpu_use", lambda _: post)
    audit = {}
    if busy:
        with pytest.raises(RuntimeError, match="invalid recording load"):
            benchmark.run_capped([sys.executable, str(script)], None, 0.03, audit=audit)
    else:
        assert not benchmark.run_capped(
            [sys.executable, str(script)], None, 0.03, audit=audit
        )
        assert audit["timed_out"] and audit["pre"] == 0.01 and audit["post"] == 0.01


def test_recording_child_errors_are_not_timeout_rows(benchmark, tmp_path):
    script = tmp_path / "child.py"
    script.write_text("raise RuntimeError('setup failed')\n")
    with pytest.raises(subprocess.CalledProcessError):
        benchmark.run_capped([sys.executable, str(script)], None, 2, audit={})


def test_recording_metadata_rejects_wrong_native_build(
    benchmark, monkeypatch, tmp_path
):
    manifest = tmp_path / "build.json"
    manifest.write_text(json.dumps(dict(native_sha256="wrong")))
    monkeypatch.setenv("PYBIGLASSO_BUILD_INFO", str(manifest))
    with pytest.raises(ValueError, match="installed native extension"):
        benchmark.compiler_provenance()


@pytest.mark.r
@pytest.mark.skipif(
    not rcompare.r_available(),
    reason="R not available",
)
def test_r_bookends_use_the_same_threshold_without_spending_fit_time(tmp_path):
    root = Path(__file__).resolve().parents[1]
    case = tmp_path / "case"
    case.mkdir()
    for name, original in [
        ("biglasso.R", root / "tests/r/fit_biglasso.R"),
        ("glmnet.R", root / "benchmarks/r/fit_glmnet.R"),
    ]:
        (case / name).write_bytes(original.read_bytes())
    script = tmp_path / "bookends.R"
    script.write_text(r"""
case <- commandArgs(trailingOnly = TRUE)[[1]]
readLines <- function(con, ...) {
  if (!identical(con, "/proc/stat")) return(base::readLines(con, ...))
  reads <<- reads + 1L
  samples[[reads]]
}
Sys.sleep <- function(...) NULL
for (name in c("biglasso.R", "glmnet.R")) {
  definitions <- Filter(function(e) is.call(e) && identical(e[[1]], as.name("<-")) &&
    identical(e[[2]], as.name("record_bookend")), as.list(parse(file.path(case, name))))
  stopifnot(length(definitions) == 1)
  eval(definitions[[1]])
  samples <- c("cpu 100 0 0 900 0 0 0 0", "cpu 110 0 0 990 0 0 0 0")
  reads <- 0L
  out <- capture.output(record_bookend("--load-pre--"))
  stopifnot(reads == 2L, as.numeric(strsplit(out[[1]], " ")[[1]][[2]]) == 0.1)
  samples <- c("cpu 100 0 0 900 0 0 0 0", "cpu 150 0 0 950 0 0 0 0")
  reads <- 0L
  error <- tryCatch(record_bookend("--load-post--"), error = identity)
  stopifnot(inherits(error, "error"), grepl("Busy recording", conditionMessage(error)))
}
writeLines('{"ok":true}', file.path(case, "out.json"))
""")
    (result,) = rcompare.run_r(
        [case], script=script, load=lambda p: json.loads(p.read_text())
    )
    assert result == {"ok": True}
