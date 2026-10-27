"""Shared isolated R runner for the optional benchmark image, not fixtures."""

import json
import os
import subprocess
import uuid
from pathlib import Path

import path as bench


def run_r(case_dir, script, timeout, *, advisory_warnings=()):
    """Pin BLAS as in path.py; remove our own named container on timeout."""
    case_dir, script = Path(case_dir).resolve(), Path(script).resolve()
    container = os.environ.get("CONTAINER", "docker")
    limits = bench.r_thread_limits(1)  # explicit fold concurrency owns the CPUs
    name = "pybiglasso-bench-" + uuid.uuid4().hex
    if container:
        command = [container, "run", "--rm", "--name", name]
        if Path(container).name == "docker":
            command += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for key, value in limits.items():
            command += ["--env", f"{key}={value}"]
        for mount in sorted({script.parent, case_dir.parent}):
            command += ["-v", f"{mount}:{mount}"]
        image = os.environ.get("R_BENCH_IMAGE", "pybiglasso-bench:r4.6.1-snpnet")
        command += [image, "Rscript"]
    else:
        command = [os.environ.get("RSCRIPT", "Rscript")]
    try:
        subprocess.run(
            [*command, str(script), str(case_dir)],
            check=True,
            timeout=timeout,
            env=dict(os.environ, **limits),
        )
    except subprocess.TimeoutExpired:
        if container:
            subprocess.run(
                [container, "rm", "-f", name], check=False, capture_output=True
            )
        raise
    result = json.loads((case_dir / "out.json").read_text())
    if "error" in result:
        raise RuntimeError(result["error"])
    observed = result.get("warnings", [])
    result["warnings"] = [observed] if isinstance(observed, str) else observed
    if any(message not in advisory_warnings for message in result["warnings"]):
        raise RuntimeError(f"R benchmark warnings: {result['warnings']}")
    return result
