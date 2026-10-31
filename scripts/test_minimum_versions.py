"""Resolve and test every declared floor without changing the development lock."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    print("+", shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True, **kwargs)


def lower_bound(requirement):
    bounds = [
        Version(s.version) for s in requirement.specifier if s.operator in (">=", "~=")
    ]
    return max(bounds) if bounds else None


def main():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = configuration["project"]
    name = canonicalize_name(project["name"])
    tools = [
        r
        for r in configuration["dependency-groups"]["dev"]
        if canonicalize_name(Requirement(r).name) != name
    ]
    build = configuration["build-system"]["requires"]
    runtime = project["dependencies"] + [
        r for group in project["optional-dependencies"].values() for r in group
    ]
    floors = {}
    for raw in runtime + tools + build:
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        floor = lower_bound(requirement)
        if floor is not None:
            package = canonicalize_name(requirement.name)
            floors[package] = max(floor, floors.get(package, floor))

    with tempfile.TemporaryDirectory(prefix="pybiglasso-minimum-") as directory:
        workspace = Path(directory)
        environment = workspace / "venv"
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        run(["uv", "venv", "--python", sys.executable, environment])
        bounded_tools = workspace / "bounded-tools.in"
        bounded_tools.write_text(
            "\n".join(
                r for r in tools + build if lower_bound(Requirement(r)) is not None
            )
            + "\n"
        )
        constraints = workspace / "minimum.txt"
        run(
            [
                "uv",
                "pip",
                "compile",
                ROOT / "pyproject.toml",
                "--all-extras",
                bounded_tools,
                "--resolution",
                "lowest-direct",
                "--python",
                python,
                "--output-file",
                constraints,
            ]
        )
        run(["uv", "pip", "sync", "--python", python, constraints])
        # Build a wheel with the minimum backend, outside the editable source tree.
        build_settings = os.environ.copy()
        build_settings["CARGO_TARGET_DIR"] = str(workspace / "target")
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "--no-build-isolation",
                ROOT,
            ],
            env=build_settings,
        )
        # Unbounded development tools stay current; runtime and declared tool floors stay fixed.
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--constraint",
                constraints,
                *tools,
            ]
        )
        query = "import importlib.metadata,json,sys; print(json.dumps({n:importlib.metadata.version(n) for n in sys.argv[1:]}))"
        installed = json.loads(
            subprocess.check_output(
                [str(python), "-c", query, *sorted(floors)], text=True
            )
        )
        for package, floor in sorted(floors.items()):
            actual = Version(installed[package])
            print(
                f"{package}: declared minimum {floor}, installed {actual}", flush=True
            )
            if actual != floor:
                raise RuntimeError(
                    f"{package} minimum {floor} is masked by installed {actual}"
                )
        run(["uv", "pip", "check", "--python", python])
        settings = os.environ.copy()
        settings.update(
            UV_PROJECT_ENVIRONMENT=str(environment),
            VIRTUAL_ENV=str(environment),
            UV_NO_SYNC="1",
            UV_LOCKED="false",
        )
        settings["PATH"] = str(python.parent) + os.pathsep + settings["PATH"]
        run(["make", "test"], env=settings)


if __name__ == "__main__":
    main()
