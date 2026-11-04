"""Run native tests with the project's Python available to Fraction audits."""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYO3_PYTHON"] = sys.executable
# uv's standalone interpreters keep libpython outside the system loader path.
# Match the interpreter used to build the embedded audit, on every platform.
libdir = sysconfig.get_config_var("LIBDIR") or sys.base_prefix
variable = (
    "PATH"
    if sys.platform == "win32"
    else "DYLD_LIBRARY_PATH"
    if sys.platform == "darwin"
    else "LD_LIBRARY_PATH"
)
env[variable] = os.pathsep.join(
    [str(libdir), str(Path(sys.base_prefix)), env.get(variable, "")]
)
raise SystemExit(
    subprocess.call(
        [
            "cargo",
            "test",
            "--release",
            "--manifest-path",
            str(root / "rust/Cargo.toml"),
        ],
        env=env,
    )
)
