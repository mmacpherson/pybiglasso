"""Log the compiler actually selected for a wheel, and check the release pin.

This also runs under the manylinux image's system Python, without dependencies.
"""

import argparse
import subprocess


def report(expected_version: str) -> None:
    rustc = subprocess.check_output(["rustc", "-vV"], text=True)
    maturin = subprocess.check_output(["maturin", "--version"], text=True)
    print(rustc.rstrip(), flush=True)
    print(maturin.rstrip(), flush=True)
    release = next(
        (
            line.removeprefix("release: ").strip()
            for line in rustc.splitlines()
            if line.startswith("release: ")
        ),
        None,
    )
    if release != expected_version:
        raise SystemExit(f"Expected Rust {expected_version}, selected {release}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-rust-version", required=True)
    args = parser.parse_args()
    if not args.expected_rust_version:
        parser.error("set RUSTUP_TOOLCHAIN to the expected wheel compiler version")
    report(args.expected_rust_version)


if __name__ == "__main__":
    main()
