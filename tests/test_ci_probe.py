import sys


def test_ci_probe_fails_on_macos():
    # CI probe for #90: a macOS-only failure must fail "CI passed".
    assert sys.platform != "darwin"
