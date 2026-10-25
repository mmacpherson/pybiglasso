"""Bounded, reproducible property tests; opt into a longer local search with
HYPOTHESIS_PROFILE=long make test (or run just tests/test_hypothesis.py).
"""

import os

from hypothesis import settings

settings.register_profile(
    "ci",
    max_examples=20,
    derandomize=True,
    database=None,
    # Compilation and scheduler load are unrelated to numerical correctness.
    deadline=None,
    print_blob=True,
)
settings.register_profile("long", parent=settings.get_profile("ci"), max_examples=200)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
