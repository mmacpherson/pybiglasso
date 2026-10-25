"""Property tests: a fixed, bounded set of examples in `make test` and CI;
`make soak-hypothesis` (part of `make soak`) searches far more widely.
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
# a fresh random search each run; failures are kept in .hypothesis/ and
# replayed first next time
settings.register_profile("soak", max_examples=5000, deadline=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
