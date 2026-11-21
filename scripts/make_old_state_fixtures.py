"""Fitted state pickled by an earlier pybiglasso, for tests/fixtures/old_state/.

Those three files exercise the refusal of fitted paths and estimators saved
before centered prediction state was recorded: a fitted Gaussian path
(default mode), a fitted BiglassoClassifier, which keeps no path, and an
unfitted BiglassoRegressor. They were made with a release-profile wheel built
from `git archive b88db8b`, installed alone in a fresh environment:

    python -I scripts/make_old_state_fixtures.py --wheel WHEEL --commit b88db8b \
        --repo . --out tests/fixtures/old_state

The script checks that the installed package is the wheel's, and that the
wheel's Python sources are those of --commit, before writing anything. It
cannot check the compiled extension against the commit; build the wheel from
that commit's archive. It writes only the pickles: they are serialization
inputs, and nothing here records predictions as expectations.
"""

import argparse
import hashlib
import pickle
import pickletools
import re
import subprocess
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pybiglasso._rust as native
from threadpoolctl import threadpool_limits

from pybiglasso import biglasso
from pybiglasso.sklearn import BiglassoClassifier, BiglassoRegressor

SEED = 820260
PROTOCOL = 4
ABSOLUTE_PATH = re.compile(
    rb"(?<![A-Za-z0-9_])(?:/[A-Za-z0-9_.-]+/|[A-Za-z]:[\\/]|\\\\[A-Za-z0-9_.-]+\\)"
)


def verify_wheel(wheel, commit, repo):
    """The installed package is `wheel`, whose Python sources are `commit`'s."""
    package = Path(native.__file__).parent
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.startswith("pybiglasso/")]
        for name in names:
            installed = package / name.removeprefix("pybiglasso/")
            assert installed.read_bytes() == archive.read(name), name
        sources = [n for n in names if n.endswith(".py")]
        assert sources, "the wheel has no Python sources"
        for name in sources:
            committed = subprocess.run(
                ["git", "-C", str(repo), "show", f"{commit}:src/{name}"],
                check=True,
                capture_output=True,
            ).stdout
            assert archive.read(name) == committed, f"{name} isn't {commit}'s"


def inputs():
    rng = np.random.default_rng(SEED)
    train = np.asfortranarray(rng.standard_normal((192, 6)))
    rng.standard_normal((8, 6))  # drawn for the original prediction probe
    shift = np.array([3.0, -2.0, 0.0, 1.0, -1.0, 0.5])
    train += shift
    gaussian = 1.5 + train @ np.array([1.0, -0.8, 0.0, 0.5, 0.0, 0.0])
    gaussian += 0.5 * rng.standard_normal(len(train))
    logits = (train - shift) @ np.array([0.9, -0.7, 0.0, 0.6, 0.0, 0.0])
    probability = 1 / (1 + np.exp(-logits))
    binomial = (rng.random(len(train)) < probability).astype(np.int64)
    return train, gaussian, binomial


def audit(contents, name):
    """No absolute paths, in the raw bytes or the pickle's string operands."""
    assert contents[:2] == bytes((0x80, PROTOCOL)), name
    assert not ABSOLUTE_PATH.search(contents), name
    for op, argument, _ in pickletools.genops(contents):
        if isinstance(argument, str):
            assert not ABSOLUTE_PATH.search(argument.encode()), (name, op.name)


def generate(out, wheel, commit, repo):
    verify_wheel(wheel, commit, repo)
    train, gaussian, binomial = inputs()
    objects = {
        "path_gaussian_r_compatible_false.pkl": biglasso(
            train, gaussian, family="gaussian", nlambda=9, lambda_min=0.03, n_threads=1
        ),
        "BiglassoClassifier_fitted.pkl": BiglassoClassifier(alpha=0.05).fit(
            train, binomial
        ),
        "BiglassoRegressor_unfitted.pkl": BiglassoRegressor(alpha=0.05),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, value in objects.items():
        contents = pickle.dumps(value, protocol=PROTOCOL)
        audit(contents, name)
        (out / name).write_bytes(contents)
        print(f"{hashlib.sha256(contents).hexdigest()}  {name}  ({commit})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path, help="the installed wheel")
    parser.add_argument("--commit", required=True, help="the commit it was built from")
    parser.add_argument("--repo", required=True, type=Path, help="a clone holding it")
    parser.add_argument("--out", required=True, type=Path, help="where to write")
    args = parser.parse_args()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error")
        generate(args.out, args.wheel, args.commit, args.repo)
