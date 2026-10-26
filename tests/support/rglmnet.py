"""R glmnet's fits of the default objective, for tests/test_vs_glmnet.py.

pybiglasso's default mode (not r_compatible) minimizes glmnet's objective, so
glmnet is an outside check of it. glmnet fits raw X with its own
standardization and lambda grid, converged far past its defaults. Stored
output, from glmnet FIXTURES_GLMNET in scripts/make_r_fixtures.py, is
tests/fixtures/r_glmnet.json.gz. Cases reuse rcompare's data, with all rows.
"""

import gzip
import json
import pathlib

import numpy as np

from support import rcompare

R_SCRIPT = rcompare.TESTS_DIR / "r" / "fit_glmnet.R"
FIXTURE_PATH = rcompare.TESTS_DIR / "fixtures" / "r_glmnet.json.gz"

# glmnet's own stopping threshold (on coefficient changes, weighted by the
# design's curvature), set far below its 1e-7 default so that its fits pass
# the KKT_TOL cap; and an iteration limit it should not reach
THRESH, MAXIT = 1e-14, 10_000_000


def glmnet_grid():
    """Tall cases (n > p), where curvature certifies distances."""
    Case = rcompare.Case
    cases = [
        Case(nrow, ncol, alpha=alpha)
        for nrow, ncol in ((300, 40), (1000, 20))
        for alpha in (1.0, 0.5, 0.2)
    ]
    cases += [
        Case(300, 40, alpha=0.5, penalty="varied"),
        Case(300, 40, alpha=1.0, penalty="zeros"),
        Case(1000, 20, alpha=0.5, data="dosage"),
        Case(300, 40, alpha=0.5, response="rare"),
    ]
    return cases


def glmnet_penalty_factor(case, penalty_factor):
    """Penalty factors scaled to sum to ncol, as glmnet rescales them, so
    both fits solve the same problem; None for all ones."""
    if penalty_factor is None:
        return None
    return penalty_factor * case.ncol / penalty_factor.sum()


def prepare_case(case, workdir):
    """Write a case's inputs for glmnet; return (fit inputs, data digest)."""
    workdir = pathlib.Path(workdir)
    data = rcompare.make_case_data(case, workdir)
    if case.rows != "all" or case.dtype != "float64":
        raise ValueError("glmnet cases fit every row of float64 X")
    data["penalty_factor"] = glmnet_penalty_factor(case, data["penalty_factor"])
    np.asarray(data["y"], dtype=np.float64).tofile(workdir / "y.bin")
    params = dict(
        nrow=case.nrow,
        ncol=case.ncol,
        alpha=case.alpha,
        nlambda=case.nlambda,
        lambda_min_ratio=case.resolved_lambda_min,
        thresh=THRESH,
        maxit=MAXIT,
    )
    if data["penalty_factor"] is not None:
        params["penalty_factor"] = data["penalty_factor"].tolist()
    (workdir / "params.json").write_text(json.dumps(params))
    return data, rcompare.data_digest(data)


def load_output(path):
    out = json.loads(pathlib.Path(path).read_text())
    if "error" in out:
        return out
    return parse(out)


def parse(out):
    """glmnet's path: lambdas, intercepts, and coefficients (ncol x nlambda)."""
    beta = np.array(out["beta"], dtype=np.float64).reshape(out["beta_dim"], order="F")
    return dict(
        out,
        lambda_=np.atleast_1d(np.array(out["lambda"], dtype=np.float64)),
        intercept=np.atleast_1d(np.array(out["intercept"], dtype=np.float64)),
        beta=beta,
    )


def run_glmnet(case_dirs):
    return rcompare.run_r(case_dirs, script=R_SCRIPT, load=load_output)


def save_fixtures(entries, path=FIXTURE_PATH):
    """`entries` maps case name -> (digest, raw glmnet output)."""
    payload = {
        name: dict(digest=digest, **{k: v for k, v in raw.items() if k in RAW_KEYS})
        for name, (digest, raw) in sorted(entries.items())
    }
    with gzip.open(path, "wt") as fp:
        json.dump(payload, fp, separators=(",", ":"))


RAW_KEYS = (
    "glmnet_version",
    "r_version",
    "lambda",
    "intercept",
    "beta",
    "beta_dim",
    "jerr",
    "warnings",
)


def load_fixtures(path=FIXTURE_PATH):
    with gzip.open(path, "rt") as fp:
        payload = json.load(fp)
    return {name: (entry["digest"], parse(entry)) for name, entry in payload.items()}
