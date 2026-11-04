"""The numpy reference's fit, in the attributes the certificate helpers read.

`reference.fit_path` returns R's layout, a dict; the certificates in
`test_hypothesis.py` and `test_duality_gap.py` read a fitted path's
`lambdas`, `intercept`, `coef`, `alpha`, `penalty_factor`, `n_rejected` and
`iterations`. This supplies exactly those, so a certificate written for two
solvers can hold the solver to the reference instead.
"""

import dataclasses

import numpy as np
import scipy.sparse as sps

from support import reference


@dataclasses.dataclass(frozen=True)
class ReferencePath:
    lambdas: np.ndarray
    intercept: np.ndarray
    coef: sps.csc_array  # ncol x nfit, on X's scale
    alpha: float
    penalty_factor: np.ndarray
    n_rejected: np.ndarray
    iterations: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    def __len__(self):
        return len(self.lambdas)


def fit(X, y, *, alpha, penalty_factor, lambdas, **options):
    """Fit the reference on a supplied grid, and refuse any path cut short.

    The solver's fits in these tests turn its saturation and convergence
    warnings into errors; the reference reports those as events instead, so
    any event here is the same failure of the tests' stated domain.
    """
    penalty_factor = np.asarray(penalty_factor, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    result = reference.fit_path(
        X,
        y,
        alpha=alpha,
        penalty_factor=penalty_factor,
        lambdas=lambdas,
        **options,
    )
    assert result["events"] == [], result["events"]
    assert len(result["lambda_"]) == len(lambdas)
    return ReferencePath(
        lambdas=result["lambda_"],
        intercept=result["beta"][0],
        coef=sps.csc_array(result["beta"][1:]),
        alpha=alpha,
        penalty_factor=penalty_factor,
        n_rejected=result["rejections"],
        iterations=result["iter"],
        center=result["center"],
        scale=result["scale"],
    )
