"""Path plots via seaborn.objects; install with `pip install 'pybiglasso[viz]'`."""

import numpy as np


def _so():
    try:
        import seaborn.objects as so
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Plotting needs seaborn: pip install 'pybiglasso[viz]'"
        ) from e
    return so


def plot_coefficients(path, max_features=None):
    """Coefficient paths against -log10(lambda), for features that ever enter.

    With `max_features`, show only the features that enter the model earliest.
    Returns a `seaborn.objects.Plot` for further customization.
    """
    so = _so()
    nonzero = path.coef.toarray() != 0
    ever = np.flatnonzero(nonzero.any(axis=1))
    if max_features is not None:
        entry = nonzero[ever].argmax(axis=1)
        ever = ever[np.argsort(entry, kind="stable")[:max_features]]
    coef = path.coef[ever, :].toarray()
    names = np.asarray(path.feature_names, dtype=object)[ever]
    data = dict(
        neg_log10_lambda=np.tile(-np.log10(path.lambdas), len(ever)),
        coefficient=coef.ravel(),
        feature=np.repeat(names, len(path)),
    )
    return (
        so.Plot(data, x="neg_log10_lambda", y="coefficient", color="feature")
        .add(so.Line())
        .label(x="−log10(λ)", y="Coefficient", color="Feature")
        .layout(size=(7, 4.5), engine="constrained")
    )


def plot_deviance(path):
    """Training deviance against -log10(lambda)."""
    so = _so()
    data = dict(neg_log10_lambda=-np.log10(path.lambdas), deviance=path.deviance)
    return (
        so.Plot(data, x="neg_log10_lambda", y="deviance")
        .add(so.Line())
        .add(so.Dot())
        .label(x="−log10(λ)", y="Deviance")
        .layout(size=(7, 4.5), engine="constrained")
    )
