#!/usr/bin/env python3

import numpy as np  # noqa: F401 -- referenced inside plotnine aes() strings


def _plotnine():
    try:
        import plotnine
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Plotting requires plotnine: pip install 'pybiglasso[viz]'"
        ) from e
    return plotnine


def plot_effects(results):
    """Plot effect size as a function of regularization strength lambda."""
    p9 = _plotnine()

    _df = (
        results["df_weights"]
        .drop("Intercept", axis="columns")
        .reset_index()
        .rename({"index": "lambda_"}, axis="columns")
        .melt("lambda_")
        # .loc[lambda f: f.value != 0]
    )

    return (
        p9.ggplot(_df, p9.aes(x="-np.log10(lambda_)", y="value", color="variable"))
        + p9.geom_line()
        + p9.geom_point()
        + p9.scale_color_brewer(type="qual", guide=p9.guide_legend(title="Term"))
        + p9.labs(x="−log10(λ)", y="Effect Size")
    )


def plot_deviance(results):
    p9 = _plotnine()
    return (
        p9.ggplot(results["df_nlambda"], p9.aes(x="-np.log10(lambda_)", y="deviance"))
        + p9.geom_line()
        + p9.geom_point()
        + p9.labs(x="−log10(λ)", y="Deviance")
    )
