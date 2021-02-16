import glmnet
import numpy as np
import sklearn.linear_model

import pybiglasso as bl


def test_output_shape():

    nr, nc, nl = 10, 20, 7
    testdata = bl.common.generate_test_data(num_rows=nr, num_features=nc, seed=123)

    result = bl.biglasso(
        testdata["X"],
        testdata["y"],
        alpha=0.8,
        family="binomial",
        nlambda=nl,
    )

    assert set(result.keys()) == set(["df_ncol", "df_nlambda", "df_weights"])

    assert result["df_weights"].shape == (nl, nc + 1)


def test_binomial_against_glmnet():

    nrow = 100
    ncol = 10
    seed = 123
    alpha = 1.0
    eps = 1e-7
    tol = 1e-3
    nlambda = 21
    lambda_min = 1e-6

    test_data = bl.common.generate_test_data(nrow, ncol, seed=seed)
    X, y = test_data["X"], test_data["y"]

    bl_fit = bl.biglasso(
        X,
        y,
        alpha=alpha,
        family="binomial",
        eps=eps,
        nlambda=nlambda,
        lambda_min=lambda_min,
    )
    df_nlambda = bl_fit["df_nlambda"]
    df_weights = bl_fit["df_weights"]

    # -- compare path against glmnet
    gn_fit = glmnet.LogitNet(
        lambda_path=df_nlambda["lambda_"], alpha=alpha, n_splits=0, tol=eps
    )
    gn_fit.fit(X, y)

    assert np.allclose(df_nlambda["lambda_"].to_numpy(), gn_fit.lambda_path_)

    assert np.allclose(
        df_weights["Intercept"].to_numpy(), gn_fit.intercept_path_, atol=tol
    )
    assert np.allclose(
        df_weights.loc[:, df_weights.columns != "Intercept"].to_numpy().transpose(),
        gn_fit.coef_path_,
        atol=tol,
    )

    # -- test the very-small λ case against straight logistic regression
    sk_fit = sklearn.linear_model.LogisticRegression(C=1e5)
    sk_fit.fit(X, y)

    assert np.allclose(
        df_weights["Intercept"].to_numpy()[-1], sk_fit.intercept_, atol=tol
    )
    assert np.allclose(
        df_weights.iloc[-1, df_weights.columns != "Intercept"].to_numpy(),
        sk_fit.coef_,
        atol=tol,
    )

    # -- now fit smaller, alpha, tweaking tolerances
    # -- FIXFIX: cleaner if we use pytest's parametrization
    alpha = 0.5
    tol = 5e-2  # BIG allowance for nonzero alpha... why that different from glmnet? who's more right here?

    bl_fit = bl.biglasso(
        X,
        y,
        alpha=alpha,
        family="binomial",
        eps=eps,
        nlambda=nlambda,
        lambda_min=lambda_min,
    )
    df_nlambda = bl_fit["df_nlambda"]
    df_weights = bl_fit["df_weights"]

    # -- compare path against glmnet
    gn_fit = glmnet.LogitNet(
        lambda_path=df_nlambda["lambda_"], alpha=alpha, n_splits=0, tol=eps
    )
    gn_fit.fit(X, y)

    assert np.allclose(df_nlambda["lambda_"].to_numpy(), gn_fit.lambda_path_)

    assert np.allclose(
        df_weights["Intercept"].to_numpy(), gn_fit.intercept_path_, atol=tol
    )
    assert np.allclose(
        df_weights.loc[:, df_weights.columns != "Intercept"].to_numpy().transpose(),
        gn_fit.coef_path_,
        atol=tol,
    )

    # -- test the very-small λ case against straight logistic regression
    sk_fit = sklearn.linear_model.LogisticRegression(C=1e5)
    sk_fit.fit(X, y)

    assert np.allclose(
        df_weights["Intercept"].to_numpy()[-1], sk_fit.intercept_, atol=tol
    )
    assert np.allclose(
        df_weights.iloc[-1, df_weights.columns != "Intercept"].to_numpy(),
        sk_fit.coef_,
        atol=tol,
    )


def assert_same_result(fit1, fit2):

    print(fit1["df_weights"])
    print(fit2["df_weights"])
    assert np.allclose(
        fit1["df_weights"].to_numpy(),
        fit2["df_weights"].to_numpy(),
    )

    print(fit1["df_nlambda"])
    print(fit2["df_nlambda"])
    assert np.allclose(
        fit1["df_nlambda"].to_numpy(),
        fit2["df_nlambda"].to_numpy(),
    )

    print(fit1["df_ncol"])
    print(fit2["df_ncol"])
    assert np.allclose(
        fit1["df_ncol"].loc[:, ["center", "scale", "penalty_factor"]].to_numpy(),
        fit2["df_ncol"].loc[:, ["center", "scale", "penalty_factor"]].to_numpy(),
    )


def test_binomial_row_indices():

    nrow = 100
    ncol = 10
    seed = 123
    alpha = 1.0
    eps = 1e-7
    nlambda = 21
    lambda_min = 1e-6

    test_data = bl.common.generate_test_data(nrow, ncol, seed=seed)
    X, y = test_data["X"], test_data["y"]

    # -- no row indices should be same as all row indices
    bl_fit_1 = bl.biglasso(
        X,
        y,
        alpha=alpha,
        family="binomial",
        eps=eps,
        nlambda=nlambda,
        lambda_min=lambda_min,
    )
    bl_fit_2 = bl.biglasso(
        X,
        y,
        row_indices=np.arange(X.shape[0]),
        alpha=alpha,
        family="binomial",
        eps=eps,
        nlambda=nlambda,
        lambda_min=lambda_min,
    )
    assert_same_result(bl_fit_1, bl_fit_2)

    # -- select row indices should be same as preselected X, y
    row_indices = np.flip(np.arange(X.shape[0])[::2])
    print(row_indices)
    bl_fit_1 = bl.biglasso(
        np.asfortranarray(X[row_indices, :]),
        y[row_indices],
        alpha=alpha,
        family="binomial",
        eps=eps,
        nlambda=nlambda,
        lambda_min=lambda_min,
    )
    bl_fit_2 = bl.biglasso(
        X,
        y,
        row_indices=row_indices,
        alpha=alpha,
        family="binomial",
        eps=eps,
        nlambda=nlambda,
        lambda_min=lambda_min,
    )
    assert_same_result(bl_fit_1, bl_fit_2)
