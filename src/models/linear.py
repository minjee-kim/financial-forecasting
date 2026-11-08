

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error


# --------------------------------------------------
# 1. OLS
# --------------------------------------------------

def fit_ols(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model.predict(X_test), model


# --------------------------------------------------
# 2. Ridge
# --------------------------------------------------

def tune_ridge(X_train, y_train, alphas=None):

    if alphas is None:
        alphas = np.logspace(-6, 3, 20)

    cv = TimeSeriesSplit(n_splits=5, gap=1)

    pipeline = make_pipeline(
        StandardScaler(),
        Ridge()
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"ridge__alpha": alphas},
        scoring="neg_mean_squared_error",
        cv=cv,
        refit=True,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search


# --------------------------------------------------
# 3. LASSO with warm starts
# --------------------------------------------------

def tune_lasso(X_train, y_train, alphas=None):

    if alphas is None:
        alphas = np.logspace(-7, -2, 30)

    alphas = np.sort(np.asarray(alphas, dtype=float))[::-1]

    if not np.isfinite(alphas).all() or np.any(alphas <= 0):
        raise ValueError("All alpha values must be positive and finite.")

    cv = TimeSeriesSplit(n_splits=5, gap=1)

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)

    cv_errors = np.zeros((5, len(alphas)))

    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):

        X_tr = X[train_idx]
        X_val = X[val_idx]

        y_tr = y[train_idx]
        y_val = y[val_idx]

        # Fit scaler on current training fold only
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        # Reuse coefficients across alpha values
        model = Lasso(
            warm_start=True,
            max_iter=100000,
            tol=1e-5
        )

        for j, alpha in enumerate(alphas):

            model.set_params(alpha=alpha)
            model.fit(X_tr_scaled, y_tr)

            pred = model.predict(X_val_scaled)

            cv_errors[fold, j] = mean_squared_error(
                y_val, pred
            )

    # Select alpha with lowest average validation MSE
    mean_cv_errors = cv_errors.mean(axis=0)
    best_idx = np.argmin(mean_cv_errors)
    best_alpha = alphas[best_idx]

    # Refit on all supplied training observations
    final_model = make_pipeline(
        StandardScaler(),
        Lasso(
            alpha=best_alpha,
            max_iter=100000,
            tol=1e-5
        )
    )

    final_model.fit(X_train, y_train)

    return {
        "model": final_model,
        "best_alpha": best_alpha,
        "alphas": alphas,
        "cv_errors": cv_errors,
        "mean_cv_errors": mean_cv_errors
    }
