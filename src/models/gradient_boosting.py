
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def tune_gradient_boosting(X_train, y_train):
    """
    Tune gradient boosting using expanding-window
    time-series cross-validation.
    """

    cv = TimeSeriesSplit(
        n_splits=5,
        gap=1
    )

    model = GradientBoostingRegressor(
        loss="squared_error",
        subsample=1.0,
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.1],
        "max_depth": [1, 2],
        "min_samples_leaf": [10, 30]
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        refit=True,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search
