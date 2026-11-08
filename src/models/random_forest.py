
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def tune_random_forest(X_train, y_train):
    """
    Tune random forest regression using expanding-window CV.
    """

    cv = TimeSeriesSplit(n_splits=5, gap=1)

    model = RandomForestRegressor(
        n_estimators=200,
        bootstrap=True,
        random_state=42,
        n_jobs=1
    )

    param_grid = {
        "max_depth": [3, 5, 10],
        "min_samples_leaf": [5, 20, 50],
        "max_features": [0.5, 1.0]
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

