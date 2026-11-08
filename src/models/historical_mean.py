
import numpy as np


def fit_predict(X_train, y_train, X_test):
    """
    Fit a historical-mean benchmark and generate forecasts.

    Parameters
    ----------
    X_train : array-like
        Training features (unused by this benchmark).
    y_train : array-like
        Observed training returns.
    X_test : array-like
        Features for the forecast period (used for its length).

    Returns
    -------
    np.ndarray
        Constant predictions in log-return units.
    """
    y_train = np.asarray(y_train, dtype=float)

    if y_train.ndim != 1 or len(y_train) == 0:
        raise ValueError("y_train must be a nonempty 1D array.")
    if not np.isfinite(y_train).all():
        raise ValueError("y_train contains nonfinite values.")

    mean_return = y_train.mean()

    predictions = np.full(len(X_test), mean_return)

    return predictions
