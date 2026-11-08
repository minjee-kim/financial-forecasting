
import numpy as np
from scipy import stats


def fit_predict(X_train, y_train, X_test):
    """Fit a Cauchy distribution and predict its location."""

    y = np.asarray(y_train, dtype=float)

    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("y_train must be a nonempty, finite 1D array.")

    # Fit in percentage units
    loc, scale = stats.cauchy.fit(y * 100)

    # Convert back to log-return units
    predictions = np.full(len(X_test), loc / 100)

    return predictions
