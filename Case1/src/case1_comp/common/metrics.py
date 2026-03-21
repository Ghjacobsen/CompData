from __future__ import annotations

from sklearn.metrics import root_mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(root_mean_squared_error(y_true, y_pred))
