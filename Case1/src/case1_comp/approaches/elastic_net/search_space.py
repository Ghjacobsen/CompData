from __future__ import annotations

import numpy as np


def get_param_grid() -> dict:
    alpha_grid = np.logspace(-4, 1, 36).tolist()
    return {
        "model": [
            "ridge",
            "lasso",
            "elasticnet",
        ],
        "alpha": alpha_grid,
        "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
    }
