import numpy as np


def get_param_grid() -> dict:
    return {
        "model": ["pls", "pcr"],
        "n_components": [5, 10, 15, 20, 25, 30, 40, 50, 60],
        "alpha": np.logspace(-3, 2, 12).tolist(),
    }
