def get_param_grid() -> dict:
    return {
        "model": ["rf", "extra"],
        "n_estimators": [200, 400, 800],
        "max_features": ["sqrt", 0.5, 0.8],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_depth": [None, 8, 16, 24],
    }
