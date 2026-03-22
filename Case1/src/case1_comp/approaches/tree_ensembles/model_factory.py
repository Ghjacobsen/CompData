from __future__ import annotations

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from case1_comp.common.preprocessing import build_preprocessor


class TreeSwitchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model: str = "rf",
        n_estimators: int = 400,
        max_features: float | str = "sqrt",
        min_samples_leaf: int = 1,
        max_depth: int | None = None,
        random_state: int = 42,
    ):
        self.model = model
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state

    def _build_model(self):
        common = {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "n_jobs": 1,
        }
        if self.model == "extra":
            return ExtraTreesRegressor(**common)
        return RandomForestRegressor(**common)

    def fit(self, X, y):
        self.model_ = self._build_model()
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["model_"])
        return self.model_.predict(X)


def build_estimator(numeric_cols: list[str], categorical_cols: list[str], seed: int):
    pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)
    reg = TreeSwitchRegressor(random_state=seed)
    return Pipeline(steps=[("pre", pre), ("reg", reg)])


def complexity_key(params: dict) -> tuple:
    # Prefer fewer trees, shallower depth, bigger leaves, and smaller feature subsampling.
    depth = params["reg__max_depth"]
    depth_val = 10**6 if depth is None else int(depth)
    mf = params["reg__max_features"]
    if isinstance(mf, str):
        mf_val = 0.5
    else:
        mf_val = float(mf)
    model_rank = {"rf": 0, "extra": 1}[params["reg__model"]]
    return (
        int(params["reg__n_estimators"]),
        depth_val,
        -int(params["reg__min_samples_leaf"]),
        mf_val,
        model_rank,
    )
