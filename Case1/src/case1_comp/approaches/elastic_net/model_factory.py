from __future__ import annotations

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from case1_comp.common.preprocessing import build_preprocessor


class LinearSwitchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model: str = "elasticnet", alpha: float = 0.1, l1_ratio: float = 0.5, random_state: int = 42):
        self.model = model
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def _build_model(self):
        if self.model == "ridge":
            return Ridge(alpha=self.alpha, random_state=self.random_state)
        if self.model == "lasso":
            return Lasso(alpha=self.alpha, random_state=self.random_state, max_iter=20000)
        return ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state,
            max_iter=20000,
        )

    def fit(self, X, y):
        self.model_ = self._build_model()
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["model_"])
        return self.model_.predict(X)


def build_estimator(numeric_cols: list[str], categorical_cols: list[str], seed: int):
    pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    reg = LinearSwitchRegressor(random_state=seed)
    return Pipeline(steps=[("pre", pre), ("reg", reg)])


def complexity_key(params: dict) -> tuple:
    # Prefer stronger regularization and simpler penalties.
    model = params["reg__model"]
    alpha = float(params["reg__alpha"])
    l1 = float(params["reg__l1_ratio"])
    model_rank = {"ridge": 0, "elasticnet": 1, "lasso": 2}[model]
    return (-alpha, model_rank, l1)
