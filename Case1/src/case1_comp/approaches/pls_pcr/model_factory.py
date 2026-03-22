from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from case1_comp.common.preprocessing import build_preprocessor


class PLSorPCRRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model: str = "pls",
        n_components: int = 20,
        alpha: float = 1.0,
        random_state: int = 42,
    ):
        self.model = model
        self.n_components = n_components
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        max_components = max(1, min(X.shape[0] - 1, X.shape[1]))
        n_comp = min(int(self.n_components), max_components)

        if self.model == "pls":
            self.model_ = PLSRegression(n_components=n_comp)
            self.model_.fit(X, y)
        else:
            self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
            Z = self.pca_.fit_transform(X)
            self.model_ = Ridge(alpha=self.alpha, random_state=self.random_state)
            self.model_.fit(Z, y)
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["model_"])
        if self.model == "pls":
            yhat = self.model_.predict(X)
            return np.asarray(yhat).reshape(-1)
        check_is_fitted(self, attributes=["pca_"])
        Z = self.pca_.transform(X)
        return self.model_.predict(Z)


def build_estimator(numeric_cols: list[str], categorical_cols: list[str], seed: int):
    pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    reg = PLSorPCRRegressor(random_state=seed)
    return Pipeline(steps=[("pre", pre), ("reg", reg)])


def complexity_key(params: dict) -> tuple:
    model_rank = {"pcr": 0, "pls": 1}[params["reg__model"]]
    return (
        int(params["reg__n_components"]),
        -float(params["reg__alpha"]),
        model_rank,
    )
