from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold

from .metrics import rmse
from .selection import ComplexityFn, select_1se_index


@dataclass
class NestedCVResult:
    outer_rows: list[dict[str, Any]]
    tuning_rows: list[dict[str, Any]]

    @property
    def mean_outer_rmse(self) -> float:
        return float(np.mean([r["outer_rmse"] for r in self.outer_rows]))

    @property
    def std_outer_rmse(self) -> float:
        return float(np.std([r["outer_rmse"] for r in self.outer_rows], ddof=1))


def run_nested_cv(
    estimator,
    param_grid: dict[str, Any],
    complexity_fn: ComplexityFn,
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int,
    inner_splits: int,
    seed: int,
    n_jobs: int,
) -> NestedCVResult:
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    outer_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold_idx)
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=False,
        )
        search.fit(X_train, y_train)

        chosen_idx = select_1se_index(
            cv_results=search.cv_results_,
            n_splits_inner=inner_splits,
            complexity_fn=complexity_fn,
        )
        chosen_params = search.cv_results_["params"][chosen_idx]

        chosen_model = clone(estimator).set_params(**chosen_params)
        chosen_model.fit(X_train, y_train)
        y_pred = chosen_model.predict(X_test)
        fold_rmse = rmse(y_test, y_pred)

        outer_rows.append(
            {
                "fold": fold_idx,
                "outer_rmse": fold_rmse,
                "chosen_params": chosen_params,
            }
        )

        means = search.cv_results_["mean_test_score"]
        stds = search.cv_results_["std_test_score"]
        params = search.cv_results_["params"]
        for idx, (m, s, p) in enumerate(zip(means, stds, params)):
            tuning_rows.append(
                {
                    "fold": fold_idx,
                    "candidate_index": idx,
                    "mean_inner_neg_rmse": float(m),
                    "std_inner_neg_rmse": float(s),
                    "is_1se_selected": idx == chosen_idx,
                    "params": p,
                }
            )

    return NestedCVResult(outer_rows=outer_rows, tuning_rows=tuning_rows)


def tune_full_data_1se(
    estimator,
    param_grid: dict[str, Any],
    complexity_fn: ComplexityFn,
    X: pd.DataFrame,
    y: pd.Series,
    inner_splits: int,
    seed: int,
    n_jobs: int,
):
    cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs,
        refit=False,
    )
    search.fit(X, y)

    idx = select_1se_index(search.cv_results_, inner_splits, complexity_fn)
    params = search.cv_results_["params"][idx]
    model = clone(estimator).set_params(**params)
    model.fit(X, y)

    return model, params, search.cv_results_
