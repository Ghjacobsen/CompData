from __future__ import annotations

from typing import Any, Callable

import numpy as np


ComplexityFn = Callable[[dict[str, Any]], tuple]


def select_1se_index(
    cv_results: dict[str, Any],
    n_splits_inner: int,
    complexity_fn: ComplexityFn,
) -> int:
    means = np.asarray(cv_results["mean_test_score"], dtype=float)
    stds = np.asarray(cv_results["std_test_score"], dtype=float)

    finite_mask = np.isfinite(means)
    if not np.any(finite_mask):
        raise ValueError(
            "All inner-CV candidate scores are non-finite. "
            "Check estimator fit/predict behavior and parameter grid validity."
        )

    means_safe = means.copy()
    means_safe[~finite_mask] = -np.inf

    idx_best = int(np.argmax(means_safe))
    best_mean = float(means[idx_best])
    best_std = float(stds[idx_best]) if np.isfinite(stds[idx_best]) else 0.0
    se = best_std / np.sqrt(max(1, n_splits_inner))
    threshold = best_mean - se

    candidate_idx = np.where((means >= threshold) & finite_mask)[0]
    if candidate_idx.size == 0:
        candidate_idx = np.array([idx_best], dtype=int)

    params_list = cv_results["params"]
    ranked = sorted(candidate_idx, key=lambda i: complexity_fn(params_list[int(i)]))
    return int(ranked[0])
