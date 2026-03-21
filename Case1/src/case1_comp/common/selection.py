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

    idx_best = int(np.argmax(means))
    best_mean = means[idx_best]
    se = stds[idx_best] / np.sqrt(max(1, n_splits_inner))
    threshold = best_mean - se

    candidate_idx = np.where(means >= threshold)[0]

    params_list = cv_results["params"]
    ranked = sorted(candidate_idx, key=lambda i: complexity_fn(params_list[int(i)]))
    return int(ranked[0])
