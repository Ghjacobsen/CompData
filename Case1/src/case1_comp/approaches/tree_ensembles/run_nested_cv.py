from __future__ import annotations

import argparse
from pathlib import Path

from case1_comp.approaches.tree_ensembles.config import (
    APPROACH_NAME,
    DEFAULT_N_JOBS,
    INNER_SPLITS,
    OUTER_SPLITS,
)
from case1_comp.approaches.tree_ensembles.model_factory import build_estimator, complexity_key
from case1_comp.approaches.tree_ensembles.search_space import get_param_grid
from case1_comp.common.artifacts import ensure_dir, now_utc_iso, write_csv, write_json
from case1_comp.common.data import load_case1_data
from case1_comp.common.nested_cv import run_nested_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Case1/data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-splits", type=int, default=OUTER_SPLITS)
    parser.add_argument("--inner-splits", type=int, default=INNER_SPLITS)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    data = load_case1_data(args.data_dir)
    estimator = build_estimator(data.numeric_cols, data.categorical_cols, seed=args.seed)

    raw_grid = get_param_grid()
    param_grid = {f"reg__{k}": v for k, v in raw_grid.items()}

    result = run_nested_cv(
        estimator=estimator,
        param_grid=param_grid,
        complexity_fn=complexity_key,
        X=data.X_train,
        y=data.y_train,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    write_csv(result.outer_rows, args.out_dir / "metrics_outer.csv")
    write_csv(result.tuning_rows, args.out_dir / "tuning_trace.csv")
    write_json(
        {
            "approach": APPROACH_NAME,
            "mean_outer_rmse": result.mean_outer_rmse,
            "std_outer_rmse": result.std_outer_rmse,
            "created_utc": now_utc_iso(),
        },
        args.out_dir / "summary.json",
    )
    write_json(
        {
            "approach": APPROACH_NAME,
            "seed": args.seed,
            "outer_splits": args.outer_splits,
            "inner_splits": args.inner_splits,
            "n_jobs": args.n_jobs,
            "created_utc": now_utc_iso(),
        },
        args.out_dir / "run_metadata.json",
    )


if __name__ == "__main__":
    main()
