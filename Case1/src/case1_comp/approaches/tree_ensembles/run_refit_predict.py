from __future__ import annotations

import argparse
from pathlib import Path

from case1_comp.approaches.tree_ensembles.config import APPROACH_NAME, DEFAULT_N_JOBS, INNER_SPLITS
from case1_comp.approaches.tree_ensembles.model_factory import build_estimator, complexity_key
from case1_comp.approaches.tree_ensembles.search_space import get_param_grid
from case1_comp.common.artifacts import ensure_dir, now_utc_iso, write_json, write_submission_predictions
from case1_comp.common.data import load_case1_data
from case1_comp.common.nested_cv import tune_full_data_1se


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Case1/data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
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

    model, params, cv_results = tune_full_data_1se(
        estimator=estimator,
        param_grid=param_grid,
        complexity_fn=complexity_key,
        X=data.X_train,
        y=data.y_train,
        inner_splits=args.inner_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    preds = model.predict(data.X_new)
    write_submission_predictions(preds, args.out_dir / "predictions.csv")
    write_json(params, args.out_dir / "chosen_config.json")
    write_json(
        {
            "approach": APPROACH_NAME,
            "seed": args.seed,
            "inner_splits": args.inner_splits,
            "n_candidates": len(cv_results["params"]),
            "created_utc": now_utc_iso(),
        },
        args.out_dir / "refit_metadata.json",
    )


if __name__ == "__main__":
    main()
