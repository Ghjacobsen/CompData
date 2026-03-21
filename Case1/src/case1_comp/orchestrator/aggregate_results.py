from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from case1_comp.common.artifacts import ensure_dir, write_submission_rmse_estimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("Case1/results/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("Case1/results"))
    parser.add_argument("--student-no", type=str, default="YourStudentNo")
    parser.add_argument("--rmse-margin", type=float, default=0.2)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    approach_dirs = [p for p in args.runs_root.iterdir() if p.is_dir()]
    rows = []
    for approach_dir in approach_dirs:
        nested_summary = load_summary(approach_dir / "nested" / "summary.json")
        rows.append(
            {
                "approach": nested_summary["approach"],
                "mean_outer_rmse": float(nested_summary["mean_outer_rmse"]),
                "std_outer_rmse": float(nested_summary["std_outer_rmse"]),
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values(["mean_outer_rmse", "std_outer_rmse"])
    winner = leaderboard.iloc[0]
    winner_name = winner["approach"]

    pred_src = args.runs_root / winner_name / "refit" / "predictions.csv"
    pred_dst = args.output_dir / f"sample_predictions_{args.student_no}.csv"
    pd.read_csv(pred_src).to_csv(pred_dst, index=False)

    rmse_estimate = float(winner["mean_outer_rmse"]) + args.rmse_margin * float(winner["std_outer_rmse"])
    rmse_dst = args.output_dir / f"sample_estimatedRMSE_{args.student_no}.csv"
    write_submission_rmse_estimate(rmse_estimate, rmse_dst)

    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    with (args.output_dir / "winner.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "winner": winner_name,
                "mean_outer_rmse": float(winner["mean_outer_rmse"]),
                "std_outer_rmse": float(winner["std_outer_rmse"]),
                "rmse_estimate": rmse_estimate,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
