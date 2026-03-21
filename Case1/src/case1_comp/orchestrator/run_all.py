from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from case1_comp.common.artifacts import ensure_dir
from case1_comp.orchestrator.registry import APPROACH_MODULES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Case1/data"))
    parser.add_argument("--runs-root", type=Path, default=Path("Case1/results/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("Case1/results"))
    parser.add_argument("--student-no", type=str, default="YourStudentNo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-splits", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--rmse-margin", type=float, default=0.2)
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    ensure_dir(args.runs_root)
    ensure_dir(args.output_dir)

    for approach, module_base in APPROACH_MODULES.items():
        approach_dir = args.runs_root / approach
        nested_dir = approach_dir / "nested"
        refit_dir = approach_dir / "refit"
        ensure_dir(nested_dir)
        ensure_dir(refit_dir)

        run_cmd(
            [
                sys.executable,
                "-m",
                f"{module_base}.run_nested_cv",
                "--data-dir",
                str(args.data_dir),
                "--out-dir",
                str(nested_dir),
                "--seed",
                str(args.seed),
                "--outer-splits",
                str(args.outer_splits),
                "--inner-splits",
                str(args.inner_splits),
                "--n-jobs",
                str(args.n_jobs),
            ]
        )

        run_cmd(
            [
                sys.executable,
                "-m",
                f"{module_base}.run_refit_predict",
                "--data-dir",
                str(args.data_dir),
                "--out-dir",
                str(refit_dir),
                "--seed",
                str(args.seed),
                "--inner-splits",
                str(args.inner_splits),
                "--n-jobs",
                str(args.n_jobs),
            ]
        )

    run_cmd(
        [
            sys.executable,
            "-m",
            "case1_comp.orchestrator.aggregate_results",
            "--runs-root",
            str(args.runs_root),
            "--output-dir",
            str(args.output_dir),
            "--student-no",
            args.student_no,
            "--rmse-margin",
            str(args.rmse_margin),
        ]
    )


if __name__ == "__main__":
    main()
