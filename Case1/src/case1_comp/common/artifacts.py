from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_submission_predictions(predictions, path: Path) -> None:
    values = [float(v) for v in predictions]
    pd.DataFrame(values).to_csv(path, index=False, header=False)


def write_submission_rmse_estimate(estimate: float, path: Path) -> None:
    pd.DataFrame([float(estimate)]).to_csv(path, index=False, header=False)
