from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Case1Data:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_new: pd.DataFrame
    numeric_cols: list[str]
    categorical_cols: list[str]


def load_case1_data(data_dir: Path) -> Case1Data:
    train_path = data_dir / "case1Data.csv"
    new_path = data_dir / "case1Data_Xnew.csv"

    train_df = pd.read_csv(train_path)
    new_df = pd.read_csv(new_path)

    y_col = "y"
    feature_cols = [c for c in train_df.columns if c != y_col]

    numeric_cols = [c for c in feature_cols if c.startswith("x_")]
    categorical_cols = [c for c in feature_cols if c.startswith("C_")]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_new = new_df[feature_cols].copy()

    return Case1Data(
        X_train=X_train,
        y_train=y_train,
        X_new=X_new,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
