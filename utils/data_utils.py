from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def prepare_data_from_df(df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Rolling dealing range for simplified PD Array
    df["range_high"] = df["high"].rolling(rolling_window).max()
    df["range_low"] = df["low"].rolling(rolling_window).min()

    denom = (df["range_high"] - df["range_low"]).replace(0, np.nan)
    df["pd_pos"] = (df["close"] - df["range_low"]) / denom
    df["pd_pos"] = df["pd_pos"].clip(0.0, 1.0)

    df["is_premium"] = (df["pd_pos"] > 0.5).astype(int)
    df["is_discount"] = (df["pd_pos"] < 0.5).astype(int)

    # Optional normalized price feature for stability
    df["close_norm"] = df["close"] / df["close"].rolling(rolling_window).mean()

    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable rows after rolling-window feature generation.")

    return df

def load_and_prepare_data(csv_path: str, rolling_window: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return prepare_data_from_df(df, rolling_window)


def split_data(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    if min(len(train_df), len(val_df), len(test_df)) < 5:
        raise ValueError("Dataset split too small. Please provide more rows.")

    return train_df, val_df, test_df
