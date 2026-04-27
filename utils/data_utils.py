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
    
    # 計算 SMC 特徵
    try:
        from smartmoneyconcepts import smc
        # 需要確定傳入的 df index 是 datetime 或連續數字，smc 用 Series
        df_smc = df.copy()
        shl = smc.swing_highs_lows(df_smc)
        fvg = smc.fvg(df_smc)
        ob = smc.ob(df_smc, shl)
        liq = smc.liquidity(df_smc, shl)
        phl = smc.previous_high_low(df_smc)
        
        # FVG 距離特徵: 分為 Bullish(1) / Bearish(-1)
        # 用前值填充以保留最近的 FVG
        df["fvg"] = fvg["FVG"].fillna(0)
        df["fvg_top"] = fvg["Top"]
        df["fvg_bottom"] = fvg["Bottom"]
        
        # Order Block: 
        # OBVolume, Percentage 等
        df["ob"] = ob["OB"].fillna(0)
        df["ob_top"] = ob["Top"]
        df["ob_bottom"] = ob["Bottom"]
        
        # Liquidity Swept 紀錄 (獵取流動性)
        df["liq_swept"] = liq["Swept"].fillna(0)
        df["liq_level"] = liq["Level"]
        
        # Old Highs / Lows 距離 (正規化)
        df["old_high"] = phl["PreviousHigh"].ffill().fillna(df["high"])
        df["old_low"] = phl["PreviousLow"].ffill().fillna(df["low"])
        
        # 將盤面絕對數值轉為「相對距離百分比」，DRL 才看得懂
        df["dist_old_high"] = (df["old_high"] - df["close"]) / df["close"] * 100
        df["dist_old_low"] = (df["close"] - df["old_low"]) / df["close"] * 100
        
    except ImportError:
        print("未安裝 smartmoneyconcepts，將使用原本邏輯")
        df["fvg"] = 0
        df["ob"] = 0
        df["liq_swept"] = 0
        df["dist_old_high"] = 0
        df["dist_old_low"] = 0

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

    # 避免一開始的 NaN 把資料刪光：使用 bfill 向前填充前幾天的滾動空值
    fill_cols = ["range_high", "range_low", "pd_pos", "close_norm"]
    if "dist_old_high" in df.columns:
        fill_cols.extend(["dist_old_high", "dist_old_low"])
        
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].bfill().fillna(0) # 第一道防線填充

    subset = ["date", "close", "close_norm", "range_high", "range_low", "pd_pos"]
    if "dist_old_high" in df.columns:
        subset.extend(["dist_old_high", "dist_old_low"])
    
    df = df.dropna(subset=subset).reset_index(drop=True)
    
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
