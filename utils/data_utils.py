"""Multi-timeframe data utilities for W1+D1+H4+H1 SMC+DQN system."""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import MACD


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


# ── FEATURE_COLUMNS ──
FEATURE_COLUMNS = [
    "w1_return_1", "w1_return_5", "w1_ma_gap_20", "w1_ma_gap_60",
    "w1_rsi", "w1_atr_pct", "w1_macd_diff", "w1_range_position",
    "w1_premium_zone", "w1_discount_zone", "w1_structure_direction", "w1_smc_bias",
    "d1_return_1", "d1_return_5", "d1_volume_change", "d1_ma_gap_20", "d1_ma_gap_60",
    "d1_rsi", "d1_atr_pct", "d1_macd_diff", "d1_range_position",
    "d1_premium_zone", "d1_discount_zone", "d1_structure_direction", "d1_smc_bias",
    "h4_return_1", "h4_return_5", "h4_ma_gap_20", "h4_rsi", "h4_atr_pct",
    "h4_macd_diff", "h4_bos_bullish", "h4_bos_bearish",
    "h4_choch_bullish", "h4_choch_bearish",
    "h4_liquidity_sweep_high", "h4_liquidity_sweep_low",
    "h4_bullish_fvg", "h4_bearish_fvg",
    "h4_bullish_ob_distance", "h4_bearish_ob_distance",
    "h4_range_position", "h4_structure_direction", "h4_smc_bias",
    "h1_return_1", "h1_return_5", "h1_volume_change",
    "h1_ma_gap_10", "h1_ma_gap_20", "h1_rsi", "h1_atr_pct", "h1_macd_diff",
    "h1_bos_bullish", "h1_bos_bearish",
    "h1_choch_bullish", "h1_choch_bearish",
    "h1_liquidity_sweep_high", "h1_liquidity_sweep_low",
    "h1_bullish_fvg", "h1_bearish_fvg",
    "h1_bullish_ob_distance", "h1_bearish_ob_distance",
    "h1_range_position", "h1_structure_direction", "h1_smc_bias",
    "mtf_bullish_score", "mtf_bearish_score", "mtf_confluence_score",
    "mtf_conflict", "mtf_all_bullish", "mtf_all_bearish",
    "higher_tf_bullish", "higher_tf_bearish",
]


# ── Datetime helpers ──

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df.sort_index()
    df.index.name = None
    return df


def reset_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_datetime_index(df)
    out = df.copy()
    out["datetime"] = out.index
    out = out.reset_index(drop=True)
    out = out.sort_values("datetime")
    return out


# ── OHLCV helpers ──

def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        if "Open" in level0 or "Close" in level0:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}. Available: {list(df.columns)}")
    df = df[required].copy().dropna()
    df = ensure_datetime_index(df)
    return df


def download_ohlcv_basic(ticker, start=None, end=None, interval="1d", period=None):
    kwargs = {"tickers": ticker, "interval": interval, "auto_adjust": True, "progress": False, "threads": True}
    if period is not None:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    df = yf.download(**kwargs)
    return normalize_ohlcv_columns(df)


def download_ohlcv_with_fallback(ticker, start, end, interval, fallback_periods=("730d", "365d", "180d", "90d", "60d")):
    print(f"Downloading {ticker}, interval={interval}, start={start}, end={end} ...")
    try:
        df = download_ohlcv_basic(ticker=ticker, start=start, end=end, interval=interval)
        if not df.empty:
            print(f"Downloaded by start/end: rows={len(df)}, {df.index.min()} -> {df.index.max()}")
            return df
    except Exception as e:
        print(f"Start/end download failed: {e}")

    if interval.lower() in {"1h", "60m", "30m", "15m", "5m", "1m"}:
        alt_intervals = [interval]
        if interval.lower() == "1h":
            alt_intervals.append("60m")
        elif interval.lower() == "60m":
            alt_intervals.append("1h")
        for alt in alt_intervals:
            for period in fallback_periods:
                try:
                    print(f"Trying fallback: interval={alt}, period={period} ...")
                    df = download_ohlcv_basic(ticker=ticker, interval=alt, period=period)
                    if not df.empty:
                        print(f"Fallback OK: rows={len(df)}")
                        return df
                except Exception as e:
                    print(f"Fallback failed: {e}")

    if interval.lower() in {"1d", "1wk"}:
        for period in fallback_periods:
            try:
                df = download_ohlcv_basic(ticker=ticker, interval=interval, period=period)
                if not df.empty:
                    return df
            except Exception:
                pass

    raise ValueError(f"No data downloaded for {ticker}, interval={interval}.")


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = ensure_datetime_index(df)
    out = pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum(),
    }).dropna()
    return ensure_datetime_index(out)


# ── Indicators & SMC features ──

def add_basic_indicators(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = ensure_datetime_index(df).copy()
    df[f"{prefix}_return_1"] = df["close"].pct_change()
    df[f"{prefix}_return_5"] = df["close"].pct_change(5)
    df[f"{prefix}_volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    df[f"{prefix}_sma_10"] = df["close"].rolling(10).mean()
    df[f"{prefix}_sma_20"] = df["close"].rolling(20).mean()
    df[f"{prefix}_sma_60"] = df["close"].rolling(60).mean()
    df[f"{prefix}_ma_gap_10"] = df["close"] / df[f"{prefix}_sma_10"] - 1
    df[f"{prefix}_ma_gap_20"] = df["close"] / df[f"{prefix}_sma_20"] - 1
    df[f"{prefix}_ma_gap_60"] = df["close"] / df[f"{prefix}_sma_60"] - 1
    df[f"{prefix}_rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df[f"{prefix}_atr"] = atr.average_true_range()
    df[f"{prefix}_atr_pct"] = df[f"{prefix}_atr"] / df["close"]
    macd = MACD(close=df["close"])
    df[f"{prefix}_macd"] = macd.macd()
    df[f"{prefix}_macd_signal"] = macd.macd_signal()
    df[f"{prefix}_macd_diff"] = macd.macd_diff()
    return df.replace([np.inf, -np.inf], np.nan)


def add_smc_features(df: pd.DataFrame, prefix: str, swing_window: int = 5, lookback_range: int = 60) -> pd.DataFrame:
    df = ensure_datetime_index(df).copy()
    shc = f"{prefix}_swing_high"
    slc = f"{prefix}_swing_low"
    lhc = f"{prefix}_last_swing_high"
    llc = f"{prefix}_last_swing_low"

    df[shc] = 0
    df[slc] = 0
    for i in range(swing_window, len(df) - swing_window):
        local_high = df["high"].iloc[i - swing_window:i + swing_window + 1].max()
        local_low = df["low"].iloc[i - swing_window:i + swing_window + 1].min()
        if df["high"].iloc[i] == local_high:
            df.iloc[i, df.columns.get_loc(shc)] = 1
        if df["low"].iloc[i] == local_low:
            df.iloc[i, df.columns.get_loc(slc)] = 1

    df[lhc] = np.nan
    df[llc] = np.nan
    last_high = last_low = np.nan
    for i in range(len(df)):
        if df[shc].iloc[i] == 1:
            last_high = df["high"].iloc[i]
        if df[slc].iloc[i] == 1:
            last_low = df["low"].iloc[i]
        df.iloc[i, df.columns.get_loc(lhc)] = last_high
        df.iloc[i, df.columns.get_loc(llc)] = last_low

    # BOS
    df[f"{prefix}_bos_bullish"] = ((df["close"] > df[lhc].shift(1)) & df[lhc].shift(1).notna()).astype(int)
    df[f"{prefix}_bos_bearish"] = ((df["close"] < df[llc].shift(1)) & df[llc].shift(1).notna()).astype(int)

    # Structure direction
    df[f"{prefix}_structure_direction"] = 0
    current_dir = 0
    for i in range(len(df)):
        if df[f"{prefix}_bos_bullish"].iloc[i] == 1:
            current_dir = 1
        elif df[f"{prefix}_bos_bearish"].iloc[i] == 1:
            current_dir = -1
        df.iloc[i, df.columns.get_loc(f"{prefix}_structure_direction")] = current_dir

    # CHOCH
    df[f"{prefix}_choch_bullish"] = ((df[f"{prefix}_structure_direction"].shift(1) == -1) & (df[f"{prefix}_bos_bullish"] == 1)).astype(int)
    df[f"{prefix}_choch_bearish"] = ((df[f"{prefix}_structure_direction"].shift(1) == 1) & (df[f"{prefix}_bos_bearish"] == 1)).astype(int)

    # Liquidity Sweep
    df[f"{prefix}_prev_high_20"] = df["high"].rolling(20).max().shift(1)
    df[f"{prefix}_prev_low_20"] = df["low"].rolling(20).min().shift(1)
    df[f"{prefix}_liquidity_sweep_high"] = ((df["high"] > df[f"{prefix}_prev_high_20"]) & (df["close"] < df[f"{prefix}_prev_high_20"])).astype(int)
    df[f"{prefix}_liquidity_sweep_low"] = ((df["low"] < df[f"{prefix}_prev_low_20"]) & (df["close"] > df[f"{prefix}_prev_low_20"])).astype(int)

    # Premium / Discount
    df[f"{prefix}_range_high"] = df["high"].rolling(lookback_range).max()
    df[f"{prefix}_range_low"] = df["low"].rolling(lookback_range).min()
    df[f"{prefix}_range_position"] = (df["close"] - df[f"{prefix}_range_low"]) / (df[f"{prefix}_range_high"] - df[f"{prefix}_range_low"])
    df[f"{prefix}_premium_zone"] = (df[f"{prefix}_range_position"] > 0.5).astype(int)
    df[f"{prefix}_discount_zone"] = (df[f"{prefix}_range_position"] <= 0.5).astype(int)

    # FVG
    df[f"{prefix}_bullish_fvg"] = (df["low"] > df["high"].shift(2)).astype(int)
    df[f"{prefix}_bearish_fvg"] = (df["high"] < df["low"].shift(2)).astype(int)

    # OB distance
    df[f"{prefix}_bullish_ob_price"] = np.nan
    df[f"{prefix}_bearish_ob_price"] = np.nan
    last_bull_ob = last_bear_ob = np.nan
    for i in range(1, len(df)):
        if df["close"].iloc[i - 1] < df["open"].iloc[i - 1]:
            last_bull_ob = df["low"].iloc[i - 1]
        if df["close"].iloc[i - 1] > df["open"].iloc[i - 1]:
            last_bear_ob = df["high"].iloc[i - 1]
        df.iloc[i, df.columns.get_loc(f"{prefix}_bullish_ob_price")] = last_bull_ob
        df.iloc[i, df.columns.get_loc(f"{prefix}_bearish_ob_price")] = last_bear_ob

    df[f"{prefix}_bullish_ob_distance"] = (df["close"] - df[f"{prefix}_bullish_ob_price"]) / df["close"]
    df[f"{prefix}_bearish_ob_distance"] = (df[f"{prefix}_bearish_ob_price"] - df["close"]) / df["close"]

    # SMC score
    df[f"{prefix}_smc_bull_score"] = (
        df[f"{prefix}_bos_bullish"] + df[f"{prefix}_choch_bullish"]
        + df[f"{prefix}_liquidity_sweep_low"] + df[f"{prefix}_discount_zone"]
        + df[f"{prefix}_bullish_fvg"]
    )
    df[f"{prefix}_smc_bear_score"] = (
        df[f"{prefix}_bos_bearish"] + df[f"{prefix}_choch_bearish"]
        + df[f"{prefix}_liquidity_sweep_high"] + df[f"{prefix}_premium_zone"]
        + df[f"{prefix}_bearish_fvg"]
    )
    df[f"{prefix}_smc_bias"] = df[f"{prefix}_smc_bull_score"] - df[f"{prefix}_smc_bear_score"]

    return df.replace([np.inf, -np.inf], np.nan)


def prepare_timeframe_features(df, prefix, cfg):
    df = ensure_datetime_index(df)
    df = add_basic_indicators(df, prefix=prefix)
    df = add_smc_features(df, prefix=prefix, swing_window=cfg.swing_window, lookback_range=cfg.lookback_range)
    return df


# ── MTF alignment ──

def select_prefixed_feature_columns(df, prefix):
    exclude = {"open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c.startswith(f"{prefix}_") and c not in exclude]


def merge_asof_higher_tf(base_df, higher_df, higher_feature_cols):
    base = reset_datetime_index(base_df)
    higher = higher_df[higher_feature_cols].copy()
    higher = reset_datetime_index(higher)
    base = base.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    higher = higher.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    merged = pd.merge_asof(base, higher, on="datetime", direction="backward")
    merged = merged.set_index("datetime")
    merged.index.name = None
    return merged


def add_mtf_confluence_features(df):
    df = df.copy()
    df["mtf_bullish_score"] = (
        3.0 * (df["w1_smc_bias"] > 0).astype(int)
        + 2.0 * (df["d1_smc_bias"] > 0).astype(int)
        + 1.5 * (df["h4_smc_bias"] > 0).astype(int)
        + 1.0 * (df["h1_smc_bias"] > 0).astype(int)
    )
    df["mtf_bearish_score"] = (
        3.0 * (df["w1_smc_bias"] < 0).astype(int)
        + 2.0 * (df["d1_smc_bias"] < 0).astype(int)
        + 1.5 * (df["h4_smc_bias"] < 0).astype(int)
        + 1.0 * (df["h1_smc_bias"] < 0).astype(int)
    )
    df["mtf_confluence_score"] = df["mtf_bullish_score"] - df["mtf_bearish_score"]
    df["mtf_conflict"] = (
        ((df["w1_smc_bias"] > 0) & (df["h1_smc_bias"] < 0))
        | ((df["w1_smc_bias"] < 0) & (df["h1_smc_bias"] > 0))
        | ((df["d1_smc_bias"] > 0) & (df["h1_smc_bias"] < 0))
        | ((df["d1_smc_bias"] < 0) & (df["h1_smc_bias"] > 0))
    ).astype(int)
    df["mtf_all_bullish"] = ((df["w1_smc_bias"] > 0) & (df["d1_smc_bias"] > 0) & (df["h4_smc_bias"] > 0) & (df["h1_smc_bias"] > 0)).astype(int)
    df["mtf_all_bearish"] = ((df["w1_smc_bias"] < 0) & (df["d1_smc_bias"] < 0) & (df["h4_smc_bias"] < 0) & (df["h1_smc_bias"] < 0)).astype(int)
    df["higher_tf_bullish"] = ((df["w1_smc_bias"] > 0) & (df["d1_smc_bias"] > 0)).astype(int)
    df["higher_tf_bearish"] = ((df["w1_smc_bias"] < 0) & (df["d1_smc_bias"] < 0)).astype(int)
    return df


# ── Data split & standardization ──

def split_data_time_order(df, cfg):
    n = len(df)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def fit_standardizer(train_df, feature_cols):
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)
    return mean, std


def apply_standardizer(df, feature_cols, mean, std):
    df = df.copy()
    df[feature_cols] = (df[feature_cols] - mean) / std
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    return df


# ── Full MTF pipeline ──

def build_mtf_dataset(df_h1_raw, df_d1_raw, cfg):
    """Build W1+D1+H4+H1 MTF dataset from raw H1 and D1 data."""
    df_h1_raw = ensure_datetime_index(df_h1_raw)
    df_d1_raw = ensure_datetime_index(df_d1_raw)
    df_h4_raw = resample_ohlcv(df_h1_raw, "4h")
    df_w1_raw = resample_ohlcv(df_d1_raw, "1W")

    df_h1_feat = prepare_timeframe_features(df_h1_raw, "h1", cfg)
    df_h4_feat = prepare_timeframe_features(df_h4_raw, "h4", cfg)
    df_d1_feat = prepare_timeframe_features(df_d1_raw, "d1", cfg)
    df_w1_feat = prepare_timeframe_features(df_w1_raw, "w1", cfg)

    h4_cols = select_prefixed_feature_columns(df_h4_feat, "h4")
    d1_cols = select_prefixed_feature_columns(df_d1_feat, "d1")
    w1_cols = select_prefixed_feature_columns(df_w1_feat, "w1")

    out = merge_asof_higher_tf(df_h1_feat, df_h4_feat, h4_cols)
    out = merge_asof_higher_tf(out, df_d1_feat, d1_cols)
    out = merge_asof_higher_tf(out, df_w1_feat, w1_cols)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.bfill().ffill().fillna(0) # 避免新標的沒有長線資料時被 dropna 刪光
    
    out = add_mtf_confluence_features(out)
    out = out.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0)
    
    # 將 MTF 訓練區間切齊回使用者指定的原 start_date
    start_dt = pd.to_datetime(cfg.start_date)
    if start_dt.tz is None and out.index.tz is not None:
        start_dt = start_dt.tz_localize(out.index.tz)
    out = out[out.index >= start_dt]
    
    return out


def download_and_build_mtf(cfg, progress_callback=None):
    """Download data and build full MTF dataset."""
    if progress_callback:
        progress_callback("Downloading H1 data...")
        
    # 往前多抓 60 天的資料，作為計算 H1 / H4 指標 (如 sma_60 等) 的暖機期 (Warm-up)，
    # 確保不會因為 dropna 連帶刪掉了使用者指定的分析區間。
    h1_fetch_start = str((pd.to_datetime(cfg.start_date) - pd.Timedelta(days=60)).date())
    
    df_h1_raw = download_ohlcv_with_fallback(
        ticker=cfg.ticker, start=h1_fetch_start, end=cfg.end_date,
        interval=cfg.base_interval, fallback_periods=cfg.intraday_fallback_periods,
    )

    daily_start = str((df_h1_raw.index.min() - pd.Timedelta(days=cfg.daily_extra_days)).date())
    daily_end = str((df_h1_raw.index.max() + pd.Timedelta(days=2)).date())

    if progress_callback:
        progress_callback("Downloading D1 data...")
    df_d1_raw = download_ohlcv_with_fallback(
        ticker=cfg.ticker, start=daily_start, end=daily_end,
        interval="1d", fallback_periods=("10y", "5y", "2y", "1y"),
    )

    if progress_callback:
        progress_callback("Building MTF features...")
    mtf_df = build_mtf_dataset(df_h1_raw, df_d1_raw, cfg)

    missing = [c for c in FEATURE_COLUMNS if c not in mtf_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return mtf_df, df_h1_raw, df_d1_raw


# ── Legacy compat for app.py SMC chart (smartmoneyconcepts) ──

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def prepare_data_for_chart(df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """Prepare data with smartmoneyconcepts features for chart display only."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    try:
        from smartmoneyconcepts import smc
        df_smc = df.copy()
        shl = smc.swing_highs_lows(df_smc)
        fvg = smc.fvg(df_smc)
        ob = smc.ob(df_smc, shl)
        liq = smc.liquidity(df_smc, shl)
        phl = smc.previous_high_low(df_smc)
        df["fvg"] = fvg["FVG"].fillna(0)
        df["fvg_top"] = fvg["Top"]
        df["fvg_bottom"] = fvg["Bottom"]
        df["ob"] = ob["OB"].fillna(0)
        df["ob_top"] = ob["Top"]
        df["ob_bottom"] = ob["Bottom"]
        df["liq_swept"] = liq["Swept"].fillna(0)
        df["liq_level"] = liq["Level"]
        df["old_high"] = phl["PreviousHigh"].ffill().fillna(df["high"])
        df["old_low"] = phl["PreviousLow"].ffill().fillna(df["low"])
        df["dist_old_high"] = (df["old_high"] - df["close"]) / df["close"] * 100
        df["dist_old_low"] = (df["close"] - df["old_low"]) / df["close"] * 100
    except ImportError:
        print("smartmoneyconcepts not installed, chart SMC features unavailable")
        df["fvg"] = 0
        df["fvg_top"] = np.nan
        df["fvg_bottom"] = np.nan
        df["ob"] = 0
        df["ob_top"] = np.nan
        df["ob_bottom"] = np.nan
        df["liq_swept"] = 0
        df["liq_level"] = np.nan
        df["old_high"] = df["high"].cummax()
        df["old_low"] = df["low"].cummin()
        df["dist_old_high"] = (df["old_high"] - df["close"]) / df["close"] * 100
        df["dist_old_low"] = (df["close"] - df["old_low"]) / df["close"] * 100

    df["range_high"] = df["high"].rolling(rolling_window).max()
    df["range_low"] = df["low"].rolling(rolling_window).min()
    denom = (df["range_high"] - df["range_low"]).replace(0, np.nan)
    df["pd_pos"] = ((df["close"] - df["range_low"]) / denom).clip(0.0, 1.0)
    df["is_premium"] = (df["pd_pos"] > 0.5).astype(int)
    df["is_discount"] = (df["pd_pos"] < 0.5).astype(int)

    fill_cols = ["range_high", "range_low", "pd_pos"]
    for col in fill_cols:
        df[col] = df[col].bfill().fillna(0)

    df = df.dropna(subset=["date", "close"]).reset_index(drop=True)
    return df
