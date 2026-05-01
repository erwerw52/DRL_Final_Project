"""K-line chart with PD Array overlay using MTF data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from utils.data_utils import download_and_build_mtf, ensure_datetime_index


def draw_candlesticks(ax, df: pd.DataFrame):
    colors = np.where(df['close'] >= df['open'], 'red', 'green')
    x = np.arange(len(df))
    ax.vlines(x, df['low'], df['high'], color=colors, linewidth=1)
    bottoms = np.minimum(df['open'], df['close'])
    heights = np.abs(df['close'] - df['open'])
    heights = np.where(heights == 0, 0.5, heights)
    ax.bar(x, heights, bottom=bottoms, color=colors, width=0.6, align='center')
    return x


def overlay_pd_array(ax, x, df: pd.DataFrame):
    if 'h1_range_high' in df.columns:
        rh = df['h1_range_high'].values
        rl = df['h1_range_low'].values
    else:
        rh = df['high'].rolling(60).max().values
        rl = df['low'].rolling(60).min().values
    mid = (rh + rl) / 2.0
    ax.plot(x, rh, color='darkred', linestyle='--', alpha=0.6, label='Range High')
    ax.plot(x, rl, color='darkgreen', linestyle='--', alpha=0.6, label='Range Low')
    ax.plot(x, mid, color='gray', linestyle=':', alpha=0.8, label='Equilibrium')
    ax.fill_between(x, mid, rh, color='lightcoral', alpha=0.15, label='Premium Zone')
    ax.fill_between(x, rl, mid, color='lightgreen', alpha=0.15, label='Discount Zone')


def main():
    cfg = Config()
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {cfg.ticker} data...")
    mtf_df, _, _ = download_and_build_mtf(cfg)

    LOOKBACK = 150
    df_subset = mtf_df.tail(LOOKBACK).copy().reset_index()
    if 'index' in df_subset.columns:
        df_subset.rename(columns={'index': 'datetime'}, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = draw_candlesticks(ax, df_subset)
    overlay_pd_array(ax, x, df_subset)

    step = max(1, len(x) // 15)
    if 'datetime' in df_subset.columns:
        labels = pd.to_datetime(df_subset['datetime']).dt.strftime('%Y-%m-%d %H:%M')
    else:
        labels = [str(i) for i in range(len(df_subset))]
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels.iloc[::step], rotation=45, ha='right')

    ax.set_title(f"{cfg.ticker} MTF PD Array & K-line (Last {LOOKBACK} bars)")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = cfg.outputs_dir / "kline_pd_array.png"
    plt.savefig(out_path, dpi=200)
    print(f"Chart saved to: {out_path}")


if __name__ == "__main__":
    main()
