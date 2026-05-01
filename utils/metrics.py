"""Performance metrics for multi-timeframe trading backtest."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def calculate_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_cash: float,
) -> Dict:
    """Calculate comprehensive backtest metrics."""
    if equity_df.empty:
        return {}

    final_value = equity_df["portfolio_value"].iloc[-1]
    total_return = final_value / initial_cash - 1.0

    returns = equity_df["portfolio_value"].pct_change().dropna()
    max_drawdown = equity_df["drawdown"].max()

    # 1H annualized Sharpe: ~252 trading days × ~6.5 hours/day
    sharpe = np.nan
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5)

    win_periods = (returns > 0).sum()
    total_periods = len(returns)
    period_win_rate = win_periods / total_periods if total_periods > 0 else np.nan

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    trade_win_rate = np.nan
    completed_trade_returns = []

    if not trades_df.empty and "type" in trades_df.columns:
        buy_price = None
        for _, row in trades_df.iterrows():
            if row["type"] == "BUY" and buy_price is None:
                buy_price = row["price"]
            elif row["type"] == "SELL" and buy_price is not None:
                completed_trade_returns.append(row["price"] / buy_price - 1.0)
                buy_price = None
        if completed_trade_returns:
            trade_win_rate = np.mean([r > 0 for r in completed_trade_returns])

    return {
        "final_value": final_value,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "period_win_rate": period_win_rate,
        "trade_win_rate": trade_win_rate,
        "profit_factor": profit_factor,
        "num_trades": 0 if trades_df.empty else len(trades_df),
    }


def compute_performance(equity_curve: List[float]) -> Dict[str, float]:
    """Legacy compat: simple equity curve metrics."""
    import math
    if len(equity_curve) < 2:
        return {"cumulative_return": 0.0, "annualized_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
    equity = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(equity) / np.clip(equity[:-1], 1e-8, None)
    cumulative_return = equity[-1] / equity[0] - 1.0
    periods_per_year = 252
    n_periods = len(returns)
    annualized_return = (equity[-1] / equity[0]) ** (periods_per_year / max(n_periods, 1)) - 1.0
    ret_std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    sharpe_ratio = (returns.mean() / ret_std) * math.sqrt(periods_per_year) if ret_std > 1e-12 else 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / np.clip(running_max, 1e-8, None)
    max_drawdown = drawdowns.min()
    return {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
    }
