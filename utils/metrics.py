from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def compute_performance(equity_curve: List[float]) -> Dict[str, float]:
    if len(equity_curve) < 2:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    equity = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(equity) / np.clip(equity[:-1], 1e-8, None)

    cumulative_return = equity[-1] / equity[0] - 1.0

    periods_per_year = 252
    n_periods = len(returns)
    annualized_return = (equity[-1] / equity[0]) ** (periods_per_year / max(n_periods, 1)) - 1.0

    ret_std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    sharpe_ratio = 0.0
    if ret_std > 1e-12:
        sharpe_ratio = (returns.mean() / ret_std) * math.sqrt(periods_per_year)

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / np.clip(running_max, 1e-8, None)
    max_drawdown = drawdowns.min()

    return {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
    }
