"""Multi-timeframe backtest with full metrics and Q-value recording."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config import ACTION_NAMES, ACTION_POSITION_RATIOS
from env.trading_env import MTFTradingEnv
from utils.metrics import calculate_metrics


def backtest(env: MTFTradingEnv, agent) -> Dict:
    """Run greedy backtest and return equity/actions/trades/q_values/metrics."""
    state = env.reset()
    actions = []
    q_records = []

    while True:
        q_values = agent.get_q_values(state)
        action = int(np.argmax(q_values))
        next_state, reward, done, info = env.step(action)

        actions.append({
            "step": env.step_idx,
            "datetime": env.df.loc[env.step_idx, "datetime"],
            "action": action,
            "action_name": ACTION_NAMES[action],
            "target_ratio": ACTION_POSITION_RATIOS[action],
            "portfolio_value": info["portfolio_value"],
            "period_return": info["period_return"],
            "drawdown": info["drawdown"],
            "mtf_confluence_score": info["mtf_confluence_score"],
        })
        q_records.append(q_values)
        state = next_state
        if done:
            break

    equity_df = pd.DataFrame(env.equity_curve)
    actions_df = pd.DataFrame(actions)
    trades_df = pd.DataFrame(env.trades)
    q_df = pd.DataFrame(q_records, columns=[f"Q_{ACTION_NAMES[i]}" for i in range(len(ACTION_NAMES))])

    metrics = calculate_metrics(equity_df, trades_df, env.initial_cash)

    return {
        "equity_df": equity_df,
        "actions_df": actions_df,
        "trades_df": trades_df,
        "q_df": q_df,
        "metrics": metrics,
    }
