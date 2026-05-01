"""Future scenario analysis using the MTF DQN model."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agent.dqn_agent import DQNAgent
from config import Config, ACTION_NAMES, ACTION_POSITION_RATIOS
from recommend import recommend_strategy
from utils.data_utils import (
    FEATURE_COLUMNS, download_and_build_mtf,
    fit_standardizer, apply_standardizer, split_data_time_order,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str):
    """Load trained MTF model and return agent + metadata."""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    feature_cols = checkpoint["feature_columns"]
    feature_mean = pd.Series(checkpoint["feature_mean"])
    feature_std = pd.Series(checkpoint["feature_std"])

    state_dim = len(feature_cols) + 3  # +3 for portfolio features
    action_dim = len(checkpoint.get("action_position_ratios", ACTION_POSITION_RATIOS))

    cfg_dict = checkpoint.get("config", {})
    cfg = Config()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    agent = DQNAgent(state_dim, action_dim, cfg)
    agent.policy_net.load_state_dict(checkpoint["model_state_dict"])
    agent.target_net.load_state_dict(checkpoint["model_state_dict"])
    agent.epsilon = checkpoint.get("epsilon", 0.03)
    agent.policy_net.eval()
    agent.target_net.eval()

    return agent, cfg, feature_cols, feature_mean, feature_std


def main():
    cfg = Config()
    model_path = cfg.outputs_dir / "mtf_dqn_model.pth"

    print("\n" + "=" * 60)
    print("📈 MTF DQN+SMC 策略建議分析報告")
    print("=" * 60)

    if not model_path.exists():
        print(f"[錯誤] 模型不存在: {model_path}")
        print("請先執行 train.py 訓練模型。")
        return

    # 1. Load model
    agent, cfg, feature_cols, feature_mean, feature_std = load_model(str(model_path))
    print(f"✅ 模型載入成功 | Epsilon: {agent.epsilon:.4f}")

    # 2. Download latest data
    print(f"\n📥 下載 {cfg.ticker} 最新資料...")
    mtf_df, _, _ = download_and_build_mtf(cfg)
    print(f"MTF dataset rows: {len(mtf_df)}")

    # 3. Generate recommendation
    recommendation = recommend_strategy(
        agent=agent,
        latest_mtf_raw=mtf_df,
        cfg=cfg,
        feature_cols=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    # 4. Print results
    print(f"\n📊 【最新市場狀態】")
    print(f"股票代號: {recommendation['ticker']}")
    print(f"最新收盤價: {recommendation['latest_close']:.2f}")

    print(f"\n🎯 【模型建議】")
    print(f"建議動作: {recommendation['best_action_name']}")
    print(f"交易方向: {recommendation['trade_direction']}")
    print(f"目標部位比例: {recommendation['target_position_ratio']:.0%}")

    print(f"\n🌐 【多時區 SMC 快照】")
    snap = recommendation["mtf_snapshot"]
    print(f"W1 bias: {snap['w1_smc_bias']:.0f} | D1 bias: {snap['d1_smc_bias']:.0f}")
    print(f"H4 bias: {snap['h4_smc_bias']:.0f} | H1 bias: {snap['h1_smc_bias']:.0f}")
    print(f"MTF 共振分數: {snap['mtf_confluence_score']:.1f}")
    print(f"高時區衝突: {'是' if snap['mtf_conflict'] else '否'}")

    print(f"\n📐 【風險報酬計劃】")
    rr = recommendation["risk_reward_plan"]
    print(f"有效: {rr['risk_reward_valid']}")
    print(f"說明: {rr['risk_reward_note']}")
    if rr.get("risk_reward_valid"):
        print(f"進場價: {rr['entry_price']:.2f}")
        print(f"停損價: {rr['stop_loss_price']:.2f}")
        print(f"停利價: {rr['take_profit_price']:.2f}")
        print(f"RR Ratio: {rr['risk_reward_ratio']:.2f}")

    print(f"\n📊 【Q Values】")
    for name, q in recommendation["q_values"].items():
        print(f"  {name}: {q:.6f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
