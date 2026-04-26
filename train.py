from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from agent.dqn_agent import DQNAgent
from config import Config
from env.trading_env import TradingEnv
from utils.data_utils import load_and_prepare_data, set_seed, split_data, prepare_data_from_df
from utils.replay_buffer import ReplayBuffer
import pandas as pd


def run_training_from_df(df: pd.DataFrame, cfg: Config) -> dict:
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    df_prepared = prepare_data_from_df(df, rolling_window=cfg.rolling_window)
    train_df, val_df, test_df = split_data(df_prepared, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)

    env = TradingEnv(
        data=train_df,
        initial_balance=cfg.initial_balance,
        transaction_cost_rate=cfg.transaction_cost_rate,
        pd_bonus=cfg.pd_bonus,
        reward_scale=cfg.reward_scale,
    )

    agent = DQNAgent(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        gamma=cfg.gamma,
        lr=cfg.lr,
        epsilon_start=cfg.epsilon_start,
        epsilon_decay=cfg.epsilon_decay,
        epsilon_min=cfg.epsilon_min,
    )
    replay_buffer = ReplayBuffer(capacity=cfg.replay_capacity)

    episode_rewards = []
    episode_losses = []
    training_logs = []

    for episode in range(cfg.episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.update(replay_buffer, batch_size=cfg.batch_size)
            if loss > 0:
                losses.append(loss)

            state = next_state
            total_reward += reward

        if (episode + 1) % cfg.target_update_freq == 0:
            agent.update_target()

        episode_rewards.append(total_reward)
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        episode_losses.append(avg_loss)
        
        log_line = (
            f"Episode {episode + 1:03d}/{cfg.episodes} | "
            f"Reward: {total_reward:.4f} | "
            f"Avg Loss: {avg_loss:.6f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )
        print(log_line)
        training_logs.append(log_line)

    agent.save(str(cfg.model_path))

    # Save split metadata
    meta = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "data_source": "dynamic_df"
    }
    with open(cfg.outputs_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Plot training rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward")
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.outputs_dir / "training_rewards.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episode_losses, label="Average Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.outputs_dir / "training_losses.png", dpi=150)
    plt.close()

    print(f"Model saved to: {cfg.model_path}")
    print(f"Artifacts saved in: {cfg.outputs_dir}")

    # Evaluate model on the test_df for final performance metrics
    test_env = TradingEnv(
        data=test_df,
        initial_balance=cfg.initial_balance,
        transaction_cost_rate=cfg.transaction_cost_rate,
        pd_bonus=cfg.pd_bonus,
        reward_scale=cfg.reward_scale,
    )

    state = test_env.reset()
    done = False
    test_total_reward = 0.0
    equity_curve = [test_env.net_worth]
    wins = 0
    trades = 0

    while not done:
        # Use greedy=True for evaluation to disable random exploration
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, _ = test_env.step(action)
        
        # Track win rate roughly by profitable actions
        if action in [1, 2]:
            trades += 1
            if reward > 0:
                wins += 1

        state = next_state
        test_total_reward += reward
        equity_curve.append(test_env.net_worth)

    from utils.metrics import compute_performance
    perf = compute_performance(equity_curve)
    win_rate = (wins / trades) if trades > 0 else 0.0

    return {
        "status": "success",
        "logs": training_logs,
        "model_path": str(cfg.model_path),
        "total_reward": test_total_reward,
        "final_equity": test_env.net_worth,
        "win_rate": win_rate,
        "sharpe_ratio": perf["sharpe_ratio"]
    }


def main() -> None:
    cfg = Config()
    df = load_and_prepare_data(str(cfg.data_path), rolling_window=cfg.rolling_window)
    run_training_from_df(df, cfg)

if __name__ == "__main__":
    main()
