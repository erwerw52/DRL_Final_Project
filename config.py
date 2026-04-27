from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    project_dir: Path = Path(__file__).resolve().parent
    data_path: Path = project_dir / "data" / "sample_data.csv"
    outputs_dir: Path = project_dir / "outputs"
    model_path: Path = outputs_dir / "dqn_model.pt"

    # Data
    rolling_window: int = 100
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Environment
    initial_balance: float = 100000.0
    transaction_cost_rate: float = 0.001
    pd_bonus: float = 0.05
    reward_scale: float = 1.0

    # Agent / training
    state_dim: int = 12
    action_dim: int = 3
    episodes: int = 30
    batch_size: int = 64
    replay_capacity: int = 100000
    gamma: float = 0.99
    lr: float = 0.001
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    target_update_freq: int = 1

    # Reproducibility
    seed: int = 42
