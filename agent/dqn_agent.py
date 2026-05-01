"""DQN Agent with integrated replay buffer, step-based target update, and gradient clipping."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.network import QNetwork
from utils.replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Use cfg if provided, otherwise use defaults
        if cfg is not None:
            self.gamma = cfg.gamma
            self.lr = cfg.lr
            self.epsilon = cfg.epsilon_start
            self.epsilon_decay = cfg.epsilon_decay
            self.epsilon_min = cfg.epsilon_min
            self.min_replay_size = cfg.min_replay_size
            self.batch_size = cfg.batch_size
            self.target_update_freq = cfg.target_update_freq
            replay_size = cfg.replay_size
        else:
            self.gamma = 0.95
            self.lr = 1e-4
            self.epsilon = 1.0
            self.epsilon_decay = 0.965
            self.epsilon_min = 0.03
            self.min_replay_size = 1200
            self.batch_size = 64
            self.target_update_freq = 300
            replay_size = 100000

        self.policy_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(replay_size)
        self.steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        return q_values

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str, **extra) -> None:
        checkpoint = {
            "model_state_dict": self.policy_net.state_dict(),
            "epsilon": self.epsilon,
            **extra,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net.load_state_dict(checkpoint["model_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            return checkpoint
        else:
            # Legacy format: raw state_dict
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)
            return {}
