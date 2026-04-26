from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.network import QNetwork


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, replay_buffer, batch_size: int = 64) -> float:
        if len(replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
