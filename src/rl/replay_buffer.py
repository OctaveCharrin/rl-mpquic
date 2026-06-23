"""Fixed-capacity ring replay buffer for off-policy SAC."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Pre-allocated circular buffer of (obs, action, reward, next_obs, done).

    Actions are stored in the agent's normalized action space (``[-1, 1]``).
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = 1.0 if done else 0.0
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
        )
