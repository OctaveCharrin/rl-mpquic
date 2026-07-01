"""Fixed-capacity ring replay buffers for off-policy SAC."""

from __future__ import annotations

from typing import Dict, Tuple

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


class StructuredReplayBuffer:
    """Ring buffer of set-shaped transitions for the scoring transport agent.

    Each transition is a ``(global, paths, mask)`` state, a per-path action
    latent, a reward, the next state, and a done flag. The path count is fixed at
    ``num_paths`` (the candidate cap), so all arrays are rectangular — the mask,
    not ragged padding, carries which rows are live. Actions are stored in the
    policy's raw (pre-softmax) per-path latent space.
    """

    def __init__(self, global_dim: int, path_dim: int, num_paths: int, capacity: int):
        self.capacity = int(capacity)
        n = int(num_paths)
        self.glob = np.zeros((self.capacity, global_dim), dtype=np.float32)
        self.paths = np.zeros((self.capacity, n, path_dim), dtype=np.float32)
        self.mask = np.zeros((self.capacity, n), dtype=np.float32)
        self.act = np.zeros((self.capacity, n), dtype=np.float32)
        self.rew = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_glob = np.zeros((self.capacity, global_dim), dtype=np.float32)
        self.next_paths = np.zeros((self.capacity, n, path_dim), dtype=np.float32)
        self.next_mask = np.zeros((self.capacity, n), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        glob: np.ndarray,
        paths: np.ndarray,
        mask: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_glob: np.ndarray,
        next_paths: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        i = self._ptr
        self.glob[i] = glob
        self.paths[i] = paths
        self.mask[i] = mask
        self.act[i] = act
        self.rew[i] = rew
        self.next_glob[i] = next_glob
        self.next_paths[i] = next_paths
        self.next_mask[i] = next_mask
        self.done[i] = 1.0 if done else 0.0
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "glob": self.glob[idx],
            "paths": self.paths[idx],
            "mask": self.mask[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_glob": self.next_glob[idx],
            "next_paths": self.next_paths[idx],
            "next_mask": self.next_mask[idx],
            "done": self.done[idx],
        }
