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

    # Persistence (P3): survive --resume so long runs keep their off-policy data.
    _ARRAYS = ("obs", "next_obs", "act", "rew", "done")

    def _extra_arrays(self) -> Dict[str, np.ndarray]:
        """Subclass hook: extra named arrays/scalars to persist (e.g. PER priorities)."""
        return {}

    def _load_extra(self, data) -> None:
        """Subclass hook: restore whatever ``_extra_arrays`` saved."""

    def save(self, path: str, stores: int = 0, updates: int = 0) -> None:
        arrays = {name: getattr(self, name) for name in self._ARRAYS}
        arrays.update(self._extra_arrays())
        np.savez(
            path,
            _ptr=np.int64(self._ptr),
            _size=np.int64(self._size),
            _stores=np.int64(stores),
            _updates=np.int64(updates),
            capacity=np.int64(self.capacity),
            **arrays,
        )

    def load(self, path: str) -> Tuple[int, int]:
        """Restore buffer state in place; return ``(stores, updates)`` counters."""
        with np.load(path) as data:
            if int(data["capacity"]) != self.capacity:
                raise ValueError(
                    f"buffer capacity mismatch: file {int(data['capacity'])} "
                    f"vs current {self.capacity}"
                )
            for name in self._ARRAYS:
                cur = getattr(self, name)
                arr = data[name]
                if arr.shape != cur.shape:
                    raise ValueError(
                        f"buffer array '{name}' shape mismatch: file {arr.shape} "
                        f"vs current {cur.shape}"
                    )
                cur[...] = arr
            self._ptr = int(data["_ptr"])
            self._size = int(data["_size"])
            self._load_extra(data)
            # _updates absent in buffers saved before faithful-PER-resume support.
            updates = int(data["_updates"]) if "_updates" in data else 0
            return int(data["_stores"]), updates


class StructuredReplayBuffer:
    """Ring buffer of set-shaped transitions for the scoring path agent.

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

    # Persistence (P3): survive --resume so long runs keep their off-policy data.
    _ARRAYS = (
        "glob",
        "paths",
        "mask",
        "act",
        "rew",
        "next_glob",
        "next_paths",
        "next_mask",
        "done",
    )

    def _extra_arrays(self) -> Dict[str, np.ndarray]:
        """Subclass hook: extra named arrays/scalars to persist (e.g. PER priorities)."""
        return {}

    def _load_extra(self, data) -> None:
        """Subclass hook: restore whatever ``_extra_arrays`` saved."""

    def save(self, path: str, stores: int = 0, updates: int = 0) -> None:
        arrays = {name: getattr(self, name) for name in self._ARRAYS}
        arrays.update(self._extra_arrays())
        np.savez(
            path,
            _ptr=np.int64(self._ptr),
            _size=np.int64(self._size),
            _stores=np.int64(stores),
            _updates=np.int64(updates),
            capacity=np.int64(self.capacity),
            **arrays,
        )

    def load(self, path: str) -> Tuple[int, int]:
        """Restore buffer state in place; return ``(stores, updates)`` counters."""
        with np.load(path) as data:
            if int(data["capacity"]) != self.capacity:
                raise ValueError(
                    f"buffer capacity mismatch: file {int(data['capacity'])} "
                    f"vs current {self.capacity}"
                )
            for name in self._ARRAYS:
                cur = getattr(self, name)
                arr = data[name]
                if arr.shape != cur.shape:
                    raise ValueError(
                        f"buffer array '{name}' shape mismatch: file {arr.shape} "
                        f"vs current {cur.shape}"
                    )
                cur[...] = arr
            self._ptr = int(data["_ptr"])
            self._size = int(data["_size"])
            self._load_extra(data)
            # _updates absent in buffers saved before faithful-PER-resume support.
            updates = int(data["_updates"]) if "_updates" in data else 0
            return int(data["_stores"]), updates


class _SumTree:
    """Array-backed sum tree for O(log n) proportional sampling.

    Leaf ``l`` (0..capacity-1) lives at tree index ``l + capacity - 1``; each
    internal node holds the sum of its subtree, so ``tree[0]`` is the total.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def update(self, leaf: int, value: float) -> None:
        idx = leaf + self.capacity - 1
        delta = value - self.tree[idx]
        self.tree[idx] = value
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get(self, s: float) -> Tuple[int, float]:
        """Return ``(leaf, priority)`` for cumulative sum ``s`` in ``[0, total)``."""
        idx = 0
        while idx < self.capacity - 1:  # descend until a leaf
            left = 2 * idx + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx - (self.capacity - 1), float(self.tree[idx])


class _Prioritized:
    """Mixin adding proportional prioritized sampling to a ring buffer.

    Priorities are stored as ``(|td| + eps) ** alpha``. A freshly pushed leaf
    gets the running max priority so it is (almost) certainly replayed once
    before its TD error is known. Importance-sampling weights correct the bias.
    """

    _PER_EPS = 1e-6

    def _per_init(self, alpha: float) -> None:
        self._alpha = float(alpha)
        self._tree = _SumTree(self.capacity)
        self._max_p = 1.0  # max priority^alpha seen (new leaves inherit this)

    def _mark_new(self, leaf: int) -> None:
        self._tree.update(leaf, self._max_p)

    def _sample_indices(self, batch_size: int, beta: float):
        total = self._tree.total
        seg = total / batch_size
        idx = np.empty(batch_size, dtype=np.int64)
        probs = np.empty(batch_size, dtype=np.float64)
        for j in range(batch_size):
            s = np.random.uniform(seg * j, seg * (j + 1))
            leaf, p = self._tree.get(s)
            # A zero-priority (unfilled) slot can only be hit by float slop at the
            # segment edge; clamp into the live region.
            idx[j] = min(leaf, self._size - 1)
            probs[j] = p
        probs /= max(total, self._PER_EPS)
        weights = (self._size * np.maximum(probs, self._PER_EPS)) ** (-beta)
        weights /= weights.max()  # normalize to (0, 1]
        return idx, weights.astype(np.float32).reshape(-1, 1)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        p = (
            np.abs(np.asarray(td_errors, dtype=np.float64).reshape(-1)) + self._PER_EPS
        ) ** self._alpha
        for leaf, pv in zip(np.asarray(indices).reshape(-1), p):
            self._tree.update(int(leaf), float(pv))
        if p.size:
            self._max_p = max(self._max_p, float(p.max()))

    # Persistence: store the whole tree (leaf + internal) so no rebuild needed.
    def _extra_arrays(self) -> Dict[str, np.ndarray]:
        return {"_tree": self._tree.tree, "_max_p": np.float64(self._max_p)}

    def _load_extra(self, data) -> None:
        tree = data["_tree"]
        if tree.shape != self._tree.tree.shape:
            raise ValueError(
                f"PER sum-tree shape mismatch: file {tree.shape} "
                f"vs current {self._tree.tree.shape}"
            )
        self._tree.tree[...] = tree
        self._max_p = float(data["_max_p"])


class PrioritizedReplayBuffer(_Prioritized, ReplayBuffer):
    """Flat replay buffer with proportional PER (Schaul et al., 2016)."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, alpha: float = 0.6):
        super().__init__(obs_dim, act_dim, capacity)
        self._per_init(alpha)

    def push(self, obs, act, rew, next_obs, done) -> None:
        i = self._ptr
        super().push(obs, act, rew, next_obs, done)
        self._mark_new(i)

    def sample(self, batch_size: int, beta: float = 0.4):
        idx, weights = self._sample_indices(batch_size, beta)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
            idx,
            weights,
        )


class PrioritizedStructuredReplayBuffer(_Prioritized, StructuredReplayBuffer):
    """Structured (set-shaped) replay buffer with proportional PER."""

    def __init__(
        self, global_dim: int, path_dim: int, num_paths: int, capacity: int, alpha: float = 0.6
    ):
        super().__init__(global_dim, path_dim, num_paths, capacity)
        self._per_init(alpha)

    def push(self, glob, paths, mask, act, rew, next_glob, next_paths, next_mask, done) -> None:
        i = self._ptr
        super().push(glob, paths, mask, act, rew, next_glob, next_paths, next_mask, done)
        self._mark_new(i)

    def sample(self, batch_size: int, beta: float = 0.4) -> Dict[str, np.ndarray]:
        idx, weights = self._sample_indices(batch_size, beta)
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
            "indices": idx,
            "weights": weights,
        }
