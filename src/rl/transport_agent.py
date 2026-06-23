"""
Transport agent: picks the per-path traffic split every frame.

A thin wrapper over :class:`~src.rl.sac_agent.SACAgent` whose ``num_paths``-D
normalized action is turned into a split (non-negative, sums to 1) via a softmax.
Its observation includes the App agent's current target bitrate, which is what
makes the two agents *hierarchical* rather than independent.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .sac_agent import SACAgent, SACConfig


class TransportAgent:
    """SAC controller for the per-path traffic-split ratio."""

    def __init__(
        self,
        obs_dim: int,
        num_paths: int,
        *,
        config: Optional[SACConfig] = None,
        temperature: float = 1.0,
    ):
        self.num_paths = int(num_paths)
        self.temperature = float(temperature)
        self.sac = SACAgent(obs_dim, act_dim=self.num_paths, config=config)

    def to_split(self, raw_action: np.ndarray) -> np.ndarray:
        """Softmax a normalized action into a split that sums to 1."""
        logits = np.asarray(raw_action, dtype=np.float64) / max(self.temperature, 1e-6)
        logits -= logits.max()  # numerical stability
        exp = np.exp(logits)
        return (exp / exp.sum()).astype(np.float32)

    def select(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        raw = self.sac.select_action(obs, deterministic=deterministic)
        return self.to_split(raw), raw

    def store(self, obs, raw_action, reward, next_obs, done) -> None:
        self.sac.store(obs, raw_action, reward, next_obs, done)

    def update(self) -> Optional[Dict[str, float]]:
        return self.sac.update()
