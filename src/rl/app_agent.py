"""
App agent: picks the WebRTC encoder target bitrate every ``app_period_s``.

A thin wrapper over :class:`~src.rl.sac_agent.SACAgent` with a 1-D action that is
mapped from the normalized ``[-1, 1]`` space to ``[min_kbps, max_kbps]``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .sac_agent import SACAgent, SACConfig


class AppAgent:
    """SAC controller for the encoder target bitrate (kbps)."""

    def __init__(
        self,
        obs_dim: int,
        *,
        min_kbps: float,
        max_kbps: float,
        config: Optional[SACConfig] = None,
    ):
        self.min_kbps = float(min_kbps)
        self.max_kbps = float(max_kbps)
        self.sac = SACAgent(obs_dim, act_dim=1, config=config)

    def to_kbps(self, raw_action: np.ndarray) -> float:
        """Map a normalized action in ``[-1, 1]`` to a bitrate in the range."""
        u = (float(raw_action[0]) + 1.0) / 2.0  # [0, 1]
        return self.min_kbps + u * (self.max_kbps - self.min_kbps)

    def select(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[float, np.ndarray]:
        raw = self.sac.select_action(obs, deterministic=deterministic)
        return self.to_kbps(raw), raw

    def store(self, obs, raw_action, reward, next_obs, done) -> None:
        self.sac.store(obs, raw_action, reward, next_obs, done)

    def update(self) -> Optional[Dict[str, float]]:
        return self.sac.update()
