"""
Transport agent: picks the per-path traffic split every frame.

Two interchangeable backends, selected by ``arch``:

* ``"flat"`` (legacy, default) — a fixed-dimension :class:`SACAgent` over the flat
  transport observation; its ``num_paths``-D normalized action is softmaxed into a
  split. Locked to a fixed path count.
* ``"scoring"`` — a permutation-equivariant :class:`ScoringSACAgent` over the
  structured ``(glob, paths, mask)`` state, which handles a variable / changing
  set of paths. Its per-path latent is softmaxed over the *active* paths only.

Either way the observation includes the App agent's current target bitrate, which
is what makes the two agents *hierarchical* rather than independent.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .sac_agent import SACAgent, SACConfig
from .scoring_sac_agent import ScoringSACAgent

_NEG_FILL = -1e30


class TransportAgent:
    """SAC controller for the per-path traffic-split ratio (flat or scoring)."""

    def __init__(
        self,
        obs_dim: int,
        num_paths: int,
        *,
        config: Optional[SACConfig] = None,
        temperature: float = 1.0,
        arch: str = "flat",
        global_dim: Optional[int] = None,
        path_dim: Optional[int] = None,
    ):
        self.num_paths = int(num_paths)
        self.temperature = float(temperature)
        self.arch = str(arch)
        if self.arch == "scoring":
            if global_dim is None or path_dim is None:
                raise ValueError("scoring arch requires global_dim and path_dim")
            self.sac = ScoringSACAgent(
                int(global_dim), int(path_dim), self.num_paths, config=config
            )
        elif self.arch == "flat":
            self.sac = SACAgent(obs_dim, act_dim=self.num_paths, config=config)
        else:
            raise ValueError(f"unknown transport arch {arch!r} (use 'flat' or 'scoring')")

    # -- split mapping ------------------------------------------------------ #

    def to_split(self, raw_action: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Softmax a normalized action into a split that sums to 1.

        When ``mask`` is given (scoring arch), inactive paths are excluded so the
        split is supported only on the live paths.
        """
        logits = np.asarray(raw_action, dtype=np.float64) / max(self.temperature, 1e-6)
        if mask is not None:
            m = np.asarray(mask, dtype=np.float64) >= 0.5
            if not m.any():  # degenerate: nothing live -> uniform
                return np.full(len(logits), 1.0 / len(logits), dtype=np.float32)
            logits = np.where(m, logits, _NEG_FILL)
        logits -= logits.max()  # numerical stability
        exp = np.exp(logits)
        if mask is not None:
            exp = np.where(np.asarray(mask, dtype=np.float64) >= 0.5, exp, 0.0)
        return (exp / exp.sum()).astype(np.float32)

    # -- interaction (dispatches on arch) ----------------------------------- #

    def select(self, obs, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Return (split, raw_action). ``obs`` is a flat vector (flat arch) or a
        ``TransportState`` (scoring arch)."""
        if self.arch == "scoring":
            raw = self.sac.select_action(
                obs.glob, obs.paths, obs.mask, deterministic=deterministic
            )
            return self.to_split(raw, obs.mask), raw
        raw = self.sac.select_action(obs, deterministic=deterministic)
        return self.to_split(raw), raw

    def store(self, obs, raw_action, reward, next_obs, done) -> None:
        self.sac.store(obs, raw_action, reward, next_obs, done)

    def update(self) -> Optional[Dict[str, float]]:
        return self.sac.update()
