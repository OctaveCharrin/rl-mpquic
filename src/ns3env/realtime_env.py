"""
Hierarchical real-time video environment.

Wraps a :class:`~src.ns3env.dataplane.DataPlane` and exposes the two timescales
the dual-agent controller needs:

* **Transport agent (every frame):** observes per-path transport state + the
  App agent's current target bitrate; its action (a per-path split) is applied to
  the frame, and it is rewarded per frame for delivering that frame quickly and
  intact (:func:`compute_transport_reward`).
* **App agent (every ``app_period_s``):** observes the aggregate app state; its
  action (a target bitrate) persists across frames, and it is rewarded over the
  *window* of frames it governed by the VMAF-based QoE
  (:func:`compute_qoe_reward`).

The env is a pure observation/reward builder + window accumulator; the training
loop (:mod:`src.train.hierarchical_train`) owns the agents and orchestrates the
two cadences. Observations are fixed-length numpy vectors (the path count is
fixed per run), normalized to roughly ``[0, 1]`` so the SAC networks see
well-scaled inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataplane import DataPlane, FrameObs, FrameResult
from .qoe import (
    QoEWeights,
    VmafFn,
    compute_qoe_reward,
    compute_transport_reward,
    qoe_components,
)
from .video_source import VideoSourceConfig

# Per-path transport feature count in the (flat) transport observation.
_PATH_FEATURES = 5
# Global feature count prepended to the transport observation.
_TRANSPORT_GLOBAL = 4
# Per-path feature count in the *structured* (scoring) transport state: the five
# flat per-path features plus the liveness flag.
_PATH_FEATURES_SCORING = _PATH_FEATURES + 1
# App observation feature count. The policy is horizon-agnostic: it sees only
# sender-observable state, never how far through the episode it is (a real call
# has unknown duration), and the fixed episode length is handled as a value
# bootstrap (truncation), not a terminal, in the training loop.
_APP_FEATURES = 5

# Normalization references (bytes) for cwnd / send-buffer occupancy.
_CWND_NORM = 200_000.0
_BUFFER_NORM = 200_000.0


@dataclass
class StepInfo:
    """Per-frame diagnostics for logging."""

    latency_ms: float
    jitter_ms: float
    loss: float
    bytes_delivered: int
    transport_reward: float


@dataclass
class TransportState:
    """Structured (set-shaped) transport observation for the scoring agent.

    Unlike the flat ``build_transport_obs`` vector, this keeps the per-path rows
    *separate* so a permutation-equivariant policy can score a variable / changing
    set of paths. ``glob`` is the same aggregate context the flat observation
    prepends; ``paths`` is one row per candidate path; ``mask`` is the liveness
    flag (1.0 = usable). Inactive rows are still present (fixed ``num_paths``
    width) but masked out by the model.
    """

    glob: np.ndarray   # (G,)
    paths: np.ndarray  # (N, F)
    mask: np.ndarray   # (N,)  1.0 active / 0.0 churned-out


class HierarchicalRealtimeEnv:
    """Observation/reward layer over a data-plane backend."""

    def __init__(
        self,
        dataplane: DataPlane,
        *,
        video: Optional[VideoSourceConfig] = None,
        weights: Optional[QoEWeights] = None,
        episode_seconds: float = 30.0,
        vmaf_fn: Optional[VmafFn] = None,
    ):
        self.dp = dataplane
        self.video = video or VideoSourceConfig()
        self.weights = weights or QoEWeights()
        # Optional learned VMAF scorer; None => default bitrate-only log curve.
        self.vmaf_fn = vmaf_fn
        # Retained for config compatibility/logging only; no longer fed to the
        # policy (the App observation is horizon-agnostic). The episode horizon is
        # enforced by the data plane and handled as a truncation in training.
        self.episode_seconds = float(episode_seconds)

        self._obs: Optional[FrameObs] = None
        # App-reward window accumulators (frames the current bitrate governed).
        self._win_latency: List[float] = []
        self._win_jitter: List[float] = []
        self._win_loss: List[float] = []
        self._win_bitrate: float = self.video.init_bitrate_kbps

    # -- dimensions --------------------------------------------------------- #

    @property
    def num_paths(self) -> int:
        return self.dp.num_paths

    @property
    def app_obs_dim(self) -> int:
        return _APP_FEATURES

    @property
    def transport_obs_dim(self) -> int:
        return _TRANSPORT_GLOBAL + _PATH_FEATURES * self.num_paths

    @property
    def transport_global_dim(self) -> int:
        """Global-context width for the scoring (structured) transport state."""
        return _TRANSPORT_GLOBAL

    @property
    def transport_path_dim(self) -> int:
        """Per-path feature width for the scoring (structured) transport state."""
        return _PATH_FEATURES_SCORING

    @property
    def transport_act_dim(self) -> int:
        return self.num_paths

    # -- lifecycle ---------------------------------------------------------- #

    def reset(self, *, seed: Optional[int] = None) -> FrameObs:
        self._obs = self.dp.reset(seed=seed)
        self._win_latency.clear()
        self._win_jitter.clear()
        self._win_loss.clear()
        self._win_bitrate = self._obs.current_bitrate_kbps
        return self._obs

    @property
    def obs(self) -> FrameObs:
        if self._obs is None:
            raise RuntimeError("call reset() before using the env")
        return self._obs

    def is_done(self) -> bool:
        return self.dp.is_done()

    # -- observation builders ---------------------------------------------- #

    def build_app_obs(self, obs: Optional[FrameObs] = None) -> np.ndarray:
        o = obs or self.obs
        # No episode-progress feature on purpose: the policy must be deployable on
        # a call of unknown/unbounded duration, so it sees only sender-observable
        # state (bitrate it chose + measured RTT/jitter/loss/throughput).
        return np.array(
            [
                self._bitrate_norm(o.current_bitrate_kbps),
                _clip(o.rtt_ms / self.weights.latency_norm_ms),
                _clip(o.jitter_ms / self.weights.jitter_norm_ms),
                _clip(o.loss),
                _clip(o.throughput_mbps / self.dp.cap_mbps),
            ],
            dtype=np.float32,
        )

    def build_transport_obs(
        self, obs: Optional[FrameObs] = None, target_bitrate_kbps: Optional[float] = None
    ) -> np.ndarray:
        o = obs or self.obs
        br = o.current_bitrate_kbps if target_bitrate_kbps is None else target_bitrate_kbps
        feats: List[float] = [
            self._bitrate_norm(br),  # App agent's target -> hierarchy coupling
            _clip(o.rtt_ms / self.weights.latency_norm_ms),
            _clip(o.loss),
            _clip(o.throughput_mbps / self.dp.cap_mbps),
        ]
        n = self.num_paths
        for i in range(n):
            feats.extend(
                [
                    _clip(_at(o.cwnd, i) / _CWND_NORM),
                    _clip(_at(o.srtt_ms, i) / self.weights.latency_norm_ms),
                    _clip(_at(o.buffer_occ, i) / _BUFFER_NORM),
                    _clip(_at(o.path_throughput_mbps, i) / self.dp.cap_mbps),
                    _clip(_at(o.path_loss, i)),
                ]
            )
        return np.array(feats, dtype=np.float32)

    def build_transport_state(
        self, obs: Optional[FrameObs] = None, target_bitrate_kbps: Optional[float] = None
    ) -> TransportState:
        """Structured transport observation for the scoring (dynamic-input) agent.

        Same normalized features as :meth:`build_transport_obs`, but kept as a
        ``(glob, paths, mask)`` triple so a permutation-equivariant policy can
        handle a variable / changing path set. The per-path row carries the five
        flat features plus the liveness flag; ``mask`` mirrors that flag (1.0 when
        the path is usable). When the backend reports no ``path_active`` (static
        network), every path is treated as live.
        """
        o = obs or self.obs
        br = o.current_bitrate_kbps if target_bitrate_kbps is None else target_bitrate_kbps
        glob = np.array(
            [
                self._bitrate_norm(br),
                _clip(o.rtt_ms / self.weights.latency_norm_ms),
                _clip(o.loss),
                _clip(o.throughput_mbps / self.dp.cap_mbps),
            ],
            dtype=np.float32,
        )
        n = self.num_paths
        paths = np.zeros((n, _PATH_FEATURES_SCORING), dtype=np.float32)
        mask = np.ones(n, dtype=np.float32)
        for i in range(n):
            active = _at(o.path_active, i) if o.path_active else 1.0
            mask[i] = 1.0 if active >= 0.5 else 0.0
            paths[i] = [
                _clip(_at(o.cwnd, i) / _CWND_NORM),
                _clip(_at(o.srtt_ms, i) / self.weights.latency_norm_ms),
                _clip(_at(o.buffer_occ, i) / _BUFFER_NORM),
                _clip(_at(o.path_throughput_mbps, i) / self.dp.cap_mbps),
                _clip(_at(o.path_loss, i)),
                mask[i],
            ]
        return TransportState(glob=glob, paths=paths, mask=mask)

    # -- stepping ----------------------------------------------------------- #

    def step(
        self, target_bitrate_kbps: float, split_ratio: Sequence[float]
    ) -> Tuple[FrameObs, float, bool, StepInfo]:
        """Apply one frame; return (next_obs, transport_reward, done, info)."""
        result: FrameResult = self.dp.step_frame(target_bitrate_kbps, split_ratio)
        t_reward = compute_transport_reward(
            latency_ms=result.latency_ms,
            jitter_ms=result.jitter_ms,
            loss=result.loss,
            weights=self.weights,
        )
        # Accumulate into the App-reward window.
        self._win_latency.append(result.latency_ms)
        self._win_jitter.append(result.jitter_ms)
        self._win_loss.append(result.loss)
        self._win_bitrate = target_bitrate_kbps

        self._obs = self.dp.current_obs()
        info = StepInfo(
            latency_ms=result.latency_ms,
            jitter_ms=result.jitter_ms,
            loss=result.loss,
            bytes_delivered=result.bytes_delivered,
            transport_reward=t_reward,
        )
        return self._obs, t_reward, self.dp.is_done(), info

    # -- App-reward window -------------------------------------------------- #

    def pop_app_window_reward(self) -> Tuple[float, Dict[str, float]]:
        """QoE over the accumulated window; clears the window.

        Aggregates the frames the just-ended bitrate governed (mean latency,
        jitter and loss) and scores them with the VMAF-based QoE. Returns the
        reward and its unweighted components for logging.
        """
        if not self._win_latency:
            comps = qoe_components(
                bitrate_kbps=self._win_bitrate,
                latency_ms=0.0,
                jitter_ms=0.0,
                loss=0.0,
                vmaf_fn=self.vmaf_fn,
            )
            return 0.0, comps
        lat = float(np.mean(self._win_latency))
        jit = float(np.mean(self._win_jitter))
        los = float(np.mean(self._win_loss))
        reward = compute_qoe_reward(
            bitrate_kbps=self._win_bitrate,
            latency_ms=lat,
            jitter_ms=jit,
            loss=los,
            weights=self.weights,
            vmaf_fn=self.vmaf_fn,
        )
        comps = qoe_components(
            bitrate_kbps=self._win_bitrate,
            latency_ms=lat,
            jitter_ms=jit,
            loss=los,
            vmaf_fn=self.vmaf_fn,
        )
        self._win_latency.clear()
        self._win_jitter.clear()
        self._win_loss.clear()
        return reward, comps

    # -- helpers ------------------------------------------------------------ #

    def _bitrate_norm(self, kbps: float) -> float:
        lo, hi = self.video.min_bitrate_kbps, self.video.max_bitrate_kbps
        return _clip((kbps - lo) / max(hi - lo, 1e-6))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, x)))


def _at(seq: Sequence[float], i: int) -> float:
    return float(seq[i]) if i < len(seq) else 0.0
