"""
Data-plane abstraction for the real-time multipath-QUIC video environment.

A :class:`DataPlane` represents *the network* as seen by the hierarchical
controller. The decision epoch is **one video frame**: the agents pick a target
bitrate (App) and a per-path traffic split (Transport), the data plane delivers
that frame, advances its clock by ``1/fps``, and reports the realized result plus
the (now-current) per-path transport state.

Two backends implement the same interface:

* :class:`MockRealtimeDataPlane` -- pure Python, trace-driven. Each path has a
  time-varying capacity (sinusoid + bursty cross-traffic + noise) and a standing
  queue, so over-driving a path makes its latency grow (bufferbloat). Deterministic
  given a seed; runs anywhere with no NS-3 dependency, so the env, reward, agents
  and training loop are fully testable without compiling NS-3.
* :class:`Ns3DataPlane` -- drives the real NS-3 packet-level scenario
  (``ns3/realtime_mpquic.cc``) over the ns3-ai shared-memory bridge (Linux/WSL2).
  One long-lived NS-3 process serves the whole run; episode boundaries are sent
  in-band. The method contract is identical to the mock.

Both produce a :class:`FrameObs` (a snapshot of the C++ ``EnvStruct``) and a
:class:`FrameResult` (the realized frame outcome). The bridge protocol mirrors the
upstream ns3-ai examples: C++ leads with a send, so ``reset`` receives the initial
observation and each ``step_frame`` sends the action then receives the next
observation, whose ``last_*`` fields carry the just-delivered frame's result.
"""

from __future__ import annotations

import gc
import glob
import math
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .video_source import VideoSourceConfig, frame_bytes


# --------------------------------------------------------------------------- #
# Shared snapshot / result types (mirror EnvStruct in ns3/realtime_mpquic.h)
# --------------------------------------------------------------------------- #


@dataclass
class FrameObs:
    """Observable state at one frame decision point (a copy of ``EnvStruct``)."""

    num_paths: int
    clock_s: float
    done: bool
    app_decision_due: bool

    # App-level aggregate state.
    current_bitrate_kbps: float
    rtt_ms: float
    jitter_ms: float
    loss: float
    throughput_mbps: float

    # Per-path transport state (length == num_paths).
    cwnd: List[float] = field(default_factory=list)
    srtt_ms: List[float] = field(default_factory=list)
    buffer_occ: List[float] = field(default_factory=list)
    path_throughput_mbps: List[float] = field(default_factory=list)
    path_loss: List[float] = field(default_factory=list)

    # Per-path liveness mask (length == num_paths): 1.0 if the path is currently
    # usable, 0.0 if it has churned out. With static dynamics every path is always
    # 1.0. `num_paths` is the *candidate* count (the upper bound); the number of
    # active paths can vary per frame once churn is enabled.
    #
    # CONTRACT: both backends emit this mask — the mock from its churn machine,
    # the NS-3 body from `EnvStruct.pathActive[]` (whose churn drops packets via
    # drop-all error models, so a masked-live path is actually usable). Parity is
    # guarded by scripts/parity_check.py.
    path_active: List[float] = field(default_factory=list)

    # Realized result of the most-recently-completed frame.
    last_latency_ms: float = 0.0
    last_jitter_ms: float = 0.0
    last_loss: float = 0.0
    last_bytes: int = 0


@dataclass
class FrameResult:
    """Realized outcome of delivering one frame."""

    latency_ms: float
    jitter_ms: float
    loss: float
    bytes_delivered: int


class DataPlane(ABC):
    """Backend-agnostic network model. Decision epoch = one video frame."""

    #: Number of candidate paths (subflows).
    num_paths: int

    #: Normalization cap (Mbps): an upper bound on achievable aggregate
    #: throughput, used to normalize observations.
    cap_mbps: float

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None) -> FrameObs:
        """Start a new episode and return the initial observation."""

    @abstractmethod
    def current_obs(self) -> FrameObs:
        """The latest observation snapshot."""

    @abstractmethod
    def step_frame(
        self, target_bitrate_kbps: float, split_ratio: Sequence[float]
    ) -> FrameResult:
        """Deliver one frame at ``target_bitrate_kbps`` split per ``split_ratio``.

        Advances the clock by ``1/fps`` and returns the realized result. The
        next observation is available via :meth:`current_obs`.
        """

    @abstractmethod
    def is_done(self) -> bool:
        """True once the episode horizon is exhausted."""

    @property
    @abstractmethod
    def clock_s(self) -> float:
        """Current sim time in seconds."""

    def app_decision_due(self) -> bool:
        """True when the App agent should pick a new bitrate this frame."""
        return self.current_obs().app_decision_due

    def close(self) -> None:  # pragma: no cover - overridden by NS-3 backend
        """Release any resources (no-op for pure-Python backends)."""


# --------------------------------------------------------------------------- #
# Mock trace-driven backend
# --------------------------------------------------------------------------- #


@dataclass
class _PathTrace:
    """Time-varying capacity + standing-queue model for one path."""

    base_mbps: float
    base_rtt_ms: float
    amp: float
    period_s: float
    phase: float
    cross_frac: float
    cross_period_s: float
    noise_std: float

    def capacity_mbps(self, t: float, rng: np.random.Generator) -> float:
        season = 1.0 + self.amp * math.sin(2.0 * math.pi * t / self.period_s + self.phase)
        cross = self.cross_frac * (
            0.5 + 0.5 * math.sin(2.0 * math.pi * t / self.cross_period_s + self.phase)
        )
        noise = 1.0 + rng.normal(0.0, self.noise_std)
        return max(0.05, self.base_mbps * season * max(0.05, 1.0 - cross) * noise)

    def network_loss(self, cap_mbps: float) -> float:
        # Light network loss when a path is congested below ~35% of baseline.
        ratio = cap_mbps / max(self.base_mbps, 0.05)
        if ratio >= 0.5:
            return 0.0
        return float(min(0.1, 0.1 * (0.5 - ratio) / 0.5))


@dataclass
class DynamicsConfig:
    """Optional non-stationary network dynamics for :class:`MockRealtimeDataPlane`.

    All mechanisms are **off by default** so existing configs reproduce
    bit-for-bit (when ``enabled`` is False the dynamic RNG is never touched, so
    the base sinusoidal envelope and its draw sequence are unchanged). Turning
    them on makes *which path to send on* a genuinely time-varying decision:

    * **churn** — paths appear/disappear (an on/off Markov chain per path), so
      the active path count varies and bytes routed onto a dead path are lost.
    * **regime** — piecewise-constant capacity multiplier resampled at Poisson
      change-points, so the *best* path swaps abruptly (not a smooth drift).
    * **burst** — transient per-path capacity collapses (congestion spikes) that
      the scheduler must route around within a few frames.
    * **corr** — shared-bottleneck groups that degrade together, so naive
      diversification across the group does not help.
    """

    enabled: bool = False

    # Path churn (appear/disappear). Rates are per-second hazards.
    churn: bool = False
    churn_up_rate: float = 0.10      # down -> up
    churn_down_rate: float = 0.05    # up -> down
    min_active: int = 1              # never let the active set fall below this

    # Regime shifts (best-path swaps): per-path capacity multiplier resampled at
    # Poisson change-points, drawn ~ Uniform(regime_lo, regime_hi).
    regime: bool = False
    regime_rate: float = 0.20        # change-points per second per path
    regime_lo: float = 0.35
    regime_hi: float = 1.30

    # Congestion bursts: per-path transient capacity collapse.
    burst: bool = False
    burst_rate: float = 0.15         # bursts per second per path
    burst_intensity: float = 0.25    # capacity multiplier while bursting
    burst_duration_s: float = 0.5

    # Correlated failures: groups of path indices that degrade together.
    corr_groups: Sequence[Sequence[int]] = ()
    corr_rate: float = 0.05          # group events per second
    corr_intensity: float = 0.30     # capacity multiplier for all members
    corr_duration_s: float = 1.0

    # Optional per-episode domain randomization of the continuous parameters
    # above (None => fixed dynamics, legacy). Applied by the training loop, not by
    # the dataplane itself, so a sampled DynamicsConfig carries randomize=None.
    randomize: Optional["DynamicsRandomization"] = None


@dataclass
class DynamicsRandomization:
    """Per-episode domain randomization of :class:`DynamicsConfig` (Tier-2 #7).

    **Off by default.** When enabled, each configured ``(lo, hi)`` range is sampled
    uniformly per episode (seeded per episode for determinism); the base
    ``DynamicsConfig`` supplies the on/off flags, ``corr_groups``, and any parameter
    left without a range here. Randomizing the dynamics *distribution* trains a
    policy robust to a family of networks rather than one (Tobin et al., 2017).

    **Mock-only:** the NS-3 backend receives dynamics at C++ process start, so it
    keeps a single config; the training loop applies this only for ``backend=mock``.
    """

    enabled: bool = False
    churn_up_rate: Optional[Tuple[float, float]] = None
    churn_down_rate: Optional[Tuple[float, float]] = None
    regime_rate: Optional[Tuple[float, float]] = None
    regime_lo: Optional[Tuple[float, float]] = None
    regime_hi: Optional[Tuple[float, float]] = None
    burst_rate: Optional[Tuple[float, float]] = None
    burst_intensity: Optional[Tuple[float, float]] = None
    corr_rate: Optional[Tuple[float, float]] = None
    corr_intensity: Optional[Tuple[float, float]] = None

    def sample(self, base: "DynamicsConfig", rng: np.random.Generator) -> "DynamicsConfig":
        """Return a copy of ``base`` with each configured range sampled uniformly."""
        import dataclasses

        def pick(rng_range, cur):
            return float(rng.uniform(rng_range[0], rng_range[1])) if rng_range else cur

        lo = pick(self.regime_lo, base.regime_lo)
        hi = pick(self.regime_hi, base.regime_hi)
        if lo > hi:  # keep the regime multiplier band ordered
            lo, hi = hi, lo
        return dataclasses.replace(
            base,
            randomize=None,
            churn_up_rate=pick(self.churn_up_rate, base.churn_up_rate),
            churn_down_rate=pick(self.churn_down_rate, base.churn_down_rate),
            regime_rate=pick(self.regime_rate, base.regime_rate),
            regime_lo=lo,
            regime_hi=hi,
            burst_rate=pick(self.burst_rate, base.burst_rate),
            burst_intensity=pick(self.burst_intensity, base.burst_intensity),
            corr_rate=pick(self.corr_rate, base.corr_rate),
            corr_intensity=pick(self.corr_intensity, base.corr_intensity),
        )


@dataclass
class MockRealtimeConfig:
    """Configuration for :class:`MockRealtimeDataPlane`.

    Defaults describe a deliberately *asymmetric, time-varying* 3-path scenario
    (wired / Wi-Fi / LTE) matching ``configs/default.yaml`` and the C++ scenario,
    so a fixed split or single-path policy is suboptimal.
    """

    base_mbps: Sequence[float] = (8.0, 4.0, 2.0)
    base_rtt_ms: Sequence[float] = (20.0, 34.0, 60.0)
    amp: Sequence[float] = (0.45, 0.65, 0.35)
    period_s: Sequence[float] = (12.0, 7.0, 20.0)
    cross_frac: Sequence[float] = (0.45, 0.65, 0.35)
    cross_period_s: Sequence[float] = (5.0, 3.0, 8.0)
    noise_std: float = 0.05

    fps: float = 30.0
    episode_seconds: float = 30.0
    app_period_s: float = 1.0
    deadline_ms: float = 180.0
    video: VideoSourceConfig = field(default_factory=VideoSourceConfig)
    seed: int = 1

    # Optional non-stationary dynamics (None => fully static, legacy behavior).
    dynamics: Optional[DynamicsConfig] = None

    @property
    def num_paths(self) -> int:
        return len(self.base_mbps)


class MockRealtimeDataPlane(DataPlane):
    """Trace-driven, deterministic-per-seed network for fast RL iteration/tests."""

    def __init__(self, config: Optional[MockRealtimeConfig] = None):
        self.config = config or MockRealtimeConfig()
        n = self.config.num_paths
        self.num_paths = n
        self._traces: List[_PathTrace] = [
            _PathTrace(
                base_mbps=self.config.base_mbps[i],
                base_rtt_ms=self.config.base_rtt_ms[i],
                amp=self.config.amp[i],
                period_s=self.config.period_s[i],
                phase=2.0 * math.pi * i / max(1, n),
                cross_frac=self.config.cross_frac[i],
                cross_period_s=self.config.cross_period_s[i],
                noise_std=self.config.noise_std,
            )
            for i in range(n)
        ]
        # Aggregate cap = sum of per-path peaks (all paths used at once).
        self.cap_mbps = float(sum(t.base_mbps * (1.0 + t.amp) for t in self._traces))
        self._frames_per_app = max(1, round(self.config.fps * self.config.app_period_s))
        self._frames_per_episode = round(self.config.fps * self.config.episode_seconds)
        self._rng = np.random.default_rng(self.config.seed)
        self._reset_state(self.config.video.init_bitrate_kbps)

    # -- internal state ----------------------------------------------------- #

    def _reset_state(self, init_bitrate_kbps: float) -> None:
        n = self.num_paths
        self._t = 0.0
        self._frame_idx = 0  # frame_in_episode
        self._frame_total = 0
        self._busy_until = [0.0] * n  # path serialization clock (queue model)
        self._cur_bitrate = float(init_bitrate_kbps)
        self._cur_split = [1.0 / n] * n
        self._prev_latency_ms = -1.0
        self._jitter_ewma = 0.0
        self._loss_ewma = 0.0
        self._thr_ewma = 0.0
        self._rtt_ewma = 0.0
        self._path_thr_ewma = [t.base_mbps for t in self._traces]
        self._path_loss = [0.0] * n
        self._last = FrameResult(0.0, 0.0, 0.0, 0)
        self._init_dynamics()

    # -- non-stationary dynamics ------------------------------------------- #

    def _init_dynamics(self) -> None:
        """(Re)initialize the dynamic-state machines. No-op draw when disabled."""
        n = self.num_paths
        self._active = [True] * n
        self._regime_mult = [1.0] * n
        self._burst_until = [-1.0] * n
        self._corr_members: List[set] = []
        self._corr_until: List[float] = []
        d = self.config.dynamics
        if d is None or not d.enabled:
            return
        if d.regime:
            self._regime_mult = [
                float(self._rng.uniform(d.regime_lo, d.regime_hi)) for _ in range(n)
            ]
        self._corr_members = [
            {int(x) for x in grp if 0 <= int(x) < n} for grp in d.corr_groups
        ]
        self._corr_until = [-1.0] * len(self._corr_members)

    def _cap_mult(self, i: int) -> float:
        """Current dynamic capacity multiplier for path ``i`` (1.0 when static)."""
        d = self.config.dynamics
        if d is None or not d.enabled:
            return 1.0
        m = self._regime_mult[i]
        if self._burst_until[i] > self._t:
            m *= d.burst_intensity
        for g, members in enumerate(self._corr_members):
            if self._corr_until[g] > self._t and i in members:
                m *= d.corr_intensity
        return m

    def _path_capacity(self, i: int, t: float, rng: np.random.Generator) -> float:
        """Base sinusoidal envelope scaled by the current dynamic multiplier."""
        return self._traces[i].capacity_mbps(t, rng) * self._cap_mult(i)

    def _advance_dynamics(self) -> None:
        """Step the regime/burst/churn/correlation machines by one frame.

        Called at the *end* of ``step_frame`` (after the clock advances), so the
        next observation and the next delivery both see the same updated state.
        Event probabilities convert per-second hazards over one frame interval via
        ``1 - exp(-rate*dt)``. Draw order is fixed (regime, burst, corr, churn) to
        stay deterministic per seed.
        """
        d = self.config.dynamics
        if d is None or not d.enabled:
            return
        t = self._t
        dt = 1.0 / self.config.fps
        rng = self._rng
        n = self.num_paths

        if d.regime:
            p = 1.0 - math.exp(-d.regime_rate * dt)
            for i in range(n):
                if rng.random() < p:
                    self._regime_mult[i] = float(rng.uniform(d.regime_lo, d.regime_hi))
        if d.burst:
            p = 1.0 - math.exp(-d.burst_rate * dt)
            for i in range(n):
                if rng.random() < p:
                    self._burst_until[i] = t + d.burst_duration_s
        if self._corr_members:
            p = 1.0 - math.exp(-d.corr_rate * dt)
            for g in range(len(self._corr_members)):
                if rng.random() < p:
                    self._corr_until[g] = t + d.corr_duration_s
        if d.churn:
            p_up = 1.0 - math.exp(-d.churn_up_rate * dt)
            p_down = 1.0 - math.exp(-d.churn_down_rate * dt)
            nxt = list(self._active)
            for i in range(n):
                if self._active[i]:
                    if rng.random() < p_down:
                        nxt[i] = False
                elif rng.random() < p_up:
                    nxt[i] = True
            # Never let the active set fall below min_active (bring lowest idx up).
            if sum(nxt) < d.min_active:
                for i in range(n):
                    if sum(nxt) >= d.min_active:
                        break
                    nxt[i] = True
            self._active = nxt

    # -- DataPlane API ------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None) -> FrameObs:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._reset_state(self.config.video.init_bitrate_kbps)
        return self.current_obs()

    @property
    def clock_s(self) -> float:
        return self._t

    def is_done(self) -> bool:
        return self._frame_idx >= self._frames_per_episode

    def current_obs(self) -> FrameObs:
        n = self.num_paths
        active = self._active
        cwnd, srtt, buf = [], [], []
        for i, tr in enumerate(self._traces):
            if not active[i]:
                # A churned-out path reports dead state; the liveness mask lets
                # the policy/baselines exclude it. srtt = base_rtt is a neutral
                # placeholder (it is masked out downstream).
                cwnd.append(0.0)
                srtt.append(tr.base_rtt_ms)
                buf.append(0.0)
                continue
            queue_s = max(0.0, self._busy_until[i] - self._t)
            cap = self._path_capacity(
                i, self._t, np.random.default_rng(self._frame_total * 131 + i)
            )
            srtt_i = tr.base_rtt_ms + queue_s * 1000.0
            srtt.append(srtt_i)
            # Backlog bytes still queued ahead of "now" on this path.
            buf.append(queue_s * cap * 1e6 / 8.0)
            # Plausible cwnd ~ bandwidth-delay product (bytes).
            cwnd.append(cap * 1e6 / 8.0 * tr.base_rtt_ms / 1000.0)

        rtt_w = sum(self._cur_split[i] * srtt[i] for i in range(n))
        agg_rtt = rtt_w if rtt_w > 0 else self._rtt_ewma

        return FrameObs(
            num_paths=n,
            clock_s=self._t,
            done=self.is_done(),
            app_decision_due=(self._frame_idx % self._frames_per_app == 0),
            current_bitrate_kbps=self._cur_bitrate,
            rtt_ms=agg_rtt,
            jitter_ms=self._jitter_ewma,
            loss=self._loss_ewma,
            throughput_mbps=self._thr_ewma,
            cwnd=cwnd,
            srtt_ms=srtt,
            buffer_occ=buf,
            path_throughput_mbps=[
                self._path_thr_ewma[i] if active[i] else 0.0 for i in range(n)
            ],
            path_loss=[self._path_loss[i] if active[i] else 1.0 for i in range(n)],
            path_active=[1.0 if active[i] else 0.0 for i in range(n)],
            last_latency_ms=self._last.latency_ms,
            last_jitter_ms=self._last.jitter_ms,
            last_loss=self._last.loss,
            last_bytes=self._last.bytes_delivered,
        )

    def step_frame(
        self, target_bitrate_kbps: float, split_ratio: Sequence[float]
    ) -> FrameResult:
        n = self.num_paths
        self._cur_bitrate = self.config.video.clamp_bitrate(float(target_bitrate_kbps))
        self._cur_split = _normalize_split(split_ratio, n)

        total_bytes = frame_bytes(
            self._cur_bitrate, self._frame_total, self.config.video, self._rng
        )
        shares = _split_bytes(total_bytes, self._cur_split)
        active = self._active

        # Deliver each share through the per-path queue model. Bytes routed onto a
        # churned-out path never arrive and count as lost (the penalty that teaches
        # the scheduler to respect the liveness mask).
        arrivals: List[float] = []
        net_losses: List[float] = []
        dropped_bytes = 0
        for i in range(n):
            b = shares[i]
            if b == 0:
                continue
            if not active[i]:
                dropped_bytes += b
                self._path_loss[i] = 1.0
                continue
            cap = self._path_capacity(i, self._t, self._rng)
            ser_s = (b * 8.0) / (cap * 1e6)
            ready = max(self._t, self._busy_until[i])
            finish = ready + ser_s
            self._busy_until[i] = finish
            arrival = finish + (self._traces[i].base_rtt_ms / 1000.0) / 2.0
            arrivals.append(arrival)
            loss_i = self._traces[i].network_loss(cap)
            net_losses.append(loss_i)
            # Per-path goodput EWMA from this share.
            gp = (b * 8.0) / (max(arrival - self._t, 1e-6) * 1e6)
            self._path_thr_ewma[i] = 0.6 * self._path_thr_ewma[i] + 0.4 * gp
            self._path_loss[i] = loss_i

        dropped_frac = dropped_bytes / max(1, total_bytes)
        if arrivals:
            completion = max(arrivals)
            latency_ms = (completion - self._t) * 1000.0
        else:
            # Whole frame routed onto dead (or empty) paths: a deadline miss.
            latency_ms = 2.0 * self.config.deadline_ms
        late = latency_ms > self.config.deadline_ms
        net_loss = max(net_losses) if net_losses else 0.0
        base_loss = 1.0 if late else net_loss
        app_loss = float(min(1.0, max(base_loss, dropped_frac)))
        jitter_ms = (
            abs(latency_ms - self._prev_latency_ms) if self._prev_latency_ms >= 0.0 else 0.0
        )
        self._prev_latency_ms = latency_ms
        goodput_mbps = (total_bytes * 8.0) / (max(latency_ms / 1000.0, 1e-6) * 1e6)
        self._jitter_ewma = 0.7 * self._jitter_ewma + 0.3 * jitter_ms
        self._loss_ewma = 0.9 * self._loss_ewma + 0.1 * app_loss
        self._thr_ewma = 0.7 * self._thr_ewma + 0.3 * goodput_mbps
        self._rtt_ewma = (
            latency_ms if self._rtt_ewma <= 0 else 0.8 * self._rtt_ewma + 0.2 * latency_ms
        )
        delivered = 0 if (late or not arrivals) else (total_bytes - dropped_bytes)
        self._last = FrameResult(
            latency_ms=latency_ms,
            jitter_ms=jitter_ms,
            loss=app_loss,
            bytes_delivered=delivered,
        )

        # Advance the wall clock by one frame interval (real-time cadence), then
        # advance the dynamic state so the next obs/delivery share it.
        self._t += 1.0 / self.config.fps
        self._frame_idx += 1
        self._frame_total += 1
        self._advance_dynamics()
        return self._last


# --------------------------------------------------------------------------- #
# NS-3 backend (ns3-ai shared-memory bridge)
# --------------------------------------------------------------------------- #


_DEFAULT_NS3_DIR = os.path.expanduser("~/ns-3-dev")
_EXAMPLE_SUBPATH = os.path.join("contrib", "ai", "examples", "rl-mpquic")
_NS3_TARGET = "ns3ai_realtime_mpquic"
_PY_MODULE = "ns3ai_realtime_mpquic_py"

# Action commands; must match enum ActCommand in ns3/realtime_mpquic.h.
_CMD_STEP = 0
_CMD_RESET = 1
_CMD_TERMINATE = 2


@dataclass
class Ns3Config:
    """Episode parameters forwarded to the NS-3 scenario CLI on launch."""

    fps: float = 30.0
    episode_seconds: float = 30.0
    app_period_s: float = 1.0
    deadline_ms: float = 180.0
    video: VideoSourceConfig = field(default_factory=VideoSourceConfig)
    seed: int = 1

    # Per-path transport backend: "tcp" (default, byte-identical to before) or
    # "udp" (explicit app-layer deadline-drop instead of TCP's reliable
    # in-order retransmission -- see ns3/realtime_mpquic.cc's RealtimeSource).
    transport: str = "tcp"

    # Optional non-stationary dynamics forwarded to the C++ body (None / disabled
    # => the static NS-3 scenario, byte-identical to before). The same
    # `DynamicsConfig` the mock consumes; the C++ mirrors churn/regime/burst and
    # correlated failures, and emits the matching `pathActive[]` mask.
    dynamics: Optional[DynamicsConfig] = None

    # Optional per-path topology (list of {rate, delay, cross_frac} dicts). When
    # given it replaces the C++ scenario's hardcoded default path list, so a YAML
    # like configs/dynamic.yaml drives the same path count on both backends (and
    # its corr_groups indices become in-range). None => C++ default topology.
    topology: Optional[Sequence[dict]] = None

    # Diagnostic: make the C++ body log per-path churn/connection state to
    # stderr (pair with show_output=True). Off for normal runs.
    churn_log: bool = False


def _encode_topology(paths: Sequence[dict]) -> str:
    """Serialize the topology for the NS-3 CLI (shell-safe).

    Paths are separated by ':' and the ``rate,delay,cross_frac`` fields within a
    path by ',', e.g. ``"3Mbps,10ms,0.4:2Mbps,20ms,0.55"``. Rate/delay tokens are
    alphanumeric and cross_frac is a plain decimal, so no shell metacharacters
    appear (the setting dict is spliced into a ``shell=True`` command line).
    """
    return ":".join(
        f"{p['rate']},{p['delay']},{float(p.get('cross_frac', 0.4))}" for p in paths
    )


def _encode_corr_groups(groups: Sequence[Sequence[int]]) -> str:
    """Serialize correlated-failure groups for the NS-3 CLI (shell-safe).

    Groups are separated by ':' and member indices within a group by ',', e.g.
    ``[[4, 5], [0, 1]]`` -> ``"4,5:0,1"``. Avoids shell metacharacters (';', '|')
    since the setting dict is spliced into a ``shell=True`` command line.
    """
    return ":".join(",".join(str(int(x)) for x in grp) for grp in groups if len(grp))


class Ns3DataPlane(DataPlane):
    """Drives the real NS-3 ``realtime_mpquic`` scenario via the ns3-ai bridge.

    Python is the parent process: :meth:`reset` launches the NS-3 binary through
    ``ns3ai_utils.Experiment`` and drives the per-frame loop over shared memory.
    The C++ controller leads every decision with a send, so:

    * ``reset``        -> start the process (or in-band RESET), receive the
      initial observation.
    * ``step_frame``   -> send the action, receive the next observation, whose
      ``last_*`` fields carry the realized result of the frame just delivered.

    ns3-ai allows only **one** shared-memory creator per Python process, so a
    single long-lived NS-3 process serves the whole run. Use one ``Ns3DataPlane``
    per process; :meth:`close` ends the process and frees the shared memory.
    """

    def __init__(
        self,
        *,
        ns3_dir: Optional[str] = None,
        config: Optional[Ns3Config] = None,
        cap_mbps: float = 14.0,
        show_output: bool = False,
    ):
        self.config = config or Ns3Config()
        self.cap_mbps = float(cap_mbps)
        self.num_paths = 0
        self.show_output = bool(show_output)
        self.ns3_dir = os.path.abspath(
            ns3_dir or os.environ.get("NS3_DIR", _DEFAULT_NS3_DIR)
        )

        self._binding = None
        self._exp = None
        self._msg = None
        self._obs: Optional[FrameObs] = None
        self._finished = False
        self._started = False
        self._owe_send = False  # C++ blocked waiting for our action

    # -- lifecycle ---------------------------------------------------------- #

    def _import_binding(self):
        if self._binding is not None:
            return self._binding
        example_dir = os.path.join(self.ns3_dir, _EXAMPLE_SUBPATH)
        so_glob = os.path.join(example_dir, f"{_PY_MODULE}*.so")
        if not glob.glob(so_glob):
            raise RuntimeError(
                f"ns3-ai binding {_PY_MODULE} not found in {example_dir}. Build it "
                f"with `scripts/install_ns3_example.sh` (or `./ns3 build {_NS3_TARGET}`)."
            )
        if example_dir not in sys.path:
            sys.path.insert(0, example_dir)
        import importlib

        try:
            self._binding = importlib.import_module(_PY_MODULE)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"Failed to import {_PY_MODULE} from {example_dir}; the NS-3 backend "
                "only runs inside the WSL2/Linux NS-3 environment."
            ) from exc
        return self._binding

    def _teardown(self) -> None:
        if self._exp is not None:
            try:
                self._exp.kill()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._exp = None
            self._msg = None
            gc.collect()

    def _launch(self) -> None:
        binding = self._import_binding()
        # ns3ai_utils launches `./ns3` as a subprocess; keep it on the venv path.
        venv_bin = os.path.dirname(sys.executable)
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

        # ns3ai_utils ships in the ns3-ai contrib tree, not on PyPI.
        utils_dir = os.path.join(self.ns3_dir, "contrib", "ai", "python_utils")
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)

        from ns3ai_utils import Experiment  # NS-3 env only

        if self.config.transport not in ("tcp", "udp"):
            raise ValueError(
                f"Ns3Config.transport must be 'tcp' or 'udp', got {self.config.transport!r}"
            )

        cwd = os.getcwd()
        try:
            self._exp = Experiment(_NS3_TARGET, self.ns3_dir, binding, handleFinish=True)
            setting = {
                "fps": self.config.fps,
                "episodeSeconds": self.config.episode_seconds,
                "appPeriodS": self.config.app_period_s,
                "deadlineMs": self.config.deadline_ms,
                "initBitrateKbps": self.config.video.init_bitrate_kbps,
                "minBitrateKbps": self.config.video.min_bitrate_kbps,
                "maxBitrateKbps": self.config.video.max_bitrate_kbps,
                "seed": max(1, int(self.config.seed)),  # NS-3 rejects seed 0
                "transport": self.config.transport,
            }
            if self.config.topology:
                setting["paths"] = _encode_topology(self.config.topology)
            d = self.config.dynamics
            if d is not None and d.enabled:
                setting["dynamicsEnabled"] = 1
                setting["churn"] = 1 if d.churn else 0
                setting["churnUpRate"] = d.churn_up_rate
                setting["churnDownRate"] = d.churn_down_rate
                setting["minActive"] = int(d.min_active)
                setting["regime"] = 1 if d.regime else 0
                setting["regimeRate"] = d.regime_rate
                setting["regimeLo"] = d.regime_lo
                setting["regimeHi"] = d.regime_hi
                setting["burst"] = 1 if d.burst else 0
                setting["burstRate"] = d.burst_rate
                setting["burstIntensity"] = d.burst_intensity
                setting["burstDurationS"] = d.burst_duration_s
                setting["corrRate"] = d.corr_rate
                setting["corrIntensity"] = d.corr_intensity
                setting["corrDurationS"] = d.corr_duration_s
                corr = _encode_corr_groups(d.corr_groups)
                if corr:
                    setting["corrGroups"] = corr
            else:
                setting["dynamicsEnabled"] = 0
            if self.config.churn_log:
                setting["churnLog"] = 1
            self._msg = self._exp.run(setting=setting, show_output=self.show_output)
        finally:
            os.chdir(cwd)
        self._started = True
        self._finished = False

    def reset(self, *, seed: Optional[int] = None) -> FrameObs:
        if seed is not None:
            self.config.seed = max(1, int(seed))
        if not self._started:
            self._launch()
            self._obs = None
            self._recv_obs()
        else:
            self._send_act(_CMD_RESET)
            self._recv_obs()
        if self._obs is not None:
            self.num_paths = self._obs.num_paths
        return self.current_obs()

    # -- bridge primitives -------------------------------------------------- #

    def _recv_obs(self) -> None:
        msg = self._msg
        msg.PyRecvBegin()
        if msg.PyGetFinished():
            self._finished = True
            self._owe_send = False
            msg.PyRecvEnd()
            return
        e = msg.GetCpp2PyStruct()
        n = int(e.numPaths)
        self._obs = FrameObs(
            num_paths=n,
            clock_s=float(e.clockS),
            done=bool(e.done),
            app_decision_due=bool(e.appDecisionDue),
            current_bitrate_kbps=float(e.currentBitrateKbps),
            rtt_ms=float(e.rttMs),
            jitter_ms=float(e.jitterMs),
            loss=float(e.loss),
            throughput_mbps=float(e.throughputMbps),
            cwnd=[float(e.cwnd(i)) for i in range(n)],
            srtt_ms=[float(e.srtt(i)) for i in range(n)],
            buffer_occ=[float(e.bufferOcc(i)) for i in range(n)],
            path_throughput_mbps=[float(e.pathThroughput(i)) for i in range(n)],
            path_loss=[float(e.pathLoss(i)) for i in range(n)],
            # Per-path liveness mask emitted by the C++ body's churn machine
            # (all-ones when dynamics are disabled). See EnvStruct.pathActive[].
            path_active=[float(e.pathActive(i)) for i in range(n)],
            last_latency_ms=float(e.lastLatencyMs),
            last_jitter_ms=float(e.lastJitterMs),
            last_loss=float(e.lastLoss),
            last_bytes=int(e.lastBytes),
        )
        msg.PyRecvEnd()
        self._owe_send = True

    def _send_act(
        self,
        command: int,
        target_bitrate_kbps: float = 0.0,
        split_ratio: Optional[Sequence[float]] = None,
    ) -> None:
        msg = self._msg
        msg.PySendBegin()
        a = msg.GetPy2CppStruct()
        a.command = int(command)
        a.targetBitrateKbps = float(target_bitrate_kbps)
        if split_ratio is not None:
            for i, v in enumerate(split_ratio):
                a.setSplit(i, float(v))
        msg.PySendEnd()
        self._owe_send = False

    # -- DataPlane API ------------------------------------------------------ #

    @property
    def clock_s(self) -> float:
        return float(self._obs.clock_s) if self._obs is not None else 0.0

    def current_obs(self) -> FrameObs:
        if self._obs is None:
            raise RuntimeError("Ns3DataPlane.reset() must be called before use")
        return self._obs

    def is_done(self) -> bool:
        if self._finished:
            return True
        return bool(self._obs.done) if self._obs is not None else False

    def step_frame(
        self, target_bitrate_kbps: float, split_ratio: Sequence[float]
    ) -> FrameResult:
        if self._obs is None:
            raise RuntimeError("Ns3DataPlane.reset() must be called before use")
        if self.is_done():
            raise RuntimeError("episode already finished; call reset()")
        split = _normalize_split(split_ratio, self.num_paths)
        self._send_act(_CMD_STEP, target_bitrate_kbps, split)
        self._recv_obs()
        if self._obs is None:  # process finished unexpectedly
            raise RuntimeError("NS-3 process ended before reporting the frame")
        e = self._obs
        return FrameResult(
            latency_ms=e.last_latency_ms,
            jitter_ms=e.last_jitter_ms,
            loss=e.last_loss,
            bytes_delivered=int(e.last_bytes),
        )

    def close(self) -> None:
        """End the NS-3 process gracefully and release the shared memory."""
        if self._started and not self._finished and self._owe_send:
            try:
                self._send_act(_CMD_TERMINATE)
            except Exception:  # pragma: no cover - process may already be gone
                pass
        self._teardown()
        self._obs = None
        self._finished = False
        self._started = False
        self._owe_send = False

    def __del__(self):  # pragma: no cover - best-effort cleanup
        self._teardown()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalize_split(split_ratio: Sequence[float], n: int) -> List[float]:
    """Clamp negatives and renormalize to sum 1; even split if degenerate."""
    vals = [max(0.0, float(split_ratio[i])) if i < len(split_ratio) else 0.0 for i in range(n)]
    s = sum(vals)
    if s <= 1e-9:
        return [1.0 / n] * n
    return [v / s for v in vals]


def _split_bytes(total_bytes: int, split: Sequence[float]) -> List[int]:
    """Split ``total_bytes`` across paths by ``split``, reconciling rounding."""
    n = len(split)
    shares = [int(round(total_bytes * split[i])) for i in range(n)]
    assigned = sum(shares)
    largest = max(range(n), key=lambda i: split[i]) if n else 0
    if assigned < total_bytes:
        shares[largest] += total_bytes - assigned
    elif assigned > total_bytes:
        shares[largest] = max(0, shares[largest] - (assigned - total_bytes))
    return shares
