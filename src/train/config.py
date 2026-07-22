"""
Experiment configuration: parse ``configs/*.yaml`` into typed config objects and
build the data-plane backends.

One YAML drives both backends so the mock and NS-3 runs are comparable: the
``topology`` paths become NS-3 link attributes *and* the mock's per-path capacity
baselines (rate -> base Mbps, delay -> base RTT, cross_frac -> mean cross-traffic).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from ..ns3env.dataplane import (
    DataPlane,
    DynamicsConfig,
    MockRealtimeConfig,
    MockRealtimeDataPlane,
    Ns3Config,
    Ns3DataPlane,
)
from ..ns3env.qoe import QoEWeights, VmafFn
from ..ns3env.video_source import VideoSourceConfig
from ..rl.sac_agent import SACConfig

_RATE_UNITS = {"bps": 1e-6, "kbps": 1e-3, "mbps": 1.0, "gbps": 1e3}


def parse_rate_mbps(rate: str) -> float:
    """'8Mbps' -> 8.0, '500kbps' -> 0.5 (Mbps)."""
    m = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]+)\s*", str(rate))
    if not m:
        raise ValueError(f"unparseable rate: {rate!r}")
    value, unit = float(m.group(1)), m.group(2).lower()
    if unit not in _RATE_UNITS:
        raise ValueError(f"unknown rate unit in {rate!r}")
    return value * _RATE_UNITS[unit]


def parse_delay_ms(delay: str) -> float:
    """'10ms' -> 10.0, '0.5s' -> 500.0 (one-way delay, milliseconds)."""
    m = re.fullmatch(r"\s*([0-9.]+)\s*(ms|s|us)?\s*", str(delay))
    if not m:
        raise ValueError(f"unparseable delay: {delay!r}")
    value = float(m.group(1))
    unit = (m.group(2) or "ms").lower()
    return {"ms": 1.0, "s": 1000.0, "us": 0.001}[unit] * value


@dataclass
class ExperimentConfig:
    paths: List[Dict[str, Any]] = field(default_factory=list)
    video: VideoSourceConfig = field(default_factory=VideoSourceConfig)
    weights: QoEWeights = field(default_factory=QoEWeights)
    sac: SACConfig = field(default_factory=SACConfig)
    fps: float = 30.0
    episode_seconds: float = 30.0
    app_period_s: float = 1.0
    deadline_ms: float = 180.0
    warmup_s: float = 1.0
    episodes: int = 50
    seed: int = 1
    cap_mbps: float = 10.0
    out_dir: str = "runs"
    # Path-agent architecture: "flat" (legacy fixed-dim MLP SAC) or
    # "scoring" (permutation-equivariant, variable-path-count SAC). "flat" is the
    # default so existing configs/checkpoints are unaffected.
    path_arch: str = "flat"
    # Per-path transport backend for the NS-3 scenario: "tcp" (default) or "udp"
    # (explicit app-layer deadline-drop instead of TCP retransmission). Ignored
    # by the mock backend, which is already UDP-like.
    transport: str = "tcp"
    # Optional non-stationary mock dynamics (None => fully static mock network).
    # Ignored by the NS-3 backend.
    dynamics: Optional[DynamicsConfig] = None
    # Use the WebRTC-grounded learned QoS->VMAF surrogate for the App reward's
    # quality term instead of the default bitrate-only log curve.
    use_learned_vmaf: bool = False
    learned_vmaf_model: Optional[str] = None  # optional explicit model path

    # -- derived objects ---------------------------------------------------- #

    def build_vmaf_fn(self) -> Optional[VmafFn]:
        """Learned VMAF scorer if enabled, else None (default log curve)."""
        if not self.use_learned_vmaf:
            return None
        from ..ns3env.learned_vmaf import load_learned_vmaf_fn

        return load_learned_vmaf_fn(self.learned_vmaf_model)

    # -- derived backends --------------------------------------------------- #

    def mock_dataplane(self, seed: Optional[int] = None) -> MockRealtimeDataPlane:
        base_mbps = [parse_rate_mbps(p["rate"]) for p in self.paths]
        base_rtt = [2.0 * parse_delay_ms(p["delay"]) for p in self.paths]
        cross = [float(p.get("cross_frac", 0.4)) for p in self.paths]
        n = len(self.paths)
        # Per-path seasonality periods, cycled so any path count is covered
        # (truncating to [:n] under-fills when n exceeds the base list length).
        # The amp/period formulas here are duplicated by EnvelopeMult in
        # ns3/realtime_mpquic.cc (dynamics-only capacity envelope) — keep in sync.
        seasons = [12.0, 7.0, 20.0]
        cross_periods = [5.0, 3.0, 8.0]
        cfg = MockRealtimeConfig(
            base_mbps=base_mbps,
            base_rtt_ms=base_rtt,
            amp=[0.45 + 0.1 * (i % 3) for i in range(n)],
            period_s=[seasons[i % len(seasons)] for i in range(n)],
            cross_frac=cross,
            cross_period_s=[cross_periods[i % len(cross_periods)] for i in range(n)],
            fps=self.fps,
            episode_seconds=self.episode_seconds,
            app_period_s=self.app_period_s,
            deadline_ms=self.deadline_ms,
            video=self.video,
            seed=seed if seed is not None else self.seed,
            dynamics=self.dynamics,
        )
        return MockRealtimeDataPlane(cfg)

    def ns3_dataplane(
        self, seed: Optional[int] = None, show_output: bool = False
    ) -> Ns3DataPlane:
        cfg = Ns3Config(
            fps=self.fps,
            episode_seconds=self.episode_seconds,
            app_period_s=self.app_period_s,
            deadline_ms=self.deadline_ms,
            video=self.video,
            seed=seed if seed is not None else self.seed,
            dynamics=self.dynamics,
            topology=self.paths,
            transport=self.transport,
        )
        # Aggregate capacity cap ~ sum of nominal link rates (Mbps).
        cap = sum(parse_rate_mbps(p["rate"]) for p in self.paths) or self.cap_mbps
        return Ns3DataPlane(config=cfg, cap_mbps=cap, show_output=show_output)

    def make_dataplane(
        self, backend: str, seed: Optional[int] = None, show_output: bool = False
    ) -> DataPlane:
        if backend == "mock":
            return self.mock_dataplane(seed)
        if backend == "ns3":
            return self.ns3_dataplane(seed, show_output)
        raise ValueError(f"unknown backend {backend!r} (use 'mock' or 'ns3')")


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    """Load an ExperimentConfig from YAML (defaults if ``path`` is None)."""
    cfg = ExperimentConfig()
    if path is None:
        cfg.paths = _DEFAULT_PATHS()
        return cfg

    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}

    topo = data.get("topology", {})
    cfg.paths = topo.get("paths", _DEFAULT_PATHS())

    vid = data.get("video", {})
    cfg.video = VideoSourceConfig(
        fps=float(vid.get("fps", 30.0)),
        min_bitrate_kbps=float(vid.get("min_bitrate_kbps", 300.0)),
        max_bitrate_kbps=float(vid.get("max_bitrate_kbps", 6000.0)),
        init_bitrate_kbps=float(vid.get("init_bitrate_kbps", 1500.0)),
        frame_size_jitter=float(vid.get("frame_size_jitter", 0.25)),
        keyframe_interval=int(vid.get("keyframe_interval", 30)),
    )
    cfg.fps = cfg.video.fps

    reward = data.get("reward", {})
    cfg.weights = QoEWeights.from_mapping(reward)
    cfg.use_learned_vmaf = bool(reward.get("use_learned_vmaf", False))
    model_path = reward.get("learned_vmaf_model")
    cfg.learned_vmaf_model = str(model_path) if model_path else None

    ep = data.get("episode", {})
    cfg.episode_seconds = float(ep.get("seconds", 30.0))
    cfg.app_period_s = float(ep.get("app_period_s", 1.0))
    cfg.warmup_s = float(ep.get("warmup_s", 1.0))

    sac = data.get("sac", {})
    cfg.sac = SACConfig(
        hidden_dim=int(sac.get("hidden_dim", 256)),
        gamma=float(sac.get("gamma", 0.99)),
        tau=float(sac.get("tau", 0.005)),
        lr=float(sac.get("lr", 3e-4)),
        batch_size=int(sac.get("batch_size", 256)),
        buffer_size=int(sac.get("buffer_size", 200_000)),
        start_steps=int(sac.get("start_steps", 1_000)),
        update_after=int(sac.get("update_after", 1_000)),
        updates_per_step=int(sac.get("updates_per_step", 1)),
        auto_entropy=bool(sac.get("auto_entropy", True)),
        critic_layernorm=bool(sac.get("critic_layernorm", False)),
        critic_dropout=float(sac.get("critic_dropout", 0.0)),
        prioritized=bool(sac.get("prioritized", False)),
        per_alpha=float(sac.get("per_alpha", 0.6)),
        per_beta0=float(sac.get("per_beta0", 0.4)),
        per_beta_steps=int(sac.get("per_beta_steps", 100_000)),
    )

    run = data.get("run", {})
    cfg.episodes = int(run.get("episodes", 50))
    cfg.seed = int(run.get("seed", 1))
    cfg.cap_mbps = float(run.get("cap_mbps", 10.0))
    cfg.out_dir = str(run.get("out_dir", "runs"))
    cfg.path_arch = str(run.get("path_arch", sac.get("path_arch", "flat")))
    cfg.transport = str(run.get("transport", "tcp"))

    cfg.dynamics = _parse_dynamics(data.get("dynamics"))
    return cfg


def _parse_dynamics(d: Optional[Dict[str, Any]]) -> Optional[DynamicsConfig]:
    """Build a :class:`DynamicsConfig` from the optional ``dynamics:`` YAML block.

    Returns None when the block is absent or ``enabled: false`` so the mock stays
    fully static (legacy behavior). Unknown keys are ignored.
    """
    if not d or not bool(d.get("enabled", False)):
        return None
    groups = d.get("corr_groups", []) or []
    return DynamicsConfig(
        enabled=True,
        churn=bool(d.get("churn", False)),
        churn_up_rate=float(d.get("churn_up_rate", 0.10)),
        churn_down_rate=float(d.get("churn_down_rate", 0.05)),
        min_active=int(d.get("min_active", 1)),
        regime=bool(d.get("regime", False)),
        regime_rate=float(d.get("regime_rate", 0.20)),
        regime_lo=float(d.get("regime_lo", 0.35)),
        regime_hi=float(d.get("regime_hi", 1.30)),
        burst=bool(d.get("burst", False)),
        burst_rate=float(d.get("burst_rate", 0.15)),
        burst_intensity=float(d.get("burst_intensity", 0.25)),
        burst_duration_s=float(d.get("burst_duration_s", 0.5)),
        corr_groups=[[int(x) for x in grp] for grp in groups],
        corr_rate=float(d.get("corr_rate", 0.05)),
        corr_intensity=float(d.get("corr_intensity", 0.30)),
        corr_duration_s=float(d.get("corr_duration_s", 1.0)),
    )


def _DEFAULT_PATHS() -> List[Dict[str, Any]]:
    return [
        {"rate": "8Mbps", "delay": "10ms", "cross_frac": 0.45},
        {"rate": "4Mbps", "delay": "17ms", "cross_frac": 0.65},
        {"rate": "2Mbps", "delay": "30ms", "cross_frac": 0.35},
    ]
