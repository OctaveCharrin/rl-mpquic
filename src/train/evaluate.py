"""
Evaluate a trained dual-agent policy against split/bitrate baselines.

Runs each policy for a few episodes on the chosen backend and reports the mean
VMAF-QoE (the App-agent reward) plus latency / loss / bitrate, so the learned
controller can be compared to simple heuristics:

* ``even``        — split every frame equally across paths.
* ``single``      — send the whole frame on the highest-throughput path.
* ``proportional``— split in proportion to recent per-path throughput.
* ``random``      — a fresh uniform-over-simplex split every frame (seeded).
* ``learned``     — the trained App + Transport agents, acting deterministically.

Baselines use a reactive bitrate heuristic (90% of recent aggregate goodput) so
the comparison reflects scheduling quality, not a frozen bitrate.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..ns3env.dataplane import FrameObs
from ..ns3env.realtime_env import HierarchicalRealtimeEnv
from ..rl.app_agent import AppAgent
from ..rl.transport_agent import TransportAgent
from .config import ExperimentConfig

BitrateFn = Callable[[FrameObs], float]
SplitFn = Callable[[FrameObs, float], np.ndarray]


def _heuristic_bitrate(cfg: ExperimentConfig) -> BitrateFn:
    def fn(obs: FrameObs) -> float:
        target = 0.9 * obs.throughput_mbps * 1000.0  # 90% of recent goodput (kbps)
        return cfg.video.clamp_bitrate(max(cfg.video.min_bitrate_kbps, target))

    return fn


def _even_split(obs: FrameObs, target: float) -> np.ndarray:
    n = obs.num_paths
    return np.full(n, 1.0 / n, dtype=np.float32)


def _single_best(obs: FrameObs, target: float) -> np.ndarray:
    n = obs.num_paths
    split = np.zeros(n, dtype=np.float32)
    split[int(np.argmax(obs.path_throughput_mbps))] = 1.0
    return split


def _proportional(obs: FrameObs, target: float) -> np.ndarray:
    thr = np.asarray(obs.path_throughput_mbps, dtype=np.float64)
    thr = np.clip(thr, 1e-6, None)
    return (thr / thr.sum()).astype(np.float32)


def _random_split(seed: int) -> SplitFn:
    """A non-reactive baseline: draw a fresh split uniformly over the simplex
    each frame. Seeded so the evaluation stays reproducible."""
    rng = np.random.default_rng(seed)

    def fn(obs: FrameObs, target: float) -> np.ndarray:
        return rng.dirichlet(np.ones(obs.num_paths)).astype(np.float32)

    return fn


def _rollout(
    env: HierarchicalRealtimeEnv,
    bitrate_fn: BitrateFn,
    split_fn: SplitFn,
    *,
    seed: int,
) -> Dict[str, float]:
    obs = env.reset(seed=seed)
    target = bitrate_fn(obs)
    app_rewards, latencies, losses, bitrates, vmafs = [], [], [], [], []
    have_window = False
    done = env.is_done()
    while not done:
        if obs.app_decision_due:
            if have_window:
                r, comps = env.pop_app_window_reward()
                app_rewards.append(r)
                vmafs.append(comps["vmaf"])
            target = bitrate_fn(obs)
            have_window = True
        split = split_fn(obs, target)
        next_obs, _t_r, done, info = env.step(target, split)
        latencies.append(info.latency_ms)
        losses.append(info.loss)
        bitrates.append(target)
        obs = next_obs
    if have_window:
        r, comps = env.pop_app_window_reward()
        app_rewards.append(r)
        vmafs.append(comps["vmaf"])
    return {
        "qoe": float(np.mean(app_rewards)) if app_rewards else 0.0,
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "bitrate_kbps": float(np.mean(bitrates)) if bitrates else 0.0,
        "vmaf": float(np.mean(vmafs)) if vmafs else 0.0,
    }


def _learned_policies(env, app_path, transport_path, cfg):
    """Build deterministic bitrate/split fns from trained checkpoints."""
    import torch

    app = AppAgent(
        env.app_obs_dim,
        min_kbps=cfg.video.min_bitrate_kbps,
        max_kbps=cfg.video.max_bitrate_kbps,
        config=cfg.sac,
    )
    transport = TransportAgent(env.transport_obs_dim, env.num_paths, config=cfg.sac)
    app.sac.load_state_dict(torch.load(app_path, map_location="cpu"))
    transport.sac.load_state_dict(torch.load(transport_path, map_location="cpu"))

    def bitrate_fn(obs: FrameObs) -> float:
        kbps, _ = app.select(env.build_app_obs(obs), deterministic=True)
        return kbps

    def split_fn(obs: FrameObs, target: float) -> np.ndarray:
        split, _ = transport.select(
            env.build_transport_obs(obs, target), deterministic=True
        )
        return split

    return bitrate_fn, split_fn


def run_evaluation(
    cfg: ExperimentConfig,
    *,
    backend: str = "mock",
    episodes: int = 5,
    seed: int = 1000,
    app_ckpt: Optional[str] = None,
    transport_ckpt: Optional[str] = None,
    show_output: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Evaluate baselines (+ learned policy if checkpoints given). Prints a table."""
    dp = cfg.make_dataplane(backend, seed=seed, show_output=show_output)
    env = HierarchicalRealtimeEnv(
        dp, video=cfg.video, weights=cfg.weights, episode_seconds=cfg.episode_seconds
    )
    env.reset(seed=seed)

    bitrate_heur = _heuristic_bitrate(cfg)
    policies: Dict[str, tuple] = {
        "even": (bitrate_heur, _even_split),
        "single": (bitrate_heur, _single_best),
        "proportional": (bitrate_heur, _proportional),
        "random": (bitrate_heur, _random_split(seed)),
    }
    if app_ckpt and transport_ckpt:
        policies["learned"] = _learned_policies(env, app_ckpt, transport_ckpt, cfg)

    results: Dict[str, Dict[str, float]] = {}
    try:
        for name, (bfn, sfn) in policies.items():
            runs: List[Dict[str, float]] = [
                _rollout(env, bfn, sfn, seed=seed + e) for e in range(episodes)
            ]
            results[name] = {
                k: float(np.mean([r[k] for r in runs])) for k in runs[0]
            }
    finally:
        dp.close()

    print(f"\n{'policy':<14}{'QoE':>8}{'VMAF':>8}{'lat(ms)':>10}{'loss':>8}{'kbps':>9}")
    print("-" * 57)
    for name, r in results.items():
        print(
            f"{name:<14}{r['qoe']:>8.3f}{r['vmaf']:>8.1f}"
            f"{r['latency_ms']:>10.1f}{r['loss']:>8.3f}{r['bitrate_kbps']:>9.0f}"
        )
    return results
