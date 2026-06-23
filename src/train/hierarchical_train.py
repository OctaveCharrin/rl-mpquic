"""
Dual-agent hierarchical training loop.

Drives one long-lived data-plane backend (mock or NS-3) frame by frame:

* **every frame** the Transport agent observes per-path transport state + the
  current target bitrate, picks a split, the env delivers the frame, and the
  agent is stored/updated on the per-frame transport reward;
* **every ``app_period_s``** the App agent picks a new target bitrate; the
  *previous* App action is credited with the VMAF-QoE accumulated over the window
  of frames it governed (a delayed, hierarchical reward).

Episodes are delimited in-band (the backend keeps the network warm across them).
Checkpoints (``app.pth`` / ``transport.pth``) and ``stats.json`` are written to
the run directory.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from ..ns3env.realtime_env import HierarchicalRealtimeEnv
from ..rl.app_agent import AppAgent
from ..rl.transport_agent import TransportAgent
from .config import ExperimentConfig


def _app_sac_config(cfg: ExperimentConfig, frames_per_app: int):
    """Scale the App agent's warm-up to its slower (per-second) cadence."""
    fpa = max(1, frames_per_app)
    return dataclasses.replace(
        cfg.sac,
        start_steps=max(50, cfg.sac.start_steps // fpa),
        update_after=max(50, cfg.sac.update_after // fpa),
    )


def run_training(
    cfg: ExperimentConfig,
    *,
    backend: str = "mock",
    episodes: Optional[int] = None,
    show_output: bool = False,
    out_dir: Optional[str] = None,
    seed: Optional[int] = None,
    log_every: int = 1,
) -> Dict[str, object]:
    """Train both agents; return a stats dict (also written to disk)."""
    episodes = int(episodes if episodes is not None else cfg.episodes)
    base_seed = int(seed if seed is not None else cfg.seed)
    out_dir = out_dir or os.path.join(cfg.out_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    dp = cfg.make_dataplane(backend, seed=base_seed, show_output=show_output)
    env = HierarchicalRealtimeEnv(
        dp, video=cfg.video, weights=cfg.weights, episode_seconds=cfg.episode_seconds
    )
    # First reset establishes num_paths (needed to size the transport agent).
    env.reset(seed=base_seed)
    frames_per_app = max(1, round(cfg.fps * cfg.app_period_s))

    app = AppAgent(
        env.app_obs_dim,
        min_kbps=cfg.video.min_bitrate_kbps,
        max_kbps=cfg.video.max_bitrate_kbps,
        config=_app_sac_config(cfg, frames_per_app),
    )
    transport = TransportAgent(env.transport_obs_dim, env.num_paths, config=cfg.sac)

    history: List[Dict[str, float]] = []
    try:
        for ep in range(episodes):
            stats = _run_episode(env, app, transport, seed=base_seed + ep, episode=ep)
            history.append(stats)
            if log_every and ep % log_every == 0:
                print(
                    f"[ep {ep:3d}] QoE={stats['app_reward_mean']:+.3f} "
                    f"T={stats['transport_reward_mean']:+.3f} "
                    f"bitrate={stats['bitrate_mean_kbps']:6.0f}kbps "
                    f"lat={stats['latency_mean_ms']:6.1f}ms "
                    f"loss={stats['loss_mean']:.3f}",
                    flush=True,
                )
    finally:
        dp.close()

    # Persist checkpoints + stats.
    torch.save(app.sac.state_dict(), os.path.join(out_dir, "app.pth"))
    torch.save(transport.sac.state_dict(), os.path.join(out_dir, "transport.pth"))
    result = {
        "backend": backend,
        "episodes": episodes,
        "num_paths": env.num_paths,
        "config": _config_summary(cfg),
        "history": history,
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"saved checkpoints + stats to {out_dir}")
    return result


def _run_episode(
    env: HierarchicalRealtimeEnv,
    app: AppAgent,
    transport: TransportAgent,
    *,
    seed: int,
    episode: int,
) -> Dict[str, float]:
    obs = env.reset(seed=seed)
    target_kbps = obs.current_bitrate_kbps

    prev_app_obs: Optional[np.ndarray] = None
    prev_app_raw: Optional[np.ndarray] = None

    ep_app_r, n_app = 0.0, 0
    ep_t_r, n_frames = 0.0, 0
    latencies, losses, bitrates, vmafs = [], [], [], []

    done = env.is_done()
    while not done:
        # --- App decision (slow cadence) ---
        if obs.app_decision_due:
            cur_app_obs = env.build_app_obs(obs)
            if prev_app_obs is not None:
                app_r, comps = env.pop_app_window_reward()
                app.store(prev_app_obs, prev_app_raw, app_r, cur_app_obs, False)
                app.update()
                ep_app_r += app_r
                n_app += 1
                vmafs.append(comps["vmaf"])
            target_kbps, prev_app_raw = app.select(cur_app_obs)
            prev_app_obs = cur_app_obs

        # --- Transport decision (every frame) ---
        t_obs = env.build_transport_obs(obs, target_kbps)
        split, t_raw = transport.select(t_obs)
        next_obs, t_r, done, info = env.step(target_kbps, split)
        t_next_obs = env.build_transport_obs(next_obs, target_kbps)
        transport.store(t_obs, t_raw, t_r, t_next_obs, done)
        transport.update()

        ep_t_r += t_r
        n_frames += 1
        latencies.append(info.latency_ms)
        losses.append(info.loss)
        bitrates.append(target_kbps)
        obs = next_obs

    # Credit the final (partial) App window.
    if prev_app_obs is not None:
        app_r, comps = env.pop_app_window_reward()
        final_app_obs = env.build_app_obs(obs)
        app.store(prev_app_obs, prev_app_raw, app_r, final_app_obs, True)
        app.update()
        ep_app_r += app_r
        n_app += 1
        vmafs.append(comps["vmaf"])

    return {
        "episode": episode,
        "frames": n_frames,
        "app_decisions": n_app,
        "app_reward_mean": ep_app_r / max(1, n_app),
        "transport_reward_mean": ep_t_r / max(1, n_frames),
        "bitrate_mean_kbps": float(np.mean(bitrates)) if bitrates else 0.0,
        "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "loss_mean": float(np.mean(losses)) if losses else 0.0,
        "vmaf_mean": float(np.mean(vmafs)) if vmafs else 0.0,
    }


def _config_summary(cfg: ExperimentConfig) -> Dict[str, object]:
    return {
        "fps": cfg.fps,
        "episode_seconds": cfg.episode_seconds,
        "app_period_s": cfg.app_period_s,
        "deadline_ms": cfg.deadline_ms,
        "bitrate_kbps": [cfg.video.min_bitrate_kbps, cfg.video.max_bitrate_kbps],
        "reward_weights": cfg.weights.to_dict(),
        "paths": cfg.paths,
    }
