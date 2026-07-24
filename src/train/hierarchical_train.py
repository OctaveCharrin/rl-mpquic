"""
Dual-agent hierarchical training loop.

Drives one long-lived data-plane backend (mock or NS-3) frame by frame:

* **every frame** the Path agent observes per-path transport state + the
  current target bitrate, picks a split, the env delivers the frame, and the
  agent is stored/updated on the per-frame path reward;
* **every ``app_period_s``** the App agent picks a new target bitrate; the
  *previous* App action is credited with the VMAF-QoE accumulated over the window
  of frames it governed (a delayed, hierarchical reward).

Episodes are delimited in-band (the backend keeps the network warm across them).
Checkpoints (``app.pth`` / ``path.pth``) and ``stats.json`` are written to
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
from ..rl.path_agent import PathAgent
from .config import ExperimentConfig


def _build_path_agent(cfg: ExperimentConfig, env: HierarchicalRealtimeEnv) -> PathAgent:
    """Construct the Path agent for the configured architecture."""
    if cfg.path_arch in ("scoring", "scoring_attn"):
        return PathAgent(
            env.path_obs_dim,
            env.num_paths,
            config=cfg.sac,
            arch=cfg.path_arch,
            global_dim=env.path_global_dim,
            path_dim=env.path_feat_dim,
        )
    return PathAgent(env.path_obs_dim, env.num_paths, config=cfg.sac, arch="flat")


def _path_obs(env: HierarchicalRealtimeEnv, obs, target_kbps: float, arch: str):
    """Build the path-agent observation matching the agent architecture."""
    if arch in ("scoring", "scoring_attn"):  # both use the structured state
        return env.build_path_state(obs, target_kbps)
    return env.build_path_obs(obs, target_kbps)


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
    use_learned_vmaf: Optional[bool] = None,
    resume: bool = False,
    persist_buffer: bool = False,
    persist_buffer_every: int = 10,
) -> Dict[str, object]:
    """Train both agents; return a stats dict (also written to disk)."""
    episodes = int(episodes if episodes is not None else cfg.episodes)
    base_seed = int(seed if seed is not None else cfg.seed)
    out_dir = out_dir or os.path.join(cfg.out_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    if use_learned_vmaf is not None:
        cfg.use_learned_vmaf = bool(use_learned_vmaf)
    vmaf_fn = cfg.build_vmaf_fn()
    if cfg.use_learned_vmaf:
        print("App reward: using learned WebRTC QoS->VMAF surrogate", flush=True)

    dp = cfg.make_dataplane(backend, seed=base_seed, show_output=show_output)
    env = HierarchicalRealtimeEnv(
        dp,
        video=cfg.video,
        weights=cfg.weights,
        episode_seconds=cfg.episode_seconds,
        vmaf_fn=vmaf_fn,
    )
    # First reset establishes num_paths (needed to size the path agent).
    env.reset(seed=base_seed)
    frames_per_app = max(1, round(cfg.fps * cfg.app_period_s))

    app = AppAgent(
        env.app_obs_dim,
        min_kbps=cfg.video.min_bitrate_kbps,
        max_kbps=cfg.video.max_bitrate_kbps,
        config=_app_sac_config(cfg, frames_per_app),
    )
    path_agent = _build_path_agent(cfg, env)

    app_ckpt = os.path.join(out_dir, "app.pth")
    path_ckpt = os.path.join(out_dir, "path.pth")
    app_buf = os.path.join(out_dir, "app_buffer.npz")
    path_buf = os.path.join(out_dir, "path_buffer.npz")

    def _save_ckpts(save_buffers: bool = False) -> None:
        torch.save(app.sac.state_dict(), app_ckpt)
        torch.save(path_agent.sac.state_dict(), path_ckpt)
        # Buffers are large (up to buffer_size rows); only persist them on the
        # requested cadence, not after every episode.
        if save_buffers and persist_buffer:
            app.sac.save_buffer(app_buf)
            path_agent.sac.save_buffer(path_buf)

    # Optional resume: reload the latest checkpoints — lets a long train be split
    # across several shorter, interruptible runs. With --persist-buffer the replay
    # buffers are restored too (updates continue immediately); without them the
    # buffer restarts empty, so we skip the uniform-random warm-up (the legacy
    # cold-start hack) and let updates resume once it refills.
    if resume and os.path.exists(app_ckpt) and os.path.exists(path_ckpt):
        app.sac.load_state_dict(torch.load(app_ckpt, map_location="cpu"))
        path_agent.sac.load_state_dict(torch.load(path_ckpt, map_location="cpu"))
        buffers_restored = False
        if persist_buffer and os.path.exists(app_buf) and os.path.exists(path_buf):
            app.sac.load_buffer(app_buf)
            path_agent.sac.load_buffer(path_buf)
            buffers_restored = True
            print(
                f"restored replay buffers (app={len(app.sac.buffer)}, "
                f"path={len(path_agent.sac.buffer)})",
                flush=True,
            )
        if not buffers_restored:
            app.sac._stores = max(app.sac._stores, app.sac.cfg.start_steps)
            path_agent.sac._stores = max(
                path_agent.sac._stores, path_agent.sac.cfg.start_steps
            )
        print(f"resumed from checkpoints in {out_dir}", flush=True)

    # Optional per-episode domain randomization of the mock dynamics (Tier-2 #7).
    # Mock-only: the NS-3 backend gets dynamics at C++ start, so it keeps one config.
    _dr = cfg.dynamics.randomize if cfg.dynamics else None
    _do_dr = _dr is not None and _dr.enabled and backend == "mock"
    _base_dynamics = cfg.dynamics
    if _do_dr:
        print("domain randomization: resampling mock dynamics per episode", flush=True)

    history: List[Dict[str, float]] = []
    try:
        for ep in range(episodes):
            if _do_dr:
                # Sample from the *original* base each episode; seed per episode so
                # a given (seed, episode) is reproducible. reset() re-inits the
                # dynamics state machines from this mutated config.
                env.dp.config.dynamics = _dr.sample(
                    _base_dynamics, np.random.default_rng(base_seed + ep)
                )
            stats = _run_episode(env, app, path_agent, seed=base_seed + ep, episode=ep)
            history.append(stats)
            # Checkpoint after every episode so an interrupted run still leaves a
            # usable (latest) model rather than nothing. Buffers persist on cadence.
            _save_ckpts(save_buffers=((ep + 1) % max(1, persist_buffer_every) == 0))
            if log_every and ep % log_every == 0:
                print(
                    f"[ep {ep:3d}] QoE={stats['app_reward_mean']:+.3f} "
                    f"P={stats['path_reward_mean']:+.3f} "
                    f"bitrate={stats['bitrate_mean_kbps']:6.0f}kbps "
                    f"lat={stats['latency_mean_ms']:6.1f}ms "
                    f"loss={stats['loss_mean']:.3f}",
                    flush=True,
                )
    finally:
        dp.close()

    # Persist final checkpoints + stats (always flush buffers at the end).
    _save_ckpts(save_buffers=True)
    result = {
        "backend": backend,
        "episodes": episodes,
        "num_paths": env.num_paths,
        "path_arch": cfg.path_arch,
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
    path_agent: PathAgent,
    *,
    seed: int,
    episode: int,
) -> Dict[str, float]:
    obs = env.reset(seed=seed)
    target_kbps = obs.current_bitrate_kbps

    prev_app_obs: Optional[np.ndarray] = None
    prev_app_raw: Optional[np.ndarray] = None

    ep_app_r, n_app = 0.0, 0
    ep_p_r, n_frames = 0.0, 0
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

        # --- Path decision (every frame) ---
        p_obs = _path_obs(env, obs, target_kbps, path_agent.arch)
        split, p_raw = path_agent.select(p_obs)
        next_obs, p_r, done, info = env.step(target_kbps, split)
        p_next_obs = _path_obs(env, next_obs, target_kbps, path_agent.arch)
        # The episode horizon is a time-limit *truncation*, not a real terminal
        # (the call could continue), so always bootstrap off the next state
        # (done=False) to keep the policy horizon-agnostic.
        path_agent.store(p_obs, p_raw, p_r, p_next_obs, False)
        path_agent.update()

        ep_p_r += p_r
        n_frames += 1
        latencies.append(info.latency_ms)
        losses.append(info.loss)
        bitrates.append(target_kbps)
        obs = next_obs

    # Credit the final (partial) App window. The horizon is a truncation, not a
    # real terminal, so bootstrap off the final observation (done=False).
    if prev_app_obs is not None:
        app_r, comps = env.pop_app_window_reward()
        final_app_obs = env.build_app_obs(obs)
        app.store(prev_app_obs, prev_app_raw, app_r, final_app_obs, False)
        app.update()
        ep_app_r += app_r
        n_app += 1
        vmafs.append(comps["vmaf"])

    return {
        "episode": episode,
        "frames": n_frames,
        "app_decisions": n_app,
        "app_reward_mean": ep_app_r / max(1, n_app),
        "path_reward_mean": ep_p_r / max(1, n_frames),
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
        "use_learned_vmaf": cfg.use_learned_vmaf,
        "path_arch": cfg.path_arch,
        "dynamics": dataclasses.asdict(cfg.dynamics) if cfg.dynamics else None,
        "paths": cfg.paths,
    }
