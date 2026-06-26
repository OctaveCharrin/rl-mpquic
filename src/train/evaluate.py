"""
Evaluate a trained dual-agent policy against split/bitrate baselines.

Runs each policy for several episodes on the chosen backend and records, per
method:

* aggregate QoE / VMAF / latency / jitter / loss / bitrate / throughput stats,
* the **decision (inference) time** of each agent call — App-bitrate and
  Transport-split — so the learned controller's compute cost can be compared to
  the cheap heuristics,
* raw per-frame distributions (for box/CDF plots), and
* one representative per-frame time-series trace (for time-series/split plots).

Everything is dumped to ``<out_dir>/evaluation_results.json``, which
``evaluation/generate_figures.py`` turns into figures. The data plane is
backend-agnostic (mock or NS-3); the learned policy is only included when both
checkpoints are supplied, otherwise the report is baselines-only.

Baselines:

* ``even``        — split every frame equally across paths.
* ``single``      — send the whole frame on the highest-throughput path.
* ``proportional``— split in proportion to recent per-path throughput.
* ``random``      — a fresh uniform-over-simplex split every frame (seeded).
* ``learned``     — the trained App + Transport agents, acting deterministically.

Baselines use a reactive bitrate heuristic (90% of recent aggregate goodput) so
the comparison reflects scheduling quality, not a frozen bitrate.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from ..ns3env.dataplane import FrameObs
from ..ns3env.realtime_env import HierarchicalRealtimeEnv
from ..rl.app_agent import AppAgent
from ..rl.transport_agent import TransportAgent
from .config import ExperimentConfig

BitrateFn = Callable[[FrameObs], float]
SplitFn = Callable[[FrameObs, float], np.ndarray]

# Per-frame trace fields collected for the representative episode.
_TRACE_SCALARS = ("t", "latency_ms", "loss", "jitter_ms", "bitrate_kbps", "throughput_mbps")
# Distribution fields aggregated across all episodes (for box/CDF plots).
_DIST_FIELDS = (
    "qoe", "vmaf", "latency_ms", "jitter_ms", "loss", "bitrate_kbps",
    "throughput_mbps", "app_decision_ms", "transport_decision_ms",
)


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


def _timed(fn, *args):
    """Call ``fn(*args)`` and return ``(result, elapsed_ms)`` (wall-clock)."""
    t0 = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - t0) * 1000.0


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
    deadline_ms: float,
) -> Dict[str, object]:
    """One episode; returns aggregate metrics + per-frame trace + timings.

    Decision time is measured around the policy callables only (the App-bitrate
    and Transport-split calls), so for the learned policy it captures the
    observation-build + network forward pass, and for the baselines the cheap
    heuristic compute — i.e. the real cost each method pays to make a decision.
    """
    obs = env.reset(seed=seed)
    target = obs.current_bitrate_kbps

    # Per-frame trace (ordered).
    t_axis, lat, los, jit, br, thr = [], [], [], [], [], []
    split_tr: List[List[float]] = []
    pthr_tr: List[List[float]] = []
    tdec: List[float] = []          # transport decision ms (every frame)
    # Per-window (App cadence).
    win_qoe, win_vmaf, app_dec = [], [], []

    have_window = False
    done = env.is_done()
    while not done:
        if obs.app_decision_due:
            if have_window:
                r, comps = env.pop_app_window_reward()
                win_qoe.append(r)
                win_vmaf.append(comps["vmaf"])
            target, app_ms = _timed(bitrate_fn, obs)
            app_dec.append(app_ms)
            have_window = True

        split, t_ms = _timed(split_fn, obs, target)
        tdec.append(t_ms)
        next_obs, _t_r, done, info = env.step(target, split)

        t_axis.append(float(obs.clock_s))
        lat.append(info.latency_ms)
        los.append(info.loss)
        jit.append(info.jitter_ms)
        br.append(float(target))
        thr.append(float(next_obs.throughput_mbps))
        split_tr.append([float(x) for x in np.atleast_1d(split)])
        pthr_tr.append([float(x) for x in obs.path_throughput_mbps])
        obs = next_obs

    if have_window:
        r, comps = env.pop_app_window_reward()
        win_qoe.append(r)
        win_vmaf.append(comps["vmaf"])

    los_arr = np.asarray(los, dtype=np.float64)
    return {
        "trace": {
            "t": t_axis,
            "latency_ms": lat,
            "loss": los,
            "jitter_ms": jit,
            "bitrate_kbps": br,
            "throughput_mbps": thr,
            "split": split_tr,
            "path_throughput_mbps": pthr_tr,
            "transport_decision_ms": tdec,
        },
        "dist": {
            "qoe": win_qoe,
            "vmaf": win_vmaf,
            "latency_ms": lat,
            "jitter_ms": jit,
            "loss": los,
            "bitrate_kbps": br,
            "throughput_mbps": thr,
            "app_decision_ms": app_dec,
            "transport_decision_ms": tdec,
        },
        "deadline_miss_rate": float(np.mean(los_arr >= 0.999)) if los_arr.size else 0.0,
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


def _stats(x) -> Dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return {k: 0.0 for k in ("mean", "std", "p50", "p95", "min", "max")}
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def run_evaluation(
    cfg: ExperimentConfig,
    *,
    backend: str = "mock",
    episodes: int = 5,
    seed: int = 1000,
    app_ckpt: Optional[str] = None,
    transport_ckpt: Optional[str] = None,
    show_output: bool = False,
    out_dir: Optional[str] = None,
    save_json: bool = True,
    use_learned_vmaf: Optional[bool] = None,
) -> Dict[str, object]:
    """Evaluate baselines (+ learned policy if checkpoints given).

    Returns the full results dict (also written to
    ``<out_dir>/evaluation_results.json`` when ``save_json``), with keys
    ``meta`` / ``summary`` / ``distributions`` / ``traces``. Prints a table.

    ``use_learned_vmaf`` (when not None) overrides ``cfg.use_learned_vmaf`` so
    QoE is scored with the learned QoS->VMAF surrogate, matching how you trained.
    """
    if use_learned_vmaf is not None:
        cfg.use_learned_vmaf = bool(use_learned_vmaf)
    dp = cfg.make_dataplane(backend, seed=seed, show_output=show_output)
    env = HierarchicalRealtimeEnv(
        dp,
        video=cfg.video,
        weights=cfg.weights,
        episode_seconds=cfg.episode_seconds,
        vmaf_fn=cfg.build_vmaf_fn(),
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

    summary: Dict[str, Dict] = {}
    distributions: Dict[str, Dict] = {}
    traces: Dict[str, Dict] = {}
    flat: Dict[str, Dict[str, float]] = {}  # compact view for the printed table

    try:
        # Warm up each policy (stabilizes torch first-call timing).
        warm = env.reset(seed=seed)
        for bfn, sfn in policies.values():
            try:
                for _ in range(10):
                    tb = bfn(warm)
                    sfn(warm, tb)
            except Exception:  # pragma: no cover - warmup is best-effort
                pass

        for name, (bfn, sfn) in policies.items():
            rolls = [
                _rollout(env, bfn, sfn, seed=seed + e, deadline_ms=cfg.deadline_ms)
                for e in range(episodes)
            ]
            # Aggregate distributions across episodes.
            agg = {f: [] for f in _DIST_FIELDS}
            for r in rolls:
                for f in _DIST_FIELDS:
                    agg[f].extend(r["dist"][f])
            distributions[name] = {f: [float(v) for v in agg[f]] for f in _DIST_FIELDS}
            traces[name] = rolls[0]["trace"]  # representative episode

            summary[name] = {
                "qoe": _stats(agg["qoe"]),
                "vmaf": _stats(agg["vmaf"]),
                "latency_ms": _stats(agg["latency_ms"]),
                "jitter_ms": _stats(agg["jitter_ms"]),
                "loss": _stats(agg["loss"]),
                "bitrate_kbps": _stats(agg["bitrate_kbps"]),
                "throughput_mbps": _stats(agg["throughput_mbps"]),
                "app_decision_ms": _stats(agg["app_decision_ms"]),
                "transport_decision_ms": _stats(agg["transport_decision_ms"]),
                "deadline_miss_rate": float(
                    np.mean([r["deadline_miss_rate"] for r in rolls])
                ),
                "frames": len(agg["latency_ms"]),
                "app_decisions": len(agg["qoe"]),
                "episodes": episodes,
            }
            s = summary[name]
            flat[name] = {
                "qoe": s["qoe"]["mean"],
                "vmaf": s["vmaf"]["mean"],
                "latency_ms": s["latency_ms"]["mean"],
                "loss": s["loss"]["mean"],
                "bitrate_kbps": s["bitrate_kbps"]["mean"],
                "decision_ms": s["transport_decision_ms"]["p50"],
            }
    finally:
        dp.close()

    results: Dict[str, object] = {
        "meta": {
            "backend": backend,
            "episodes": episodes,
            "seed": seed,
            "num_paths": env.num_paths,
            "fps": cfg.fps,
            "episode_seconds": cfg.episode_seconds,
            "app_period_s": cfg.app_period_s,
            "deadline_ms": cfg.deadline_ms,
            "bitrate_kbps": [cfg.video.min_bitrate_kbps, cfg.video.max_bitrate_kbps],
            "reward_weights": cfg.weights.to_dict(),
            "paths": cfg.paths,
            "has_learned": "learned" in policies,
        },
        "summary": summary,
        "distributions": distributions,
        "traces": traces,
    }

    if save_json:
        out_dir = out_dir or os.path.join(cfg.out_dir, "eval-" + time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "evaluation_results.json")
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=2)
        results["meta"]["out_dir"] = out_dir
        print(f"saved evaluation results to {json_path}")

    print(f"\n{'policy':<14}{'QoE':>8}{'VMAF':>8}{'lat(ms)':>10}{'loss':>8}{'kbps':>9}{'decide(ms)':>12}")
    print("-" * 69)
    for name, r in flat.items():
        print(
            f"{name:<14}{r['qoe']:>8.3f}{r['vmaf']:>8.1f}{r['latency_ms']:>10.1f}"
            f"{r['loss']:>8.3f}{r['bitrate_kbps']:>9.0f}{r['decision_ms']:>12.4f}"
        )
    return results
