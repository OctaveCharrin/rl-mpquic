#!/usr/bin/env python3
"""Calibrate the QoE reward weights (b, c, d) against the ITU-T G.1070 oracle.

Implements the protocol in ``docs/REWARD_TUNING.md`` §5B: over a corpus of
``(bitrate, latency, jitter, loss)`` conditions, correlate the linear reward
``R = a·VMAF/100 - b·lat - c·jit - d·loss`` (as computed by
:func:`src.ns3env.qoe.compute_qoe_reward`) against the G.1070 MOS oracle
(:mod:`src.ns3env.g1070`), and pick the ``(b, d)`` maximizing rank/linear
correlation. Following the standards anchors (§5A), the quality weight ``a`` and
the normalizers are fixed; and — since neither G.1070 nor the shipped learned
surrogate resolves jitter — ``c`` is **pinned equal to ``b``** (the 1:1
post-normalization ratio argued in §3), not fit independently.

The script is **non-destructive**: it writes a ``calibration_report.json`` and
prints a ready-to-paste ``reward:`` YAML block. It never edits a config.

Corpus source:
  * default — a **synthetic** grid sweep over the four axes (self-contained,
    deterministic, guarantees axis coverage);
  * ``--runs DIR`` — additionally ingest per-window ``qoe_components`` from any
    ``evaluation_results.json`` under DIR (real visited distribution). Effective
    bitrate is reconstructed from the logged ``vmaf`` by inverting the log curve.

Usage:
    uv run python scripts/calibrate_reward.py [--out calibration_report.json]
        [--runs runs/ab_out] [--video-preset h264_bp_vga_6in]
        [--latency-norm 200] [--jitter-norm 50]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ns3env.g1070 import G1070Config, G1070Oracle  # noqa: E402
from src.ns3env.qoe import QoEWeights, compute_qoe_reward, vmaf_for_kbps  # noqa: E402

# Grid searched for the free weights. c is pinned = b; d filtered to d >= b.
_B_GRID = (0.25, 0.5, 0.75, 1.0)
_D_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)

Sample = Dict[str, float]  # {bitrate_kbps, latency_ms, jitter_ms, loss}


# --------------------------------------------------------------------------- #
# Corpus construction.
# --------------------------------------------------------------------------- #


def synthetic_corpus(
    *,
    bitrates=(300, 600, 1000, 1500, 2500, 4000, 6000),
    latencies=(0, 30, 60, 120, 200, 350, 550),
    jitters=(0, 10, 30, 60, 120),
    losses=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3),
) -> List[Sample]:
    """Full-factorial sweep of the four QoE axes."""
    corpus: List[Sample] = []
    for b in bitrates:
        for lat in latencies:
            for jit in jitters:
                for loss in losses:
                    corpus.append(
                        {
                            "bitrate_kbps": float(b),
                            "latency_ms": float(lat),
                            "jitter_ms": float(jit),
                            "loss": float(loss),
                        }
                    )
    return corpus


def _invert_vmaf(vmaf: float, lo: float = 1.0, hi: float = 1e5) -> float:
    """Reconstruct a bitrate (kbps) from a logged VMAF by bisecting the log curve."""
    v = float(vmaf)
    # Clip targets outside the achievable range to the endpoints.
    if v <= vmaf_for_kbps(lo):
        return lo
    if v >= vmaf_for_kbps(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if vmaf_for_kbps(mid) < v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def corpus_from_runs(runs_dir: str) -> List[Sample]:
    """Ingest per-window components from evaluation_results.json files under DIR.

    ``qoe_components`` logs ``vmaf`` (not raw bitrate), so bitrate is reconstructed
    by inverting the log VMAF curve — an approximation, documented in the module.
    """
    corpus: List[Sample] = []
    files = glob.glob(os.path.join(runs_dir, "**", "evaluation_results.json"), recursive=True)
    for fp in files:
        try:
            with open(fp) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        comps = data.get("qoe_components", {})
        for _policy, fields in comps.items():
            vmafs = fields.get("vmaf", [])
            lats = fields.get("latency_ms", [])
            jits = fields.get("jitter_ms", [])
            losses = fields.get("loss", [])
            n = min(len(vmafs), len(lats), len(jits), len(losses))
            for i in range(n):
                corpus.append(
                    {
                        "bitrate_kbps": _invert_vmaf(vmafs[i]),
                        "latency_ms": float(lats[i]),
                        "jitter_ms": float(jits[i]),
                        "loss": float(losses[i]),
                    }
                )
    return corpus


# --------------------------------------------------------------------------- #
# Correlation helpers (numpy-only; no scipy dependency).
# --------------------------------------------------------------------------- #


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return _pearson(rx, ry)


# --------------------------------------------------------------------------- #
# Calibration.
# --------------------------------------------------------------------------- #


def calibrate(
    corpus: List[Sample],
    oracle: G1070Oracle,
    *,
    latency_norm_ms: float = 200.0,
    jitter_norm_ms: float = 50.0,
    a_quality: float = 1.0,
) -> Dict:
    """Grid-search (b, d) with c pinned = b; rank by mean(Pearson, Spearman) vs MOS."""
    mos = np.array(
        [
            oracle.mos(
                bitrate_kbps=s["bitrate_kbps"],
                latency_ms=s["latency_ms"],
                loss=s["loss"],
                jitter_ms=s["jitter_ms"],
            )
            for s in corpus
        ]
    )

    grid: List[Dict] = []
    for b in _B_GRID:
        for d in _D_GRID:
            if d < b:  # keep loss dominant: d >= b (= c)
                continue
            w = QoEWeights(
                a_quality=a_quality, b_latency=b, c_jitter=b, d_loss=d,
                latency_norm_ms=latency_norm_ms, jitter_norm_ms=jitter_norm_ms,
            )
            r = np.array(
                [
                    compute_qoe_reward(
                        bitrate_kbps=s["bitrate_kbps"], latency_ms=s["latency_ms"],
                        jitter_ms=s["jitter_ms"], loss=s["loss"], weights=w,
                    )
                    for s in corpus
                ]
            )
            pear = _pearson(r, mos)
            spear = _spearman(r, mos)
            grid.append(
                {
                    "b": b, "c": b, "d": d,
                    "pearson": round(pear, 5),
                    "spearman": round(spear, 5),
                    "score": round(0.5 * (pear + spear), 5),
                }
            )

    grid.sort(key=lambda g: g["score"], reverse=True)
    best = grid[0]
    return {
        "n_samples": len(corpus),
        "anchors": {
            "a_quality": a_quality,
            "latency_norm_ms": latency_norm_ms,
            "jitter_norm_ms": jitter_norm_ms,
        },
        "pinned": "c = b (1:1 standards ratio; jitter unresolved by oracle)",
        "constraint": "d >= b = c (loss dominant)",
        "best": best,
        "grid": grid,
    }


def _reward_block(report: Dict, video_preset: str) -> str:
    b = report["best"]
    a = report["anchors"]
    return (
        "reward:\n"
        f"  a_quality: {a['a_quality']}\n"
        f"  b_latency: {b['b']}\n"
        f"  c_jitter: {b['c']}\n"
        f"  d_loss: {b['d']}\n"
        f"  latency_norm_ms: {a['latency_norm_ms']}\n"
        f"  jitter_norm_ms: {a['jitter_norm_ms']}\n"
        f"  # calibrated vs ITU-T G.1070 ({video_preset}); "
        f"score={b['score']} (pearson={b['pearson']}, spearman={b['spearman']})\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="calibration_report.json")
    ap.add_argument("--runs", default=None, help="dir to ingest evaluation_results.json from")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the synthetic corpus")
    ap.add_argument("--video-preset", default="h264_bp_vga_6in")
    ap.add_argument("--integration-preset", default="4.2in")
    ap.add_argument("--latency-norm", type=float, default=200.0)
    ap.add_argument("--jitter-norm", type=float, default=50.0)
    args = ap.parse_args()

    corpus: List[Sample] = []
    if not args.no_synthetic:
        corpus += synthetic_corpus()
    if args.runs:
        real = corpus_from_runs(args.runs)
        print(f"ingested {len(real)} samples from {args.runs}")
        corpus += real
    if not corpus:
        print("empty corpus (use synthetic or --runs)", file=sys.stderr)
        return 2

    oracle = G1070Oracle(
        G1070Config(video_preset=args.video_preset, integration_preset=args.integration_preset)
    )
    report = calibrate(
        corpus, oracle,
        latency_norm_ms=args.latency_norm, jitter_norm_ms=args.jitter_norm,
    )
    report["video_preset"] = args.video_preset
    report["integration_preset"] = args.integration_preset

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    best = report["best"]
    print(f"\ncorpus: {report['n_samples']} samples   oracle: G.1070 {args.video_preset}")
    print(f"best: b=c={best['b']}  d={best['d']}  "
          f"(score={best['score']}, pearson={best['pearson']}, spearman={best['spearman']})")
    print(f"report written to {args.out}\n")
    print(_reward_block(report, args.video_preset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
