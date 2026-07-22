#!/usr/bin/env python3
"""Multi-seed A/B evaluation harness -- the "ruler" for reward/algorithm changes.

Wraps ``run_evaluation(..., ablation=True)`` over a *set* of seeds and aggregates
the per-seed results into a cross-seed comparison, so no reward or algorithm
change is judged on a single lucky seed. This is the measurement foundation the
roadmap's Phase 1-3 experiments are graded against (ROADMAP task 0.2).

Two modes:

* **single** (default) -- one checkpoint set across N seeds. Emits per-policy
  cross-seed mean/std/CI, a ranking-stability verdict (REWARD_TUNING S5C), and
  the per-window ``qoe_components`` dump (via each run's JSON) for later reward
  calibration (Phase 2.2).
* **compare** (``--app-b/--path-b``) -- two checkpoint sets (e.g. baseline vs
  +P1) across the same seeds; prints per-policy QoE deltas with cross-seed
  variance -- the actual "is the change better?" test.

Usage:
    uv run python scripts/ab_eval.py --config configs/dynamic.yaml \
        --app runs/<run>/app.pth --path runs/<run>/path.pth \
        --seeds 1000 2000 3000 --episodes 5

    uv run python scripts/ab_eval.py --config configs/dynamic.yaml \
        --app  runs/base/app.pth --path  runs/base/path.pth \
        --app-b runs/p1/app.pth  --path-b runs/p1/path.pth \
        --seeds 1000 2000 3000 --label-a baseline --label-b p1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.config import load_config  # noqa: E402
from src.train.evaluate import run_evaluation  # noqa: E402

# Per-policy scalar metrics we aggregate across seeds. Each maps a per-run
# ``summary[policy]`` block to a single scalar (the run's central value).
_METRICS: Dict[str, callable] = {
    "qoe": lambda s: s["qoe"]["mean"],
    "vmaf": lambda s: s["vmaf"]["mean"],
    "latency_ms": lambda s: s["latency_ms"]["mean"],
    "jitter_ms": lambda s: s["jitter_ms"]["mean"],
    "loss": lambda s: s["loss"]["mean"],
    "deadline_miss_rate": lambda s: s["deadline_miss_rate"],
    "bitrate_kbps": lambda s: s["bitrate_kbps"]["mean"],
}


def _agg(values: List[float]) -> Dict[str, float]:
    """Cross-seed mean / std / 95% CI half-width (normal approx)."""
    n = len(values)
    mean = statistics.fmean(values) if n else 0.0
    std = statistics.stdev(values) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95, "n": n}


def _run_set(
    cfg,
    *,
    label: str,
    app_ckpt: str,
    path_ckpt: str,
    seeds: List[int],
    episodes: int,
    backend: str,
    out_dir: str,
    use_learned_vmaf: Optional[bool],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run one checkpoint set over all seeds; aggregate per policy per metric.

    Returns ``{policy: {metric: {mean,std,ci95,n}}}`` plus a special
    ``__perseed__`` entry ``{policy: {seed: qoe}}`` used for ranking stability.
    """
    # policy -> metric -> [per-seed scalar]
    collected: Dict[str, Dict[str, List[float]]] = {}
    perseed_qoe: Dict[str, Dict[int, float]] = {}

    for seed in seeds:
        seed_dir = os.path.join(out_dir, label, f"seed-{seed}")
        print(f"\n=== {label}: seed {seed} ===", flush=True)
        res = run_evaluation(
            cfg,
            backend=backend,
            episodes=episodes,
            seed=seed,
            app_ckpt=app_ckpt,
            path_ckpt=path_ckpt,
            out_dir=seed_dir,
            save_json=True,
            use_learned_vmaf=use_learned_vmaf,
            ablation=True,
        )
        summary = res["summary"]
        for policy, s in summary.items():
            pm = collected.setdefault(policy, {m: [] for m in _METRICS})
            for m, fn in _METRICS.items():
                pm[m].append(float(fn(s)))
            perseed_qoe.setdefault(policy, {})[seed] = float(s["qoe"]["mean"])

    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {
        policy: {m: _agg(vals) for m, vals in metrics.items()}
        for policy, metrics in collected.items()
    }
    aggregated["__perseed__"] = perseed_qoe  # type: ignore[assignment]
    return aggregated


def _ranking_stability(perseed_qoe: Dict[str, Dict[int, float]], seeds: List[int]):
    """Check the by-QoE policy ordering is stable across seeds (S5C).

    Returns (stable, verdict_lines). Flags: (a) the top policy differing across
    seeds, (b) ``learned`` not ranking first, (c) an ablation beating the full
    ``learned`` system in any seed.
    """
    lines: List[str] = []
    stable = True

    tops = []
    for seed in seeds:
        ranked = sorted(
            (p for p in perseed_qoe if seed in perseed_qoe[p]),
            key=lambda p: perseed_qoe[p][seed],
            reverse=True,
        )
        tops.append(ranked[0] if ranked else None)
    if len(set(tops)) == 1:
        lines.append(f"  top policy stable across all seeds: {tops[0]}")
    else:
        stable = False
        lines.append(f"  WARNING: top policy varies across seeds: {tops}")

    if "learned" in perseed_qoe:
        not_top = [s for s, t in zip(seeds, tops) if t != "learned"]
        if not_top:
            stable = False
            lines.append(f"  WARNING: 'learned' not top for seeds {not_top}")
        else:
            lines.append("  'learned' ranks first in every seed")
        for abl in ("app_only", "path_only_gcc"):
            if abl not in perseed_qoe:
                continue
            beaten = [
                s for s in seeds
                if s in perseed_qoe[abl]
                and perseed_qoe[abl][s] > perseed_qoe["learned"].get(s, -math.inf)
            ]
            if beaten:
                stable = False
                lines.append(
                    f"  WARNING: ablation '{abl}' beats 'learned' for seeds {beaten}"
                )
    return stable, lines


def _policy_order(agg: Dict) -> List[str]:
    """Policies present (excluding the private perseed entry), by mean QoE desc."""
    policies = [p for p in agg if p != "__perseed__"]
    return sorted(policies, key=lambda p: agg[p]["qoe"]["mean"], reverse=True)


def _print_table(label: str, agg: Dict, order: List[str]) -> None:
    print(f"\n{'=' * 74}\nCROSS-SEED SUMMARY [{label}]\n{'=' * 74}")
    print(
        f"{'policy':<16}{'QoE':>9}{'+/-95%':>9}{'VMAF':>8}"
        f"{'lat(ms)':>9}{'loss':>8}{'miss%':>8}{'kbps':>9}"
    )
    print("-" * 74)
    for p in order:
        a = agg[p]
        print(
            f"{p:<16}{a['qoe']['mean']:>9.3f}{a['qoe']['ci95']:>9.3f}"
            f"{a['vmaf']['mean']:>8.1f}{a['latency_ms']['mean']:>9.1f}"
            f"{a['loss']['mean']:>8.3f}{100 * a['deadline_miss_rate']['mean']:>8.1f}"
            f"{a['bitrate_kbps']['mean']:>9.0f}"
        )


def _write_csv(path: str, sets: List[Tuple[str, Dict]]) -> None:
    header = ["set", "policy", "n_seeds"] + [
        f"{m}_{stat}" for m in _METRICS for stat in ("mean", "std", "ci95")
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for label, agg in sets:
            for p in _policy_order(agg):
                a = agg[p]
                row = [label, p, a["qoe"]["n"]]
                for m in _METRICS:
                    row += [f"{a[m][s]:.6g}" for s in ("mean", "std", "ci95")]
                w.writerow(row)


def _print_compare(label_a: str, agg_a: Dict, label_b: str, agg_b: Dict) -> None:
    print(f"\n{'=' * 74}\nA/B DELTA  ({label_b} - {label_a})\n{'=' * 74}")
    print(f"{'policy':<16}{f'QoE[{label_a}]':>14}{f'QoE[{label_b}]':>14}{'delta':>10}{'+/-95%':>9}")
    print("-" * 74)
    shared = [p for p in _policy_order(agg_b) if p in agg_a]
    for p in shared:
        a, b = agg_a[p]["qoe"], agg_b[p]["qoe"]
        delta = b["mean"] - a["mean"]
        # Combined CI half-width of the difference of two independent means.
        ci = math.sqrt(a["ci95"] ** 2 + b["ci95"] ** 2)
        flag = "" if abs(delta) > ci else "  (n.s.)"
        print(
            f"{p:<16}{a['mean']:>14.3f}{b['mean']:>14.3f}"
            f"{delta:>+10.3f}{ci:>9.3f}{flag}"
        )
    print("\n(n.s. = delta within combined 95% CI; not significant at this seed count)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/dynamic.yaml", help="YAML config path")
    p.add_argument("--backend", choices=["mock", "ns3"], default="mock")
    p.add_argument("--app", required=True, help="App checkpoint (set A)")
    p.add_argument("--path", required=True, help="Path checkpoint (set A)")
    p.add_argument("--app-b", default=None, help="App checkpoint (set B, enables compare)")
    p.add_argument("--path-b", default=None, help="Path checkpoint (set B)")
    p.add_argument("--label-a", default="A", help="name for checkpoint set A")
    p.add_argument("--label-b", default="B", help="name for checkpoint set B")
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 2000, 3000])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--out", default=None, help="output dir (default runs/ab-<ts>)")
    p.add_argument("--learned-vmaf", action="store_true",
                   help="score QoE with the learned QoS->VMAF surrogate")
    args = p.parse_args()

    if (args.app_b is None) != (args.path_b is None):
        p.error("--app-b and --path-b must be given together (compare mode)")

    cfg = load_config(args.config)
    out_dir = args.out or os.path.join("runs", "ab-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    ulv = args.learned_vmaf or None

    sets: List[Tuple[str, Dict]] = []
    agg_a = _run_set(
        cfg, label=args.label_a, app_ckpt=args.app, path_ckpt=args.path,
        seeds=args.seeds, episodes=args.episodes, backend=args.backend,
        out_dir=out_dir, use_learned_vmaf=ulv,
    )
    sets.append((args.label_a, agg_a))
    if args.app_b is not None:
        agg_b = _run_set(
            cfg, label=args.label_b, app_ckpt=args.app_b, path_ckpt=args.path_b,
            seeds=args.seeds, episodes=args.episodes, backend=args.backend,
            out_dir=out_dir, use_learned_vmaf=ulv,
        )
        sets.append((args.label_b, agg_b))

    # Per-set tables + ranking-stability verdicts.
    verdicts: Dict[str, Dict] = {}
    for label, agg in sets:
        _print_table(label, agg, _policy_order(agg))
        stable, lines = _ranking_stability(agg["__perseed__"], args.seeds)
        print(f"\nRanking stability [{label}]: {'STABLE' if stable else 'UNSTABLE'}")
        for ln in lines:
            print(ln)
        verdicts[label] = {"stable": stable, "notes": lines}

    if len(sets) == 2:
        _print_compare(sets[0][0], sets[0][1], sets[1][0], sets[1][1])

    # Persist aggregated JSON + CSV. Strip the private perseed entry from JSON.
    json_sets = {
        label: {p: a for p, a in agg.items() if p != "__perseed__"}
        for label, agg in sets
    }
    summary_out = {
        "meta": {
            "config": args.config,
            "backend": args.backend,
            "seeds": args.seeds,
            "episodes": args.episodes,
            "learned_vmaf": bool(args.learned_vmaf),
            "compare": len(sets) == 2,
        },
        "sets": json_sets,
        "ranking_stability": verdicts,
    }
    json_path = os.path.join(out_dir, "ab_summary.json")
    csv_path = os.path.join(out_dir, "ab_summary.csv")
    with open(json_path, "w") as fh:
        json.dump(summary_out, fh, indent=2)
    _write_csv(csv_path, sets)
    print(f"\nwrote {json_path}\nwrote {csv_path}")
    print(f"per-seed evaluation_results.json (with qoe_components) under {out_dir}/<set>/seed-*/")


if __name__ == "__main__":
    main()
