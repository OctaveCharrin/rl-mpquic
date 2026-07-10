#!/usr/bin/env python3
"""
Generate an exhaustive figure set comparing the hierarchical RL controller to
the scheduling baselines, from an ``evaluation_results.json`` produced by
``evaluate.py`` (i.e. ``src/train/evaluate.py:run_evaluation``).

Mirrors the spirit of ``scion-dqn-sim/evaluation/06_generate_figures.py``:
LNCS-style serif figures saved as PNG. Backend-agnostic — it only reads the
JSON, so it works for mock or NS-3 runs alike, and degrades gracefully when no
``learned`` policy is present (baselines-only).

Usage:
    python evaluation/generate_figures.py <run_dir>
    python evaluation/generate_figures.py            # newest runs/eval-* dir

Figures (written to <run_dir>/figures/):
    figure1_qoe.png            App-agent QoE per method (mean +/- std)
    figure2_metric_panels.png  VMAF / latency / loss panels
    figure3_qoe_distribution.png  QoE distribution (box plot)
    figure4_decision_time.png  Per-method decision (inference) time
    figure5_quality_vs_cost.png   QoE vs decision-time Pareto scatter
    figure6_timeseries.png     Representative episode time-series
    figure7_split_behavior.png Learned per-path split over time
    figure8_latency_cdf.png    Latency CDF + deadline-miss rate
    figure9_radar.png          Normalized multi-metric radar
    figure10_path_metrics.png  Per-path throughput/sRTT/loss/split over time
    figure11_app_bitrate.png   App-agent bitrate decisions, distribution, quality
    figure12_ablation.png      Single-agent ablation (only with --ablation runs)
    summary_table.csv          Machine-readable summary
"""

import csv
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
except ImportError:  # pragma: no cover
    sys.exit(
        "matplotlib is required for figures. Install it with:\n"
        "    uv sync --extra viz       (or)   pip install matplotlib"
    )


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

rcParams["font.family"] = "serif"
# Times New Roman if available, otherwise matplotlib falls back to a serif font.
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
rcParams["font.size"] = 10
rcParams["axes.labelsize"] = 10
rcParams["axes.titlesize"] = 11
rcParams["xtick.labelsize"] = 9
rcParams["ytick.labelsize"] = 9
rcParams["legend.fontsize"] = 9
rcParams["figure.titlesize"] = 12
rcParams["axes.axisbelow"] = True

COLUMN_WIDTH = 3.5
FULL_WIDTH = 7.0

DISPLAY = {
    "learned": "Hierarchical RL (Ours)",
    "app_only": "App Agent Only",
    "path_only_gcc": "Path Agent Only (GCC bitrate)",
    "even": "Even Split",
    "single": "Single Best",
    "proportional": "Proportional",
    "webrtc": "WebRTC (GCC)",
}
COLORS = {
    "learned": "#1f77b4",
    "app_only": "#17becf",
    "path_only_gcc": "#5b2c9f",
    "even": "#ff7f0e",
    "single": "#2ca02c",
    "proportional": "#d62728",
    "webrtc": "#8c564b",
}
# Ablation variants (one learned agent disabled), in display order. ``path_only_gcc``
# runs the learned split under a WebRTC GCC bitrate driver (which, unlike the
# reactive goodput heuristic, does not spiral to the bitrate floor); its matched
# reference is the standalone ``webrtc`` baseline (picked as the ablation panel's
# hatched reference).
ABLATION_ORDER = ("learned", "app_only", "path_only_gcc")
# Which ablation entries actually involve a learned agent (drawn solid; the pure
# heuristic references in the ablation panel are hatched).
_LEARNED_ABLATIONS = frozenset({"learned", "app_only", "path_only_gcc"})
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def disp(method: str) -> str:
    return DISPLAY.get(method, method.replace("_", " ").title())


def col(method: str, _cache={}) -> str:
    if method in COLORS:
        return COLORS[method]
    if method not in _cache:
        _cache[method] = _FALLBACK_COLORS[len(_cache) % len(_FALLBACK_COLORS)]
    return _cache[method]


def _is_ours(method: str) -> bool:
    return method == "learned"


def _bar_edges(methods):
    """Heavier edge on 'our' method so it stands out in bar charts."""
    return [
        {"edgecolor": "black", "linewidth": 1.6 if _is_ours(m) else 0.6}
        for m in methods
    ]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def _resolve_run_dir(argv) -> str:
    if len(argv) > 1:
        return argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    runs = os.path.join(os.path.dirname(here), "runs")
    cands = []
    for root in (".", runs):
        if os.path.isdir(root):
            cands += [
                os.path.join(root, d)
                for d in os.listdir(root)
                if d.startswith("eval-")
                and os.path.exists(os.path.join(root, d, "evaluation_results.json"))
            ]
    if not cands:
        sys.exit("no run dir given and no runs/eval-* with evaluation_results.json found")
    return sorted(cands)[-1]


def _save(fig, fig_dir, name):
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  - {name}.png")


def _rolling(y, w):
    y = np.asarray(y, dtype=float)
    if w <= 1 or y.size < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def _episode_time(t):
    """Zero-base a trace time axis to its episode start.

    Older evaluation JSONs recorded absolute sim time, and the NS-3 clock is
    monotonic across episodes/methods, so each method's trace can start at a
    different offset. Subtracting the first sample realigns every trace at t=0
    (and is a harmless no-op for traces that already start at 0)."""
    t = np.asarray(t, dtype=float)
    return t - t[0] if t.size else t


def _norm_higher_better(vals):
    a = np.asarray(vals, dtype=float)
    lo, hi = a.min(), a.max()
    return np.full_like(a, 0.5) if hi - lo < 1e-12 else (a - lo) / (hi - lo)


def _norm_lower_better(vals):
    return 1.0 - _norm_higher_better(vals)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_qoe_bar(summary, order, fig_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3))
    means = [summary[m]["qoe"]["mean"] for m in order]
    stds = [summary[m]["qoe"]["std"] for m in order]
    bars = ax.bar(
        range(len(order)), means, yerr=stds, capsize=3,
        color=[col(m) for m in order],
    )
    for bar, edge in zip(bars, _bar_edges(order)):
        bar.set(**edge)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([disp(m) for m in order], rotation=30, ha="right")
    ax.set_ylabel("App QoE (reward / window)")
    ax.set_title("Quality of Experience by Method")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    _save(fig, fig_dir, "figure1_qoe")


def fig_metric_panels(summary, order, deadline_ms, fig_dir):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(FULL_WIDTH, 3))
    x = np.arange(len(order))
    colors = [col(m) for m in order]
    edges = _bar_edges(order)

    # (a) VMAF
    vmaf = [summary[m]["vmaf"]["mean"] for m in order]
    b = a1.bar(x, vmaf, color=colors)
    for bar, e in zip(b, edges):
        bar.set(**e)
    a1.set_ylabel("VMAF (0-100)")
    a1.set_title("(a) Perceptual Quality")
    a1.set_ylim(0, 100)

    # (b) latency p50 / p95
    w = 0.38
    p50 = [summary[m]["latency_ms"]["p50"] for m in order]
    p95 = [summary[m]["latency_ms"]["p95"] for m in order]
    a2.bar(x - w / 2, p50, w, label="p50", color=colors, alpha=0.9)
    a2.bar(x + w / 2, p95, w, label="p95", color=colors, alpha=0.5)
    a2.axhline(deadline_ms, ls="--", lw=1.0, color="black", label=f"deadline {deadline_ms:.0f} ms")
    a2.set_ylabel("Latency (ms)")
    a2.set_title("(b) Delivery Latency")
    a2.legend(fontsize=7)

    # (c) loss + deadline-miss
    loss = [100.0 * summary[m]["loss"]["mean"] for m in order]
    miss = [100.0 * summary[m]["deadline_miss_rate"] for m in order]
    a3.bar(x - w / 2, loss, w, label="loss %", color=colors, alpha=0.9)
    a3.bar(x + w / 2, miss, w, label="deadline-miss %", color=colors, alpha=0.5)
    a3.set_ylabel("Percent (%)")
    a3.set_title("(c) Loss / Deadline Misses")
    a3.legend(fontsize=7)

    for ax in (a1, a2, a3):
        ax.set_xticks(x)
        ax.set_xticklabels([disp(m) for m in order], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
    _save(fig, fig_dir, "figure2_metric_panels")


def fig_qoe_distribution(distributions, order, fig_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3))
    data = [np.asarray(distributions[m]["qoe"], dtype=float) for m in order]
    data = [d if d.size else np.array([0.0]) for d in data]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(col(m))
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([disp(m) for m in order], rotation=30, ha="right")
    ax.set_ylabel("App QoE (per window)")
    ax.set_title("QoE Distribution")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, fig_dir, "figure3_qoe_distribution")


def fig_decision_time(summary, distributions, order, fig_dir):
    """The requested decision-time-per-method comparison."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3))
    x = np.arange(len(order))
    w = 0.38

    # (a) mean app vs path decision time (log scale: RL >> heuristics).
    app = [summary[m]["app_decision_ms"]["mean"] for m in order]
    tra = [summary[m]["path_decision_ms"]["mean"] for m in order]
    eps = 1e-4
    a1.bar(x - w / 2, np.maximum(app, eps), w, label="App (bitrate)", color="#4C72B0")
    a1.bar(x + w / 2, np.maximum(tra, eps), w, label="Path (split)", color="#DD8452")
    a1.set_yscale("log")
    a1.set_ylabel("Decision time (ms, log)")
    a1.set_title("(a) Mean Inference Time per Decision")
    a1.legend(fontsize=8)
    a1.set_xticks(x)
    a1.set_xticklabels([disp(m) for m in order], rotation=35, ha="right")
    a1.grid(axis="y", alpha=0.3, which="both")
    for i, v in enumerate(tra):
        a1.text(i + w / 2, max(v, eps), f"{v:.3g}", ha="center", va="bottom", fontsize=7)

    # (b) per-frame path decision time distribution.
    data = [np.maximum(np.asarray(distributions[m]["path_decision_ms"], float), eps)
            for m in order]
    data = [d if d.size else np.array([eps]) for d in data]
    bp = a2.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(col(m))
        patch.set_alpha(0.7)
    a2.set_yscale("log")
    a2.set_ylabel("Path decision time (ms, log)")
    a2.set_title("(b) Per-Frame Split-Decision Time")
    a2.set_xticks(range(1, len(order) + 1))
    a2.set_xticklabels([disp(m) for m in order], rotation=35, ha="right")
    a2.grid(axis="y", alpha=0.3, which="both")
    _save(fig, fig_dir, "figure4_decision_time")


def fig_quality_vs_cost(summary, order, fig_dir):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3))
    for m in order:
        x = max(summary[m]["path_decision_ms"]["p50"], 1e-4)
        y = summary[m]["qoe"]["mean"]
        ax.scatter(x, y, s=90 if _is_ours(m) else 60, color=col(m),
                   edgecolor="black", linewidth=1.4 if _is_ours(m) else 0.6,
                   marker="*" if _is_ours(m) else "o", zorder=3, label=disp(m))
        ax.annotate(disp(m), (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Decision time per frame (ms, log) -> cheaper left")
    ax.set_ylabel("App QoE -> better up")
    ax.set_title("Quality vs Compute Cost")
    ax.grid(alpha=0.3, which="both")
    _save(fig, fig_dir, "figure5_quality_vs_cost")


def fig_timeseries(traces, meta, order, fig_dir):
    fps = int(round(meta.get("fps", 30)))
    deadline = meta.get("deadline_ms", 180.0)
    panels = [
        ("bitrate_kbps", "Target bitrate (kbps)", False),
        ("throughput_mbps", "Throughput (Mbps)", False),
        ("latency_ms", "Latency (ms)", True),
        ("loss", "Loss (rolling)", False),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(FULL_WIDTH, 7), sharex=True)
    for ax, (field, ylabel, mark_deadline) in zip(axes, panels):
        for m in order:
            tr = traces.get(m, {})
            t = _episode_time(tr.get("t", []))
            y = np.asarray(tr.get(field, []), dtype=float)
            if t.size == 0 or y.size == 0:
                continue
            if field in ("latency_ms", "loss"):
                y = _rolling(y, fps)
            ax.plot(t, y, color=col(m), lw=1.6 if _is_ours(m) else 1.0,
                    alpha=0.95 if _is_ours(m) else 0.7, label=disp(m))
        if mark_deadline:
            ax.axhline(deadline, ls="--", lw=1.0, color="black")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].set_title("Representative Episode Time-Series")
    axes[0].legend(ncol=len(order), fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    _save(fig, fig_dir, "figure6_timeseries")


def fig_split_behavior(traces, meta, order, fig_dir):
    # Prefer the learned policy; otherwise the first method with a split trace.
    pick = "learned" if "learned" in traces else (order[0] if order else None)
    if pick is None:
        return
    tr = traces.get(pick, {})
    t = _episode_time(tr.get("t", []))
    split = np.asarray(tr.get("split", []), dtype=float)
    if t.size == 0 or split.size == 0:
        return
    if split.ndim == 1:
        split = split.reshape(-1, 1)
    n_paths = split.shape[1]
    paths = meta.get("paths", [])
    labels = [
        f"Path {i}" + (f" ({paths[i].get('rate', '')})" if i < len(paths) else "")
        for i in range(n_paths)
    ]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3))
    ax.stackplot(t, *[split[:, i] for i in range(n_paths)], labels=labels, alpha=0.85)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Traffic split fraction")
    ax.set_title(f"Learned Path-Split Behavior — {disp(pick)}")
    ax.legend(ncol=n_paths, fontsize=7, loc="upper right")
    _save(fig, fig_dir, "figure7_split_behavior")


def _path_label(meta, i):
    paths = meta.get("paths", [])
    label = f"Path {i}"
    if i < len(paths) and isinstance(paths[i], dict):
        rate = paths[i].get("rate", "")
        if rate:
            label += f" ({rate})"
    return label


def fig_path_metrics(traces, meta, order, fig_dir):
    """Per-path network state + the policy's allocation over time, so one can see
    how the paths evolve and how the Path agent reacts. Uses the learned
    policy's representative episode (else the first method with per-path traces)."""
    pick = "learned" if "learned" in traces else (order[0] if order else None)
    if pick is None:
        return
    tr = traces.get(pick, {})
    t = _episode_time(tr.get("t", []))
    if t.size == 0:
        return

    def _as2d(field):
        arr = np.asarray(tr.get(field, []), dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    pthr = _as2d("path_throughput_mbps")
    psrtt = _as2d("path_srtt_ms")
    ploss = _as2d("path_loss")
    split = _as2d("split")
    candidates = [a for a in (pthr, psrtt, ploss, split) if a is not None]
    if not candidates:
        return
    n_paths = max(a.shape[1] for a in candidates)
    fps = int(round(meta.get("fps", 30)))
    path_colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_paths, 1)))

    panels = [
        (pthr, "Per-path goodput (Mbps)", "(a) Path Throughput", False),
        (psrtt, "Per-path sRTT (ms)", "(b) Path Latency", True),
        (ploss, "Per-path loss", "(c) Path Loss", True),
        (split, "Traffic split fraction", "(d) Allocated Split", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.5), sharex=True)
    axes = axes.ravel()
    for ax, (data, ylabel, title, smooth) in zip(axes, panels):
        if data is not None:
            for i in range(data.shape[1]):
                y = data[:, i]
                if smooth:
                    y = _rolling(y, fps)
                ax.plot(t[: len(y)], y, color=path_colors[i % len(path_colors)],
                        lw=1.3, label=_path_label(meta, i))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[3].set_ylim(0, 1)
    axes[2].set_xlabel("Time (s)")
    axes[3].set_xlabel("Time (s)")
    # Attach the path-color legend to whichever panel actually has lines: the
    # throughput panel may be empty for a partially-populated trace, and an
    # empty axes[0].legend() would silently drop the only path-color key.
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper right")
            break
    fig.suptitle(f"Per-Path Network State & Allocation — {disp(pick)}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, fig_dir, "figure10_path_metrics")


def fig_app_bitrate(traces, distributions, summary, meta, order, fig_dir):
    """How the App agent picks the target bitrate: its trajectory vs available
    goodput, the spread of chosen bitrates, and the quality it buys."""
    # Wide top row for the time series (it needs the width + a legend), with the
    # distribution / quality panels sharing the bottom row.
    fig = plt.figure(figsize=(FULL_WIDTH, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.45, wspace=0.3)
    a1 = fig.add_subplot(gs[0, :])
    a2 = fig.add_subplot(gs[1, 0])
    a3 = fig.add_subplot(gs[1, 1])
    fps = int(round(meta.get("fps", 30)))
    bounds = meta.get("bitrate_kbps", None)

    # (a) Bitrate decisions over time (step: bitrate persists across the window),
    #     over a light envelope of the achieved goodput for context.
    pick = "learned" if "learned" in traces else (order[0] if order else None)
    if pick:
        ptr = traces.get(pick, {})
        pt = _episode_time(ptr.get("t", []))
        pthr = np.asarray(ptr.get("throughput_mbps", []), dtype=float) * 1000.0
        if pt.size and pthr.size:
            a1.fill_between(pt, 0.0, _rolling(pthr, fps), color="gray", alpha=0.15,
                            zorder=0, label=f"{disp(pick)} goodput")
    for m in order:
        tr = traces.get(m, {})
        t = _episode_time(tr.get("t", []))
        br = np.asarray(tr.get("bitrate_kbps", []), dtype=float)
        if t.size == 0 or br.size == 0:
            continue
        a1.step(t, br, where="post", color=col(m), lw=1.8 if _is_ours(m) else 1.0,
                alpha=0.95 if _is_ours(m) else 0.7, label=disp(m))
    if bounds:
        a1.axhline(bounds[0], ls=":", lw=0.8, color="black")
        a1.axhline(bounds[1], ls=":", lw=0.8, color="black")
    a1.set_xlabel("Time (s)")
    a1.set_ylabel("Target bitrate (kbps)")
    a1.set_title("(a) App Bitrate Decisions over Time")
    a1.legend(fontsize=7, ncol=3, loc="upper center")
    a1.grid(alpha=0.3)

    # (b) Distribution of chosen bitrates per method.
    data = [np.asarray(distributions[m]["bitrate_kbps"], dtype=float) for m in order]
    data = [d if d.size else np.array([0.0]) for d in data]
    bp = a2.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(col(m))
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    a2.set_xticks(range(1, len(order) + 1))
    a2.set_xticklabels([disp(m) for m in order], rotation=35, ha="right")
    a2.set_ylabel("Target bitrate (kbps)")
    a2.set_title("(b) Bitrate Distribution")
    a2.grid(axis="y", alpha=0.3)

    # (c) Quality return on the chosen bitrate (mean VMAF vs mean bitrate).
    for m in order:
        x = summary[m]["bitrate_kbps"]["mean"]
        y = summary[m]["vmaf"]["mean"]
        a3.scatter(x, y, s=110 if _is_ours(m) else 60, color=col(m),
                   edgecolor="black", linewidth=1.4 if _is_ours(m) else 0.6,
                   marker="*" if _is_ours(m) else "o", zorder=3)
        a3.annotate(disp(m), (x, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=7)
    a3.set_xlabel("Mean target bitrate (kbps)")
    a3.set_ylabel("Mean VMAF")
    a3.set_title("(c) Quality vs Bitrate")
    a3.grid(alpha=0.3)
    _save(fig, fig_dir, "figure11_app_bitrate")


def fig_ablation(summary, fig_dir):
    """Single-agent ablation: the contribution of each learned agent.

    Compares the full hierarchical controller to variants with one agent
    disabled (``app_only`` = Path off, ``path_only_gcc`` = App off), with
    the best heuristic baseline (hatched) for reference. Skipped unless at least
    one ablation variant is present in the results."""
    methods = [m for m in ABLATION_ORDER if m in summary]
    if "learned" not in summary or len(methods) < 2:
        return
    baselines = [m for m in summary if m not in ABLATION_ORDER]
    ref = max(baselines, key=lambda m: summary[m]["qoe"]["mean"]) if baselines else None
    show = methods + ([ref] if ref else [])

    panels = [
        ("QoE", lambda m: summary[m]["qoe"]["mean"],
         lambda m: summary[m]["qoe"]["std"], "App QoE (per window)", "{:.3f}"),
        ("Perceptual Quality", lambda m: summary[m]["vmaf"]["mean"],
         None, "VMAF (0-100)", "{:.1f}"),
        ("Deadline Misses", lambda m: 100.0 * summary[m]["deadline_miss_rate"],
         None, "Deadline-miss (%)", "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 3.4))
    x = np.arange(len(show))
    for ax, (title, val, err, ylabel, fmt) in zip(axes, panels):
        vals = [val(m) for m in show]
        errs = [err(m) for m in show] if err else None
        bars = ax.bar(x, vals, yerr=errs, capsize=3, color=[col(m) for m in show])
        for bar, m, edge in zip(bars, show, _bar_edges(show)):
            bar.set(**edge, hatch="" if m in _LEARNED_ABLATIONS else "//")
        ax.set_xticks(x)
        ax.set_xticklabels([disp(m) for m in show], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
    axes[1].set_ylim(0, 100)
    fig.suptitle("Single-Agent Ablation — Contribution of Each Learned Agent")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, fig_dir, "figure12_ablation")


def fig_latency_cdf(summary, distributions, order, meta, fig_dir):
    deadline = meta.get("deadline_ms", 180.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3))

    for m in order:
        lat = np.sort(np.asarray(distributions[m]["latency_ms"], dtype=float))
        if lat.size == 0:
            continue
        cdf = np.arange(1, lat.size + 1) / lat.size
        a1.plot(lat, cdf, color=col(m), lw=1.8 if _is_ours(m) else 1.1,
                label=disp(m))
    a1.axvline(deadline, ls="--", lw=1.0, color="black", label=f"deadline {deadline:.0f} ms")
    a1.set_xlabel("Latency (ms)")
    a1.set_ylabel("CDF")
    a1.set_title("(a) Latency CDF")
    a1.set_ylim(0, 1)
    a1.legend(fontsize=7)
    a1.grid(alpha=0.3)

    x = np.arange(len(order))
    miss = [100.0 * summary[m]["deadline_miss_rate"] for m in order]
    bars = a2.bar(x, miss, color=[col(m) for m in order])
    for bar, e in zip(bars, _bar_edges(order)):
        bar.set(**e)
    a2.set_xticks(x)
    a2.set_xticklabels([disp(m) for m in order], rotation=35, ha="right")
    a2.set_ylabel("Deadline-miss rate (%)")
    a2.set_title("(b) Frames Past Deadline")
    a2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(miss):
        a2.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    _save(fig, fig_dir, "figure8_latency_cdf")


def fig_radar(summary, order, fig_dir):
    axes_labels = ["QoE", "VMAF", "Low\nLatency", "Low\nLoss", "Fast\nDecision"]
    qoe = _norm_higher_better([summary[m]["qoe"]["mean"] for m in order])
    vmaf = _norm_higher_better([summary[m]["vmaf"]["mean"] for m in order])
    lat = _norm_lower_better([summary[m]["latency_ms"]["mean"] for m in order])
    loss = _norm_lower_better([summary[m]["loss"]["mean"] for m in order])
    dec = _norm_lower_better(
        [np.log10(max(summary[m]["path_decision_ms"]["p50"], 1e-4)) for m in order]
    )
    metrics = np.vstack([qoe, vmaf, lat, loss, dec]).T  # (methods, axes)

    angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1, COLUMN_WIDTH + 1),
                           subplot_kw=dict(polar=True))
    for i, m in enumerate(order):
        vals = metrics[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=col(m), lw=2.0 if _is_ours(m) else 1.2,
                label=disp(m))
        ax.fill(angles, vals, color=col(m), alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.set_ylim(0, 1)
    ax.set_title("Normalized Multi-Metric Comparison\n(outer = better)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    _save(fig, fig_dir, "figure9_radar")


def write_summary_csv(summary, order, path):
    cols = [
        ("method", lambda s: None),
        ("qoe_mean", lambda s: s["qoe"]["mean"]),
        ("qoe_std", lambda s: s["qoe"]["std"]),
        ("vmaf_mean", lambda s: s["vmaf"]["mean"]),
        ("latency_mean_ms", lambda s: s["latency_ms"]["mean"]),
        ("latency_p95_ms", lambda s: s["latency_ms"]["p95"]),
        ("jitter_mean_ms", lambda s: s["jitter_ms"]["mean"]),
        ("loss_mean", lambda s: s["loss"]["mean"]),
        ("deadline_miss_rate", lambda s: s["deadline_miss_rate"]),
        ("bitrate_mean_kbps", lambda s: s["bitrate_kbps"]["mean"]),
        ("throughput_mean_mbps", lambda s: s["throughput_mbps"]["mean"]),
        ("app_decision_ms_p50", lambda s: s["app_decision_ms"]["p50"]),
        ("path_decision_ms_p50", lambda s: s["path_decision_ms"]["p50"]),
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([c for c, _ in cols])
        for m in order:
            s = summary[m]
            row = [m] + [f"{fn(s):.6g}" for _, fn in cols[1:]]
            w.writerow(row)


def print_table(summary, order):
    print("\n" + "=" * 78)
    print("PERFORMANCE COMPARISON")
    print("=" * 78)
    print(f"{'Method':<24}{'QoE':>9}{'VMAF':>8}{'Lat p50':>9}{'Miss%':>8}{'Decide(ms)':>12}")
    print("-" * 78)
    for m in order:
        s = summary[m]
        print(
            f"{disp(m):<24}{s['qoe']['mean']:>9.3f}{s['vmaf']['mean']:>8.1f}"
            f"{s['latency_ms']['p50']:>9.1f}{100 * s['deadline_miss_rate']:>8.1f}"
            f"{s['path_decision_ms']['p50']:>12.4f}"
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_all(run_dir: str) -> str:
    json_path = os.path.join(run_dir, "evaluation_results.json")
    if not os.path.exists(json_path):
        sys.exit(f"no evaluation_results.json in {run_dir}")
    with open(json_path) as fh:
        results = json.load(fh)

    meta = results.get("meta", {})
    summary = results["summary"]
    distributions = results.get("distributions", {})
    traces = results.get("traces", {})

    # Order methods by QoE (best first); keeps 'learned' wherever it ranks.
    order = sorted(summary.keys(), key=lambda m: summary[m]["qoe"]["mean"], reverse=True)

    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    deadline_ms = meta.get("deadline_ms", 180.0)

    print(f"Run dir   : {run_dir}")
    print(f"Backend   : {meta.get('backend', '?')}   episodes={meta.get('episodes', '?')}   "
          f"paths={meta.get('num_paths', '?')}   learned={meta.get('has_learned', False)}")
    print("Rendering figures:")

    fig_qoe_bar(summary, order, fig_dir)
    fig_metric_panels(summary, order, deadline_ms, fig_dir)
    fig_qoe_distribution(distributions, order, fig_dir)
    fig_decision_time(summary, distributions, order, fig_dir)
    fig_quality_vs_cost(summary, order, fig_dir)
    fig_timeseries(traces, meta, order, fig_dir)
    fig_split_behavior(traces, meta, order, fig_dir)
    fig_latency_cdf(summary, distributions, order, meta, fig_dir)
    fig_radar(summary, order, fig_dir)
    fig_path_metrics(traces, meta, order, fig_dir)
    fig_app_bitrate(traces, distributions, summary, meta, order, fig_dir)
    fig_ablation(summary, fig_dir)

    csv_path = os.path.join(fig_dir, "summary_table.csv")
    write_summary_csv(summary, order, csv_path)
    print(f"  - summary_table.csv")

    print_table(summary, order)
    print(f"\nFigures written to {fig_dir}/")
    return fig_dir


if __name__ == "__main__":
    generate_all(_resolve_run_dir(sys.argv))
