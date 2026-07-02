#!/usr/bin/env python3
"""Mock <-> NS-3 dynamics parity check (environment-level, no agents).

Runs the same neutral policy (even split, fixed bitrate) through both backends:

* mock  -- MockRealtimeDataPlane driven in-process;
* NS-3  -- the C++ scenario's --selftest mode (even split at initBitrateKbps),
           launched via `./ns3 run ... --no-build` for a scenario matrix
           (static + full dynamics, tcp + udp).

and asserts the NS-3 environment is survivable and in the same loss regime as
the mock. This is the regression guard for the CONTRACT in CLAUDE.md ("keep the
mock and C++ in sync"): it would have caught the churn zombie-link bug (NS-3
static 0% but full-dynamics 80% frame expiry vs mock 0%).

The two backends draw from different RNG streams, so parity is *behavioral*
(same loss regime), not frame-identical — hence bands, not equalities.

Usage:
    uv run python scripts/parity_check.py [--config configs/dynamic.yaml]
                                          [--episodes 3] [--mechanisms]
    NS3_DIR=/path/to/ns-3-dev overrides the NS-3 tree (default ~/ns-3-dev).
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ns3env.dataplane import _encode_corr_groups, _encode_topology  # noqa: E402
from src.train.config import load_config  # noqa: E402

# Thresholds (even split @ init bitrate). The mock's dynamic meanLoss is the
# reference point; NS-3 must be survivable (low expiry) and in the same regime.
MAX_STATIC_EXPIRY = 0.01
MAX_DYNAMIC_EXPIRY = 0.15
MAX_DYNAMIC_LOSS_GAP = 0.15  # |ns3 meanLoss - mock meanLoss| upper bound

_RESULT_RE = re.compile(
    r"\[selftest ep end\] completed=(\d+) expired=(\d+).*?"
    r"lossRate=([0-9.eE+-]+) meanLoss=([0-9.eE+-]+)"
)


def mock_even_split(cfg, episodes: int) -> dict:
    """Neutral-policy episode stats on the mock backend."""
    dp = cfg.mock_dataplane(seed=1)
    n = dp.num_paths
    bitrate = cfg.video.init_bitrate_kbps
    losses, late = [], 0
    frames = 0
    for _ in range(episodes):
        dp.reset()
        while not dp.is_done():
            r = dp.step_frame(bitrate, [1.0 / n] * n)
            losses.append(r.loss)
            late += 1 if r.latency_ms > cfg.deadline_ms else 0
            frames += 1
    return {"mean_loss": statistics.mean(losses), "expiry": late / frames}


def dynamics_args(cfg) -> str:
    d = cfg.dynamics
    if d is None or not d.enabled:
        return ""
    args = (
        f"--dynamicsEnabled=1 --churn={int(d.churn)} "
        f"--churnUpRate={d.churn_up_rate} --churnDownRate={d.churn_down_rate} "
        f"--minActive={d.min_active} --regime={int(d.regime)} "
        f"--regimeRate={d.regime_rate} --regimeLo={d.regime_lo} --regimeHi={d.regime_hi} "
        f"--burst={int(d.burst)} --burstRate={d.burst_rate} "
        f"--burstIntensity={d.burst_intensity} --burstDurationS={d.burst_duration_s} "
        f"--corrRate={d.corr_rate} --corrIntensity={d.corr_intensity} "
        f"--corrDurationS={d.corr_duration_s}"
    )
    corr = _encode_corr_groups(d.corr_groups)
    if corr:
        args += f" --corrGroups={corr}"
    return args


def run_selftest(ns3_dir: str, extra: str) -> dict:
    cmd = f'./ns3 run "ns3ai_realtime_mpquic --selftest {extra}" --no-build'
    out = subprocess.run(
        cmd, shell=True, cwd=ns3_dir, capture_output=True, text=True, timeout=600
    )
    m = _RESULT_RE.search(out.stderr + out.stdout)
    if not m:
        raise RuntimeError(f"no selftest result from: {cmd}\n{out.stderr[-2000:]}")
    completed, expired = int(m.group(1)), int(m.group(2))
    return {
        "expiry": expired / max(1, completed + expired),
        "mean_loss": float(m.group(4)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dynamic.yaml")
    ap.add_argument("--episodes", type=int, default=3, help="mock episodes to average")
    ap.add_argument(
        "--mechanisms",
        action="store_true",
        help="also run per-mechanism NS-3 ablations (churn/regime/burst/corr)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    ns3_dir = os.path.abspath(os.environ.get("NS3_DIR", os.path.expanduser("~/ns-3-dev")))
    paths = f"--paths={_encode_topology(cfg.paths)}"
    dyn = dynamics_args(cfg)
    if not dyn:
        print(f"NOTE: {args.config} has no dynamics enabled; checking static parity only.")

    mock = mock_even_split(cfg, args.episodes)
    print(f"mock  even-split @{cfg.video.init_bitrate_kbps:.0f}kbps: "
          f"meanLoss={mock['mean_loss']:.3f} expiry={mock['expiry']:.3f}")

    scenarios = [(f"static {t}", f"--transport={t} {paths}") for t in ("tcp", "udp")]
    if dyn:
        scenarios += [(f"dynamic {t}", f"--transport={t} {paths} {dyn}") for t in ("tcp", "udp")]
        if args.mechanisms:
            d = cfg.dynamics
            scenarios += [
                ("churn only tcp",
                 f"--transport=tcp {paths} --dynamicsEnabled=1 --churn=1 "
                 f"--churnUpRate={d.churn_up_rate} --churnDownRate={d.churn_down_rate} "
                 f"--minActive={d.min_active}"),
                ("regime only tcp",
                 f"--transport=tcp {paths} --dynamicsEnabled=1 --regime=1 "
                 f"--regimeRate={d.regime_rate} --regimeLo={d.regime_lo} "
                 f"--regimeHi={d.regime_hi}"),
                ("burst only tcp",
                 f"--transport=tcp {paths} --dynamicsEnabled=1 --burst=1 "
                 f"--burstRate={d.burst_rate} --burstIntensity={d.burst_intensity} "
                 f"--burstDurationS={d.burst_duration_s}"),
            ]

    failures = []
    for name, extra in scenarios:
        r = run_selftest(ns3_dir, extra)
        if name.startswith("static"):
            ok = r["expiry"] <= MAX_STATIC_EXPIRY and r["mean_loss"] <= MAX_STATIC_EXPIRY
        elif name.startswith("dynamic"):
            ok = (
                r["expiry"] <= MAX_DYNAMIC_EXPIRY
                and abs(r["mean_loss"] - mock["mean_loss"]) <= MAX_DYNAMIC_LOSS_GAP
            )
        else:  # mechanism ablations: survivability only
            ok = r["expiry"] <= MAX_DYNAMIC_EXPIRY
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures.append(name)
        print(f"ns3   {name:18s}: meanLoss={r['mean_loss']:.3f} "
              f"expiry={r['expiry']:.3f}  [{status}]")

    if failures:
        print(f"\nFAIL: {', '.join(failures)}")
        return 1
    print("\nAll parity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
