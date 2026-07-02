#!/usr/bin/env python3
"""Reproduce the TCP episode-wedge without the RL agents (diagnostic).

Drives the NS-3 bridge with a neutral policy (even split, fixed bitrate) for N
episodes and reports per-episode frame-expiry rates, with the C++ body's
--churnLog diagnostics streaming to stderr. The wedge (a whole episode at ~1.0
loss on the TCP transport, first seen at episode 32 of training runs) is driven
by the deterministic per-seed churn sequence, so it should reproduce — if it
needs the trained agent's split pattern, rerun training with churn_log instead.

Usage:
    uv run python scripts/wedge_repro.py [--episodes 34] [--bitrate 2700]
        [--config configs/dynamic.yaml] [--transport tcp] 2> wedge_diag.log
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dynamic.yaml")
    ap.add_argument("--episodes", type=int, default=34)
    ap.add_argument("--bitrate", type=float, default=2700.0)
    ap.add_argument("--transport", choices=["tcp", "udp"], default="tcp")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.transport = args.transport
    dp = cfg.ns3_dataplane(seed=1, show_output=True)
    dp.config.churn_log = True

    try:
        dp.reset(seed=1)
        n = dp.num_paths
        for ep in range(args.episodes):
            if ep > 0:
                dp.reset()
            frames = 0
            lost = 0.0
            late = 0
            while not dp.is_done():
                r = dp.step_frame(args.bitrate, [1.0 / n] * n)
                frames += 1
                lost += r.loss
                late += 1 if r.latency_ms > cfg.deadline_ms else 0
            flag = "  <== WEDGED" if late / max(1, frames) > 0.5 else ""
            print(
                f"[wedge_repro ep {ep:3d}] frames={frames} meanLoss={lost / max(1, frames):.3f} "
                f"lateRate={late / max(1, frames):.3f}{flag}",
                flush=True,
            )
    finally:
        dp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
