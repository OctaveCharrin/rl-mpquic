#!/usr/bin/env python3
"""
Evaluate a trained policy against split/bitrate baselines.

Examples:
    python evaluate.py --backend mock --app runs/<ts>/app.pth --transport runs/<ts>/transport.pth
    python evaluate.py --backend mock            # baselines only
"""

from __future__ import annotations

import argparse

from src.train.config import load_config
from src.train.evaluate import run_evaluation


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    p.add_argument("--backend", choices=["mock", "ns3"], default="mock")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--app", default=None, help="App agent checkpoint (app.pth)")
    p.add_argument("--transport", default=None, help="Transport agent checkpoint (transport.pth)")
    p.add_argument("--show-output", action="store_true", help="stream NS-3 stdout/stderr")
    p.add_argument(
        "--learned-vmaf",
        action="store_true",
        help="score QoE with the learned QoS->VMAF surrogate (match how you trained)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    run_evaluation(
        cfg,
        backend=args.backend,
        episodes=args.episodes,
        seed=args.seed,
        app_ckpt=args.app,
        transport_ckpt=args.transport,
        show_output=args.show_output,
        use_learned_vmaf=args.learned_vmaf or None,
    )


if __name__ == "__main__":
    main()
