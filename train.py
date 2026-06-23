#!/usr/bin/env python3
"""
Train the dual-agent hierarchical controller.

Examples:
    python train.py --backend mock --episodes 50
    python train.py --backend ns3 --episodes 5 --show-output
"""

from __future__ import annotations

import argparse

from src.train.config import load_config
from src.train.hierarchical_train import run_training


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    p.add_argument("--backend", choices=["mock", "ns3"], default="mock")
    p.add_argument("--episodes", type=int, default=None, help="override config episodes")
    p.add_argument("--seed", type=int, default=None, help="override base seed")
    p.add_argument("--out-dir", default=None, help="run output directory")
    p.add_argument("--show-output", action="store_true", help="stream NS-3 stdout/stderr")
    args = p.parse_args()

    cfg = load_config(args.config)
    run_training(
        cfg,
        backend=args.backend,
        episodes=args.episodes,
        seed=args.seed,
        out_dir=args.out_dir,
        show_output=args.show_output,
    )


if __name__ == "__main__":
    main()
