#!/usr/bin/env python3
"""
Evaluate a trained policy against split/bitrate baselines and dump rich results.

Writes ``<out>/evaluation_results.json`` (summary + distributions + traces +
per-method decision timings). Pass ``--figures`` to also render the figure set
via ``evaluation/generate_figures.py``.

Examples:
    python evaluate.py --backend mock --app runs/<ts>/app.pth --transport runs/<ts>/transport.pth --figures
    python evaluate.py --backend mock            # baselines only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

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
    p.add_argument("--out", default=None, help="output run dir (default runs/eval-<ts>)")
    p.add_argument(
        "--ns3-transport",
        choices=["tcp", "udp"],
        default=None,
        help="ns3 backend per-path transport protocol (overrides the config's "
        "run.transport; ignored by --backend mock, which is already UDP-like)",
    )
    p.add_argument("--figures", action="store_true", help="render figures after evaluating")
    p.add_argument("--show-output", action="store_true", help="stream NS-3 stdout/stderr")
    p.add_argument(
        "--learned-vmaf",
        action="store_true",
        help="score QoE with the learned QoS->VMAF surrogate (match how you trained)",
    )
    p.add_argument(
        "--ablation",
        action="store_true",
        help="add single-agent ablations (app_only / transport_only) to isolate "
        "each agent's contribution; requires --app and --transport",
    )
    args = p.parse_args()

    if args.ablation and not (args.app and args.transport):
        p.error("--ablation requires both --app and --transport checkpoints")

    cfg = load_config(args.config)
    if args.ns3_transport:
        cfg.transport = args.ns3_transport
    out_dir = args.out or os.path.join(cfg.out_dir, "eval-" + time.strftime("%Y%m%d-%H%M%S"))
    run_evaluation(
        cfg,
        backend=args.backend,
        episodes=args.episodes,
        seed=args.seed,
        app_ckpt=args.app,
        transport_ckpt=args.transport,
        show_output=args.show_output,
        out_dir=out_dir,
        save_json=True,
        use_learned_vmaf=args.learned_vmaf or None,
        ablation=args.ablation,
    )

    if args.figures:
        script = os.path.join(os.path.dirname(__file__), "evaluation", "generate_figures.py")
        print(f"\nrendering figures from {out_dir} ...")
        subprocess.run([sys.executable, script, out_dir], check=False)


if __name__ == "__main__":
    main()
