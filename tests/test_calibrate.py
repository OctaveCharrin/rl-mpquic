"""Tests for the G.1070 reward-calibration script."""

import importlib.util
import os

from src.ns3env.g1070 import G1070Oracle
from src.train.config import load_config

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "calibrate_reward", os.path.join(_HERE, "scripts", "calibrate_reward.py")
)
calibrate_reward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibrate_reward)


def test_synthetic_corpus_covers_axes():
    corpus = calibrate_reward.synthetic_corpus()
    assert len(corpus) > 100
    for axis in ("bitrate_kbps", "latency_ms", "jitter_ms", "loss"):
        vals = {s[axis] for s in corpus}
        assert len(vals) >= 2  # axis actually varies


def test_calibrate_respects_pin_and_constraint():
    report = calibrate_reward.calibrate(
        calibrate_reward.synthetic_corpus(), G1070Oracle()
    )
    best = report["best"]
    assert best["c"] == best["b"]          # c pinned = b
    assert best["d"] >= best["b"]          # loss stays dominant
    for g in report["grid"]:
        assert -1.0 <= g["pearson"] <= 1.0
        assert -1.0 <= g["spearman"] <= 1.0


def test_invert_vmaf_roundtrips_monotonically():
    from src.ns3env.qoe import vmaf_for_kbps

    for b in (400, 1000, 3000):
        recon = calibrate_reward._invert_vmaf(vmaf_for_kbps(b))
        assert abs(recon - b) / b < 0.05  # within 5%


def test_calibrated_config_loads_and_differs_from_default():
    base = load_config(os.path.join(_HERE, "configs", "default.yaml"))
    cal = load_config(os.path.join(_HERE, "configs", "calibrated.yaml"))
    assert cal.weights.a_quality == base.weights.a_quality  # numeraire unchanged
    assert (cal.weights.b_latency, cal.weights.d_loss) != (
        base.weights.b_latency, base.weights.d_loss
    )
    assert cal.weights.c_jitter == cal.weights.b_latency  # pinned
