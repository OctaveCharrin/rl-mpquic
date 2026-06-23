"""Tests for the VMAF curve and QoE reward shaping."""

import pytest

from src.ns3env.qoe import (
    QoEWeights,
    compute_qoe_reward,
    compute_transport_reward,
    vmaf_for_kbps,
)


def test_vmaf_monotonic_and_bounded():
    rates = [200, 300, 800, 1500, 3000, 6000, 12000]
    vmafs = [vmaf_for_kbps(r) for r in rates]
    assert all(0.0 <= v <= 100.0 for v in vmafs)
    assert all(a <= b for a, b in zip(vmafs, vmafs[1:]))  # non-decreasing


def test_vmaf_concave():
    # Diminishing returns: equal log-steps give shrinking VMAF gains is not
    # required, but a doubling low should beat a doubling high.
    low_gain = vmaf_for_kbps(600) - vmaf_for_kbps(300)
    high_gain = vmaf_for_kbps(6000) - vmaf_for_kbps(3000)
    assert low_gain > high_gain


def test_qoe_rewards_quality_penalizes_latency_loss():
    w = QoEWeights()
    good = compute_qoe_reward(bitrate_kbps=4000, latency_ms=20, jitter_ms=2, loss=0.0, weights=w)
    bad = compute_qoe_reward(bitrate_kbps=4000, latency_ms=400, jitter_ms=80, loss=0.5, weights=w)
    assert good > bad
    # Higher bitrate at equal network conditions scores higher.
    hi = compute_qoe_reward(bitrate_kbps=4000, latency_ms=30, jitter_ms=5, loss=0.0, weights=w)
    lo = compute_qoe_reward(bitrate_kbps=600, latency_ms=30, jitter_ms=5, loss=0.0, weights=w)
    assert hi > lo


def test_qoe_bounds():
    w = QoEWeights()
    r = compute_qoe_reward(bitrate_kbps=10, latency_ms=5000, jitter_ms=5000, loss=1.0, weights=w)
    assert -2.0 <= r <= 1.0


def test_transport_reward_excludes_quality_but_punishes_loss():
    w = QoEWeights()
    clean = compute_transport_reward(latency_ms=20, jitter_ms=2, loss=0.0, weights=w)
    lossy = compute_transport_reward(latency_ms=20, jitter_ms=2, loss=1.0, weights=w)
    assert clean > lossy
    assert -2.0 <= lossy <= 1.0
