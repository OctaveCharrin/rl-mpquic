"""Tests for the VMAF curve and QoE reward shaping."""

import pytest

from src.ns3env.learned_vmaf import load_learned_vmaf_fn
from src.ns3env.qoe import (
    QoEWeights,
    compute_qoe_reward,
    compute_transport_reward,
    qoe_components,
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


def test_learned_vmaf_fn_loads_and_is_bounded_and_monotonic():
    vmaf_fn = load_learned_vmaf_fn()
    scores = [
        vmaf_fn(bitrate_kbps=r, latency_ms=15, jitter_ms=5, loss=0.0)
        for r in (200, 500, 1000, 1800, 2400)
    ]
    assert all(0.0 <= s <= 100.0 for s in scores)
    assert all(a <= b + 1e-6 for a, b in zip(scores, scores[1:]))  # non-decreasing in bitrate
    # More loss should not improve quality at fixed bitrate.
    clean = vmaf_fn(bitrate_kbps=1500, latency_ms=15, jitter_ms=5, loss=0.0)
    lossy = vmaf_fn(bitrate_kbps=1500, latency_ms=15, jitter_ms=5, loss=0.08)
    assert lossy <= clean + 1e-6


def test_qoe_reward_uses_pluggable_vmaf_fn():
    w = QoEWeights()
    vmaf_fn = load_learned_vmaf_fn()
    # The reward and the logged component both reflect the supplied scorer.
    r_default = compute_qoe_reward(bitrate_kbps=1500, latency_ms=20, jitter_ms=5, loss=0.0, weights=w)
    r_learned = compute_qoe_reward(
        bitrate_kbps=1500, latency_ms=20, jitter_ms=5, loss=0.0, weights=w, vmaf_fn=vmaf_fn
    )
    assert -2.0 <= r_learned <= 1.0
    comps = qoe_components(bitrate_kbps=1500, latency_ms=20, jitter_ms=5, loss=0.0, vmaf_fn=vmaf_fn)
    assert comps["vmaf"] == pytest.approx(
        vmaf_fn(bitrate_kbps=1500, latency_ms=20, jitter_ms=5, loss=0.0)
    )
    # Default path is unchanged (still the log curve).
    assert r_default == pytest.approx(
        compute_qoe_reward(bitrate_kbps=1500, latency_ms=20, jitter_ms=5, loss=0.0, weights=w)
    )


def test_learned_reward_drops_explicit_loss_penalty():
    w = QoEWeights()
    # A scorer whose VMAF ignores loss isolates the *explicit* loss penalty.
    const_vmaf = lambda **kw: 80.0  # noqa: E731

    # Default (log-curve) reward: loss is penalized explicitly.
    d_clean = compute_qoe_reward(bitrate_kbps=1500, latency_ms=20, jitter_ms=5, loss=0.0, weights=w)
    d_lossy = compute_qoe_reward(bitrate_kbps=1500, latency_ms=20, jitter_ms=5, loss=0.5, weights=w)
    assert d_clean > d_lossy  # default path keeps -d*loss

    # Learned path: explicit loss penalty dropped, so a loss-blind VMAF makes the
    # reward invariant to loss (latency/jitter penalties still apply, unchanged).
    l_clean = compute_qoe_reward(
        bitrate_kbps=1500, latency_ms=20, jitter_ms=5, loss=0.0, weights=w, vmaf_fn=const_vmaf
    )
    l_lossy = compute_qoe_reward(
        bitrate_kbps=1500, latency_ms=20, jitter_ms=5, loss=0.5, weights=w, vmaf_fn=const_vmaf
    )
    assert l_clean == pytest.approx(l_lossy)
