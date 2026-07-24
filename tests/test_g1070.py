"""Tests for the ITU-T G.1070 QoE oracle used in reward calibration."""

import math

import pytest

from src.ns3env.g1070 import (
    G1070Config,
    G1070Oracle,
    build_composite_oracle,
    multimedia_quality,
    video_quality,
)


def _vq(b, loss, fps=30.0, preset="h264_bp_vga_6in"):
    return video_quality(
        bitrate_kbps=b, fps=fps, loss_frac=loss,
        coeffs=G1070Config(video_preset=preset).video_coeffs,
    )


def test_video_quality_bounded():
    for b in (10, 300, 1000, 6000, 20000):
        for loss in (0.0, 0.01, 0.05, 0.5, 1.0):
            assert 1.0 <= _vq(b, loss) <= 5.0


def test_video_quality_monotonic_in_bitrate():
    vqs = [_vq(b, 0.0) for b in (300, 600, 1000, 2000, 4000, 6000)]
    assert all(a <= b + 1e-9 for a, b in zip(vqs, vqs[1:]))


def test_video_quality_decreasing_in_loss():
    vqs = [_vq(2000, l) for l in (0.0, 0.005, 0.01, 0.03, 0.1)]
    assert all(a >= b - 1e-9 for a, b in zip(vqs, vqs[1:]))
    assert vqs[0] > vqs[-1]  # loss actually bites


def test_video_quality_concave_in_bitrate():
    # Diminishing returns: a low doubling beats a high doubling.
    low = _vq(600, 0.0) - _vq(300, 0.0)
    high = _vq(6000, 0.0) - _vq(3000, 0.0)
    assert low > high


def test_mmq_monotonic_decreasing_in_latency():
    o = G1070Oracle()
    mos = [o.mos(bitrate_kbps=2000, latency_ms=lat, loss=0.0)
           for lat in (0, 50, 100, 200, 400, 800, 1000)]
    assert all(a >= b - 1e-9 for a, b in zip(mos, mos[1:]))
    assert mos[0] > mos[-1]  # delay actually costs something


def test_mos_bounded():
    o = G1070Oracle()
    for b in (10, 2000, 20000):
        for lat in (0, 200, 5000):
            for loss in (0.0, 0.5, 1.0):
                assert 1.0 <= o.mos(bitrate_kbps=b, latency_ms=lat, loss=loss) <= 5.0


def test_video_quality_anchor_spotcheck():
    # Hand-computed reference for the default preset at (1000 kbps, 30 fps, 0 loss).
    # Vq = 1 + Icoding (PplV=0), Icoding ~ 2.87 -> Vq ~ 3.87.
    vq = _vq(1000, 0.0)
    assert vq == pytest.approx(3.87, abs=0.05)


def test_presets_available_and_distinct():
    b, loss = 1000, 0.02
    a = _vq(b, loss, preset="h264_bp_vga_6in")
    c = _vq(b, loss, preset="h264_vga_9in")
    assert a != c  # different coefficient tables -> different MOS
    # 65-inch preset also loads.
    assert 1.0 <= _vq(b, loss, preset="h264_bp_vga_65in") <= 5.0


def test_jitter_ignored_without_donor():
    o = G1070Oracle()
    base = o.mos(bitrate_kbps=2000, latency_ms=30, loss=0.0, jitter_ms=0.0)
    for j in (10, 40, 200):
        assert o.mos(bitrate_kbps=2000, latency_ms=30, loss=0.0, jitter_ms=j) == base


def test_composite_dormant_with_degenerate_shipped_donor():
    # The vendored reward_model.npz is flat in jitter -> composite must stay dormant.
    o = build_composite_oracle(load_default_donor=True)
    assert o._jitter_donor is None
    base = o.mos(bitrate_kbps=2000, latency_ms=30, loss=0.0, jitter_ms=0.0)
    assert o.mos(bitrate_kbps=2000, latency_ms=30, loss=0.0, jitter_ms=80.0) == base


def test_composite_activates_with_synthetic_jitter_donor():
    # A donor that *does* vary with jitter activates the composite and lowers MOS.
    def donor(*, bitrate_kbps, latency_ms=0.0, jitter_ms=0.0, loss=0.0):
        return max(0.0, 90.0 - 0.5 * jitter_ms)  # VMAF drops with jitter

    o = build_composite_oracle(jitter_donor=donor)
    assert o._jitter_donor is not None
    clean = o.mos(bitrate_kbps=2000, latency_ms=30, loss=0.0, jitter_ms=0.0)
    jit = o.mos(bitrate_kbps=2000, latency_ms=30, loss=0.0, jitter_ms=60.0)
    assert jit < clean
    assert 1.0 <= jit <= 5.0
