"""Tests for per-episode domain randomization of the mock dynamics (Phase 4.1)."""

import numpy as np

from src.ns3env.dataplane import DynamicsConfig, DynamicsRandomization
from src.train.config import _parse_dynamics


def _base():
    return DynamicsConfig(
        enabled=True, churn=True, regime=True, burst=True,
        corr_groups=[[0, 1]], regime_lo=0.4, regime_hi=1.2,
    )


def test_parse_absent_randomize_is_none():
    dyn = _parse_dynamics({"enabled": True, "regime": True})
    assert dyn is not None
    assert dyn.randomize is None  # legacy: fixed dynamics


def test_parse_randomize_block():
    dyn = _parse_dynamics(
        {
            "enabled": True,
            "regime": True,
            "randomize": {
                "enabled": True,
                "regime_rate": [0.1, 0.5],
                "burst_intensity": [0.1, 0.4],
            },
        }
    )
    assert dyn.randomize is not None
    assert dyn.randomize.regime_rate == (0.1, 0.5)
    assert dyn.randomize.burst_intensity == (0.1, 0.4)
    assert dyn.randomize.churn_up_rate is None  # unset key not randomized


def test_sample_varies_within_ranges():
    dr = DynamicsRandomization(
        enabled=True, regime_rate=(0.1, 0.5), burst_intensity=(0.1, 0.4)
    )
    base = _base()
    samples = [dr.sample(base, np.random.default_rng(1000 + e)) for e in range(20)]
    rates = {s.regime_rate for s in samples}
    assert len(rates) > 1                       # actually varies episode-to-episode
    assert all(0.1 <= s.regime_rate <= 0.5 for s in samples)
    assert all(0.1 <= s.burst_intensity <= 0.4 for s in samples)


def test_sample_preserves_flags_groups_and_clears_randomize():
    dr = DynamicsRandomization(enabled=True, regime_rate=(0.1, 0.5))
    base = _base()
    s = dr.sample(base, np.random.default_rng(1))
    assert s.churn and s.regime and s.burst          # on/off flags preserved
    assert s.corr_groups == [[0, 1]]                 # structure preserved
    assert s.randomize is None                       # sampled config doesn't recurse
    assert s.churn_up_rate == base.churn_up_rate     # unrandomized param unchanged


def test_sample_deterministic_per_seed():
    dr = DynamicsRandomization(enabled=True, regime_rate=(0.1, 0.5), corr_intensity=(0.1, 0.5))
    base = _base()
    a = dr.sample(base, np.random.default_rng(42))
    b = dr.sample(base, np.random.default_rng(42))
    assert a.regime_rate == b.regime_rate
    assert a.corr_intensity == b.corr_intensity


def test_sample_keeps_regime_band_ordered():
    dr = DynamicsRandomization(enabled=True, regime_lo=(0.5, 1.5), regime_hi=(0.5, 1.5))
    base = _base()
    for e in range(30):
        s = dr.sample(base, np.random.default_rng(e))
        assert s.regime_lo <= s.regime_hi
