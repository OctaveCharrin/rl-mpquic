"""Tests for the non-stationary mock dynamics (churn / regime / bursts / corr)."""

import numpy as np

from src.ns3env.dataplane import DynamicsConfig, MockRealtimeConfig, MockRealtimeDataPlane


def _dp(dynamics, seed=7, n=4, episode_seconds=10.0):
    cfg = MockRealtimeConfig(
        base_mbps=(3.0,) * n,
        base_rtt_ms=(20.0,) * n,
        amp=(0.4,) * n,
        period_s=(12.0,) * n,
        cross_frac=(0.4,) * n,
        cross_period_s=(5.0,) * n,
        episode_seconds=episode_seconds,
        seed=seed,
        dynamics=dynamics,
    )
    return MockRealtimeDataPlane(cfg)


def test_static_when_dynamics_off():
    # No dynamics block: every path always live, mask all-ones.
    dp = _dp(None)
    obs = dp.reset(seed=7)
    n = dp.num_paths
    assert obs.path_active == [1.0] * n
    for _ in range(30):
        dp.step_frame(1500, [1.0 / n] * n)
        assert dp.current_obs().path_active == [1.0] * n


def test_churn_varies_active_count():
    d = DynamicsConfig(enabled=True, churn=True, churn_up_rate=0.3,
                       churn_down_rate=0.4, min_active=1)
    dp = _dp(d, episode_seconds=30.0)
    dp.reset(seed=7)
    n = dp.num_paths
    counts = set()
    while not dp.is_done():
        counts.add(int(sum(dp.current_obs().path_active)))
        dp.step_frame(1500, [1.0 / n] * n)
    assert len(counts) > 1  # active count actually changes
    assert min(counts) >= 1  # min_active respected


def test_regime_swaps_best_path():
    d = DynamicsConfig(enabled=True, regime=True, regime_rate=0.6,
                       regime_lo=0.2, regime_hi=1.4)
    dp = _dp(d, episode_seconds=30.0)
    dp.reset(seed=3)
    n = dp.num_paths
    best, swaps, prev = None, 0, None
    while not dp.is_done():
        o = dp.current_obs()
        b = int(np.argmax(o.path_throughput_mbps))
        if prev is not None and b != prev:
            swaps += 1
        prev = b
        dp.step_frame(1500, [1.0 / n] * n)
    assert swaps >= 1  # the best path changes over the episode


def test_bytes_on_dead_path_count_as_loss():
    # Force path 0 down hard, then route the whole frame onto it -> loss ~ 1.
    d = DynamicsConfig(enabled=True, churn=True, churn_up_rate=0.0,
                       churn_down_rate=10.0, min_active=1)
    dp = _dp(d, n=4, episode_seconds=10.0)
    dp.reset(seed=1)
    n = dp.num_paths
    # Advance a few frames so churn drives some paths down.
    for _ in range(10):
        dp.step_frame(1500, [1.0 / n] * n)
    o = dp.current_obs()
    dead = [i for i in range(n) if o.path_active[i] < 0.5]
    assert dead, "expected at least one churned-out path"
    split = [0.0] * n
    split[dead[0]] = 1.0
    res = dp.step_frame(1500, split)
    assert res.loss > 0.5  # routing onto a dead path is penalized
    assert res.bytes_delivered == 0


def test_dynamics_deterministic_per_seed():
    d = DynamicsConfig(enabled=True, churn=True, regime=True, burst=True,
                       corr_groups=[[0, 1]])
    a = _dp(d, seed=11); a.reset(seed=11)
    b = _dp(d, seed=11); b.reset(seed=11)
    n = a.num_paths
    ra = [a.step_frame(1500, [1.0 / n] * n).latency_ms for _ in range(60)]
    rb = [b.step_frame(1500, [1.0 / n] * n).latency_ms for _ in range(60)]
    assert ra == rb
