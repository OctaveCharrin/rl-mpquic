"""Tests for the trace-driven mock data plane."""

import numpy as np

from src.ns3env.dataplane import (
    MockRealtimeConfig,
    MockRealtimeDataPlane,
    _normalize_split,
    _split_bytes,
)


def _even(n):
    return [1.0 / n] * n


def test_reset_obs_shape_and_initial_state():
    dp = MockRealtimeDataPlane(MockRealtimeConfig(seed=3))
    obs = dp.reset(seed=3)
    n = dp.num_paths
    assert obs.num_paths == n
    assert obs.app_decision_due is True  # frame 0 is an app decision
    assert obs.last_bytes == 0
    assert len(obs.cwnd) == len(obs.srtt_ms) == len(obs.path_loss) == n


def test_determinism_same_seed():
    cfg = MockRealtimeConfig(seed=7, episode_seconds=2.0)
    dp1 = MockRealtimeDataPlane(cfg)
    dp2 = MockRealtimeDataPlane(MockRealtimeConfig(seed=7, episode_seconds=2.0))
    dp1.reset(seed=7)
    dp2.reset(seed=7)
    n = dp1.num_paths
    r1, r2 = [], []
    for _ in range(20):
        r1.append(dp1.step_frame(1500, _even(n)).latency_ms)
        r2.append(dp2.step_frame(1500, _even(n)).latency_ms)
    assert r1 == r2


def test_overdriving_one_path_builds_queue():
    # Send a huge bitrate entirely on path 0: serialization >> capacity, so the
    # standing queue (hence latency) grows over time.
    dp = MockRealtimeDataPlane(MockRealtimeConfig(seed=1, episode_seconds=5.0))
    dp.reset(seed=1)
    n = dp.num_paths
    split = [1.0] + [0.0] * (n - 1)
    early = np.mean([dp.step_frame(6000, split).latency_ms for _ in range(5)])
    for _ in range(40):
        dp.step_frame(6000, split)
    late = np.mean([dp.step_frame(6000, split).latency_ms for _ in range(5)])
    assert late > early


def test_episode_horizon():
    dp = MockRealtimeDataPlane(MockRealtimeConfig(seed=1, episode_seconds=1.0, fps=30.0))
    dp.reset(seed=1)
    n = dp.num_paths
    steps = 0
    while not dp.is_done() and steps < 1000:
        dp.step_frame(1500, _even(n))
        steps += 1
    assert dp.is_done()
    assert steps == 30  # 1 s * 30 fps


def test_normalize_and_split_helpers():
    assert _normalize_split([0, 0, 0], 3) == [1 / 3, 1 / 3, 1 / 3]
    s = _normalize_split([1, 3], 2)
    assert abs(sum(s) - 1.0) < 1e-9 and s[1] > s[0]
    shares = _split_bytes(1000, [0.5, 0.5])
    assert sum(shares) == 1000
    shares = _split_bytes(1001, [0.3, 0.3, 0.4])
    assert sum(shares) == 1001
