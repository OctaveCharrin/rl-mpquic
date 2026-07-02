"""Env observation/reward tests and a mock end-to-end training smoke test."""

import numpy as np

from src.ns3env.dataplane import MockRealtimeConfig, MockRealtimeDataPlane
from src.ns3env.realtime_env import HierarchicalRealtimeEnv
from src.train.config import ExperimentConfig, load_config, parse_delay_ms, parse_rate_mbps
from src.train.hierarchical_train import run_training


def _env(episode_seconds=2.0):
    dp = MockRealtimeDataPlane(MockRealtimeConfig(seed=2, episode_seconds=episode_seconds))
    return HierarchicalRealtimeEnv(dp, episode_seconds=episode_seconds)


def test_obs_dimensions():
    env = _env()
    env.reset(seed=2)
    n = env.num_paths
    assert env.app_obs_dim == 5
    assert env.transport_obs_dim == 4 + 5 * n
    assert env.transport_act_dim == n
    assert env.build_app_obs().shape == (5,)
    assert env.build_transport_obs().shape == (4 + 5 * n,)


def test_transport_obs_encodes_target_bitrate():
    env = _env()
    obs = env.reset(seed=2)
    lo = env.build_transport_obs(obs, env.video.min_bitrate_kbps)[0]
    hi = env.build_transport_obs(obs, env.video.max_bitrate_kbps)[0]
    assert hi > lo  # first feature is the normalized target bitrate (hierarchy)


def test_app_window_reward_accumulates_and_clears():
    env = _env()
    env.reset(seed=2)
    n = env.num_paths
    for _ in range(10):
        env.step(1500, [1.0 / n] * n)
    r, comps = env.pop_app_window_reward()
    assert np.isfinite(r)
    assert "vmaf" in comps
    # Window cleared: a second pop with no new frames yields the neutral reward.
    r2, _ = env.pop_app_window_reward()
    assert r2 == 0.0


def test_rate_delay_parsers():
    assert parse_rate_mbps("8Mbps") == 8.0
    assert abs(parse_rate_mbps("500kbps") - 0.5) < 1e-9
    assert parse_delay_ms("10ms") == 10.0
    assert parse_delay_ms("0.5s") == 500.0


def test_transport_defaults_to_tcp_and_round_trips():
    cfg = ExperimentConfig()
    assert cfg.transport == "tcp"
    assert cfg.ns3_dataplane().config.transport == "tcp"

    cfg.transport = "udp"
    assert cfg.ns3_dataplane().config.transport == "udp"


def test_load_config_parses_run_transport(tmp_path):
    path = tmp_path / "udp.yaml"
    path.write_text("run:\n  transport: udp\n")
    cfg = load_config(str(path))
    assert cfg.transport == "udp"
    assert load_config(None).transport == "tcp"


def test_training_smoke_mock(tmp_path):
    cfg = load_config(None)
    cfg.episode_seconds = 1.0  # 30 frames/episode -> fast
    cfg.sac.batch_size = 16
    cfg.sac.start_steps = 0
    cfg.sac.update_after = 16
    cfg.sac.hidden_dim = 32
    out = run_training(
        cfg, backend="mock", episodes=2, out_dir=str(tmp_path), log_every=0
    )
    assert out["episodes"] == 2
    assert len(out["history"]) == 2
    for ep in out["history"]:
        assert np.isfinite(ep["app_reward_mean"])
        assert np.isfinite(ep["transport_reward_mean"])
    assert (tmp_path / "app.pth").exists()
    assert (tmp_path / "transport.pth").exists()
    assert (tmp_path / "stats.json").exists()
