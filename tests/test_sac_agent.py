"""Smoke tests for the SAC agent and the App/Path wrappers."""

import numpy as np

from src.rl.app_agent import AppAgent
from src.rl.sac_agent import SACAgent, SACConfig
from src.rl.path_agent import PathAgent


def _tiny_cfg():
    return SACConfig(
        hidden_dim=16,
        batch_size=8,
        buffer_size=200,
        start_steps=0,
        update_after=8,
        updates_per_step=1,
        device="cpu",
    )


def test_select_action_shape_and_range():
    agent = SACAgent(obs_dim=5, act_dim=3, config=_tiny_cfg())
    a = agent.select_action(np.zeros(5, dtype=np.float32))
    assert a.shape == (3,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_update_runs_after_enough_data():
    agent = SACAgent(obs_dim=4, act_dim=2, config=_tiny_cfg())
    assert agent.update() is None  # not enough data yet
    rng = np.random.default_rng(0)
    for _ in range(32):
        o = rng.standard_normal(4).astype(np.float32)
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        no = rng.standard_normal(4).astype(np.float32)
        agent.store(o, a, float(rng.standard_normal()), no, False)
    losses = agent.update()
    assert losses is not None
    assert {"critic_loss", "policy_loss", "alpha"} <= set(losses)
    assert np.isfinite(losses["critic_loss"])


def test_app_agent_maps_to_bitrate_range():
    app = AppAgent(obs_dim=6, min_kbps=300, max_kbps=6000, config=_tiny_cfg())
    for _ in range(20):
        kbps, raw = app.select(np.zeros(6, dtype=np.float32))
        assert 300.0 - 1e-6 <= kbps <= 6000.0 + 1e-6
        assert raw.shape == (1,)


def test_path_agent_split_is_simplex():
    tr = PathAgent(obs_dim=10, num_paths=3, config=_tiny_cfg())
    split, raw = tr.select(np.zeros(10, dtype=np.float32))
    assert split.shape == (3,)
    assert abs(float(split.sum()) - 1.0) < 1e-5
    assert np.all(split >= 0.0)


def test_checkpoint_roundtrip():
    a = SACAgent(obs_dim=4, act_dim=2, config=_tiny_cfg())
    b = SACAgent(obs_dim=4, act_dim=2, config=_tiny_cfg())
    b.load_state_dict(a.state_dict())
    o = np.ones(4, dtype=np.float32)
    assert np.allclose(
        a.select_action(o, deterministic=True), b.select_action(o, deterministic=True)
    )
