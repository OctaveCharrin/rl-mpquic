"""Round-trip tests for replay-buffer persistence across --resume (P3)."""

import numpy as np
import pytest

from src.rl.replay_buffer import ReplayBuffer, StructuredReplayBuffer
from src.rl.sac_agent import SACAgent, SACConfig
from src.rl.scoring_sac_agent import ScoringSACAgent


def _tiny_cfg():
    return SACConfig(
        hidden_dim=16, batch_size=8, buffer_size=64, start_steps=0,
        update_after=8, updates_per_step=1, device="cpu",
    )


def _fill_flat(buf, n, rng):
    for _ in range(n):
        buf.push(
            rng.standard_normal(4).astype(np.float32),
            rng.uniform(-1, 1, 2).astype(np.float32),
            float(rng.standard_normal()),
            rng.standard_normal(4).astype(np.float32),
            bool(rng.integers(2)),
        )


def test_flat_buffer_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    a = ReplayBuffer(4, 2, capacity=64)
    _fill_flat(a, 100, rng)  # wraps past capacity so _ptr != _size
    path = tmp_path / "buf.npz"
    a.save(str(path), stores=100)

    b = ReplayBuffer(4, 2, capacity=64)
    stores, updates = b.load(str(path))
    assert stores == 100
    assert updates == 0  # default when not supplied
    assert len(b) == len(a) == 64
    assert b._ptr == a._ptr
    for name in ReplayBuffer._ARRAYS:
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_structured_buffer_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    g, f, n = 4, 6, 5
    a = StructuredReplayBuffer(g, f, n, capacity=32)
    for _ in range(50):
        a.push(
            rng.random(g).astype(np.float32),
            rng.random((n, f)).astype(np.float32),
            np.ones(n, dtype=np.float32),
            rng.random(n).astype(np.float32),
            float(rng.standard_normal()),
            rng.random(g).astype(np.float32),
            rng.random((n, f)).astype(np.float32),
            np.ones(n, dtype=np.float32),
            False,
        )
    path = tmp_path / "sbuf.npz"
    a.save(str(path), stores=50)

    b = StructuredReplayBuffer(g, f, n, capacity=32)
    assert b.load(str(path)) == (50, 0)
    assert len(b) == len(a)
    for name in StructuredReplayBuffer._ARRAYS:
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_capacity_mismatch_raises(tmp_path):
    a = ReplayBuffer(4, 2, capacity=64)
    path = tmp_path / "buf.npz"
    a.save(str(path))
    with pytest.raises(ValueError, match="capacity mismatch"):
        ReplayBuffer(4, 2, capacity=128).load(str(path))


def test_shape_mismatch_raises(tmp_path):
    a = ReplayBuffer(4, 2, capacity=64)
    path = tmp_path / "buf.npz"
    a.save(str(path))
    with pytest.raises(ValueError, match="shape mismatch"):
        ReplayBuffer(5, 2, capacity=64).load(str(path))


def test_agent_save_load_buffer_restores_stores(tmp_path):
    rng = np.random.default_rng(2)
    a = SACAgent(obs_dim=4, act_dim=2, config=_tiny_cfg())
    for _ in range(30):
        a.store(
            rng.standard_normal(4).astype(np.float32),
            rng.uniform(-1, 1, 2).astype(np.float32),
            float(rng.standard_normal()),
            rng.standard_normal(4).astype(np.float32),
            False,
        )
    path = tmp_path / "app_buffer.npz"
    a.save_buffer(str(path))

    b = SACAgent(obs_dim=4, act_dim=2, config=_tiny_cfg())
    assert len(b.buffer) == 0
    b.load_buffer(str(path))
    assert b._stores == a._stores == 30
    assert len(b.buffer) == 30


def test_scoring_agent_save_load_buffer(tmp_path):
    g, f, n = 4, 6, 5
    rng = np.random.default_rng(3)
    a = ScoringSACAgent(g, f, n, config=_tiny_cfg())
    for _ in range(20):
        a.buffer.push(
            rng.random(g).astype(np.float32),
            rng.random((n, f)).astype(np.float32),
            np.ones(n, dtype=np.float32),
            rng.random(n).astype(np.float32),
            0.0,
            rng.random(g).astype(np.float32),
            rng.random((n, f)).astype(np.float32),
            np.ones(n, dtype=np.float32),
            False,
        )
        a._stores += 1
    path = tmp_path / "path_buffer.npz"
    a.save_buffer(str(path))

    b = ScoringSACAgent(g, f, n, config=_tiny_cfg())
    b.load_buffer(str(path))
    assert b._stores == 20
    assert len(b.buffer) == 20
