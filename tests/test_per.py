"""Tests for Prioritized Experience Replay (P2): buffers + agent integration."""

import numpy as np
import torch

from src.rl.replay_buffer import (
    PrioritizedReplayBuffer,
    PrioritizedStructuredReplayBuffer,
)
from src.rl.sac_agent import SACAgent, SACConfig
from src.rl.scoring_sac_agent import ScoringSACAgent


def _cfg(**kw):
    base = dict(
        hidden_dim=32, batch_size=16, buffer_size=128, start_steps=0,
        update_after=16, updates_per_step=1, device="cpu",
    )
    base.update(kw)
    return SACConfig(**base)


def _fill_flat(buf, n, rng):
    for _ in range(n):
        buf.push(
            rng.standard_normal(4).astype(np.float32),
            rng.uniform(-1, 1, 2).astype(np.float32),
            float(rng.standard_normal()),
            rng.standard_normal(4).astype(np.float32),
            False,
        )


def test_high_priority_index_oversampled():
    rng = np.random.default_rng(0)
    buf = PrioritizedReplayBuffer(4, 2, capacity=64, alpha=0.6)
    _fill_flat(buf, 64, rng)
    # Give index 7 a huge TD error; everything else tiny.
    lo = np.arange(64)
    buf.update_priorities(lo, np.full(64, 0.01))
    buf.update_priorities(np.array([7]), np.array([100.0]))
    counts = np.zeros(64, dtype=int)
    for _ in range(200):
        _, _, _, _, _, idx, _ = buf.sample(16, beta=0.4)
        for i in idx:
            counts[i] += 1
    assert counts[7] == counts.max()
    assert counts[7] > 5 * (counts.sum() - counts[7]) / 63  # >> average


def test_is_weights_in_unit_interval():
    rng = np.random.default_rng(1)
    buf = PrioritizedReplayBuffer(4, 2, capacity=64, alpha=0.6)
    _fill_flat(buf, 64, rng)
    buf.update_priorities(np.arange(64), rng.uniform(0.1, 5.0, 64))
    _, _, _, _, _, _, w = buf.sample(16, beta=0.5)
    assert w.shape == (16, 1)
    assert np.all(w > 0.0) and np.all(w <= 1.0 + 1e-6)
    assert np.isclose(w.max(), 1.0)  # normalized by batch max


def test_structured_per_sample_has_indices_and_weights():
    rng = np.random.default_rng(2)
    g, f, n = 4, 6, 5
    buf = PrioritizedStructuredReplayBuffer(g, f, n, capacity=64, alpha=0.6)
    for _ in range(64):
        buf.push(
            rng.random(g).astype(np.float32), rng.random((n, f)).astype(np.float32),
            np.ones(n, np.float32), rng.random(n).astype(np.float32), 0.0,
            rng.random(g).astype(np.float32), rng.random((n, f)).astype(np.float32),
            np.ones(n, np.float32), False,
        )
    b = buf.sample(16, beta=0.4)
    assert b["indices"].shape == (16,)
    assert b["weights"].shape == (16, 1)
    assert np.all(b["weights"] > 0) and np.all(b["weights"] <= 1.0 + 1e-6)


def test_prioritized_false_uniform_untouched():
    """A default (prioritized=False) agent must not build a PER buffer."""
    agent = SACAgent(obs_dim=4, act_dim=2, config=_cfg())
    from src.rl.replay_buffer import ReplayBuffer
    assert type(agent.buffer) is ReplayBuffer  # exact type, not a PER subclass


def test_per_agent_update_finite():
    agent = SACAgent(obs_dim=4, act_dim=2, config=_cfg(prioritized=True, updates_per_step=5))
    assert isinstance(agent.buffer, PrioritizedReplayBuffer)
    rng = np.random.default_rng(3)
    for _ in range(64):
        agent.store(
            rng.standard_normal(4).astype(np.float32),
            rng.uniform(-1, 1, 2).astype(np.float32),
            float(rng.standard_normal()),
            rng.standard_normal(4).astype(np.float32), False,
        )
    losses = agent.update()
    assert losses is not None and np.isfinite(losses["critic_loss"])


def test_scoring_per_agent_update_finite():
    agent = ScoringSACAgent(
        4, 6, 5, config=_cfg(prioritized=True, critic_layernorm=True, updates_per_step=5)
    )
    assert isinstance(agent.buffer, PrioritizedStructuredReplayBuffer)
    rng = np.random.default_rng(4)
    for _ in range(64):
        agent.buffer.push(
            rng.random(4).astype(np.float32), rng.random((5, 6)).astype(np.float32),
            np.ones(5, np.float32), rng.random(5).astype(np.float32), 0.0,
            rng.random(4).astype(np.float32), rng.random((5, 6)).astype(np.float32),
            np.ones(5, np.float32), False,
        )
        agent._stores += 1
    losses = agent.update()
    assert losses is not None and np.isfinite(losses["critic_loss"])


def test_per_resume_restores_updates_and_beta(tmp_path):
    """Faithful PER resume: _updates (and thus annealed beta) survives save/load."""
    cfg = _cfg(prioritized=True, per_beta0=0.4, per_beta_steps=1000)
    a = SACAgent(obs_dim=4, act_dim=2, config=cfg)
    rng = np.random.default_rng(6)
    _fill_flat(a.buffer, 64, rng)
    for _ in range(50):
        a._update_once()
    assert a._updates == 50
    beta_before = a._beta()
    assert beta_before > cfg.per_beta0  # beta has annealed past its start

    path = tmp_path / "app_buffer.npz"
    a.save_buffer(str(path))

    b = SACAgent(obs_dim=4, act_dim=2, config=cfg)
    assert b._updates == 0 and b._beta() == cfg.per_beta0  # fresh agent
    b.load_buffer(str(path))
    assert b._updates == 50
    assert b._beta() == beta_before  # anneal schedule continues, not restarts


def test_per_buffer_persistence_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    a = PrioritizedReplayBuffer(4, 2, capacity=64, alpha=0.6)
    _fill_flat(a, 80, rng)
    a.update_priorities(np.arange(64), rng.uniform(0.1, 3.0, 64))
    path = tmp_path / "per.npz"
    a.save(str(path), stores=80)

    b = PrioritizedReplayBuffer(4, 2, capacity=64, alpha=0.6)
    assert b.load(str(path)) == (80, 0)
    assert np.array_equal(a._tree.tree, b._tree.tree)
    assert a._max_p == b._max_p
    assert np.array_equal(a.obs, b.obs)
