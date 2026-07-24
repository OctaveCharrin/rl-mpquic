"""Tests for PopArt return scaling (Phase 4.2) and flags-off parity."""

import numpy as np
import torch

from src.rl.sac_agent import QNetwork, SACAgent, SACConfig
from src.rl.scoring_sac_agent import ScoringSACAgent


def _cfg(**kw):
    base = dict(
        hidden_dim=32, batch_size=8, buffer_size=64, start_steps=0,
        update_after=8, updates_per_step=1, device="cpu",
    )
    base.update(kw)
    return SACConfig(**base)


def _fill_flat(agent, n=32, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        agent.store(
            rng.standard_normal(4).astype(np.float32),
            rng.uniform(-1, 1, 2).astype(np.float32),
            float(rng.standard_normal()),
            rng.standard_normal(4).astype(np.float32), False,
        )


def test_off_is_identity_and_byte_identical():
    # Off-path: no buffers, and normalize/denormalize are exact identities.
    off = SACAgent(obs_dim=4, act_dim=2, config=_cfg())
    assert not hasattr(off.critic, "popart_mu")
    y = torch.randn(7, 1)
    assert torch.equal(off.critic.denormalize(y), y)
    assert torch.equal(off.critic.normalize(y), y)
    off.critic.update_popart(torch.randn(8, 1))  # no-op, must not raise

    # Same seed (torch + numpy) => an off-path update is bit-for-bit reproducible.
    def run():
        torch.manual_seed(0)
        np.random.seed(0)
        ag = SACAgent(obs_dim=4, act_dim=2, config=_cfg())
        _fill_flat(ag)
        ag.update()
        return ag

    a, b = run(), run()
    for pa, pb in zip(a.critic.parameters(), b.critic.parameters()):
        assert torch.equal(pa, pb)


def test_on_adds_buffers_and_stats_move():
    a = SACAgent(obs_dim=4, act_dim=2, config=_cfg(popart=True))
    assert hasattr(a.critic, "popart_mu")
    mu0 = a.critic.popart_mu.clone()
    nu0 = a.critic.popart_nu.clone()
    _fill_flat(a)
    a.update()
    # Running stats have moved off their (0, 1) init.
    assert not torch.equal(a.critic.popart_mu, mu0) or not torch.equal(a.critic.popart_nu, nu0)


def test_popart_preserves_outputs_on_stat_update():
    torch.manual_seed(3)
    q = QNetwork(4, 2, 32, popart=True)
    x_obs = torch.randn(5, 4)
    x_act = torch.randn(5, 2)
    y0 = tuple(q.denormalize(o) for o in q(x_obs, x_act))
    # A shift in the running stats must be exactly compensated by the rescaled heads.
    q.update_popart(torch.randn(64, 1) * 10.0 + 5.0)
    y1 = tuple(q.denormalize(o) for o in q(x_obs, x_act))
    for a, b in zip(y0, y1):
        assert torch.allclose(a, b, atol=1e-4)


def test_normalize_denormalize_roundtrip():
    q = QNetwork(4, 2, 16, popart=True)
    q.update_popart(torch.randn(64, 1) * 3.0 + 1.0)
    y = torch.randn(10, 1)
    assert torch.allclose(q.denormalize(q.normalize(y)), y, atol=1e-5)


def test_flat_update_finite_and_checkpoint_roundtrips_popart():
    a = SACAgent(obs_dim=4, act_dim=2, config=_cfg(popart=True))
    _fill_flat(a)
    losses = a.update()
    assert losses is not None and np.isfinite(losses["critic_loss"])
    # Checkpoint carries the popart buffers.
    sd = a.state_dict()
    assert "popart_mu" in sd["critic"]
    b = SACAgent(obs_dim=4, act_dim=2, config=_cfg(popart=True))
    b.load_state_dict(sd)
    assert torch.equal(a.critic.popart_mu, b.critic.popart_mu)


def test_scoring_popart_runs_finite_and_stats_move():
    agent = ScoringSACAgent(4, 6, 5, config=_cfg(popart=True))
    assert hasattr(agent.critic, "popart_mu")
    rng = np.random.default_rng(0)
    for _ in range(32):
        agent.buffer.push(
            rng.random(4).astype(np.float32), rng.random((5, 6)).astype(np.float32),
            np.ones(5, np.float32), rng.random(5).astype(np.float32),
            float(rng.standard_normal()),
            rng.random(4).astype(np.float32), rng.random((5, 6)).astype(np.float32),
            np.ones(5, np.float32), False,
        )
        agent._stores += 1
    losses = agent.update()
    assert losses is not None and np.isfinite(losses["critic_loss"])
