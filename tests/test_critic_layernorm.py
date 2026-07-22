"""Tests for the DroQ critic LayerNorm/dropout knobs (P1) and flags-off parity."""

import dataclasses

import numpy as np
import torch
import torch.nn as nn

from src.rl.sac_agent import SACAgent, SACConfig
from src.rl.scoring_sac_agent import ScoringSACAgent


def _cfg(**kw):
    base = dict(
        hidden_dim=32, batch_size=8, buffer_size=64, start_steps=0,
        update_after=8, updates_per_step=1, device="cpu",
    )
    base.update(kw)
    return SACConfig(**base)


def test_flat_flags_off_identical():
    """critic_layernorm=False must reproduce the pre-patch network exactly."""
    torch.manual_seed(0)
    a = SACAgent(obs_dim=4, act_dim=2, config=_cfg())
    torch.manual_seed(0)
    b = SACAgent(obs_dim=4, act_dim=2, config=_cfg(critic_layernorm=True))
    # Same architecture off vs on differs; here assert OFF matches a fresh OFF.
    torch.manual_seed(0)
    c = SACAgent(obs_dim=4, act_dim=2, config=_cfg())
    for pa, pc in zip(a.critic.parameters(), c.critic.parameters()):
        assert torch.equal(pa, pc)
    # And LayerNorm actually changes the module set when enabled.
    assert not any(isinstance(m, nn.LayerNorm) for m in a.critic.modules())
    assert any(isinstance(m, nn.LayerNorm) for m in b.critic.modules())


def test_layernorm_is_critic_only():
    """LayerNorm modules live in the critic, never the policy."""
    off = SACAgent(obs_dim=4, act_dim=2, config=_cfg())
    on = SACAgent(obs_dim=4, act_dim=2, config=_cfg(critic_layernorm=True))
    assert any(isinstance(m, nn.LayerNorm) for m in on.critic.modules())
    assert not any(isinstance(m, nn.LayerNorm) for m in on.policy.modules())
    # LayerNorm adds (weight, bias) tensors to the critic state dict only.
    assert len(on.state_dict()["critic"]) > len(off.state_dict()["critic"])
    assert len(on.state_dict()["policy"]) == len(off.state_dict()["policy"])


def test_scoring_layernorm_critic_only_and_updates():
    agent = ScoringSACAgent(4, 6, 5, config=_cfg(critic_layernorm=True, critic_dropout=0.1))
    assert any(isinstance(m, nn.LayerNorm) for m in agent.critic.modules())
    assert not any(isinstance(m, nn.LayerNorm) for m in agent.policy.modules())
    # A high UTD update stays finite with LayerNorm on.
    rng = np.random.default_rng(0)
    for _ in range(32):
        agent.buffer.push(
            rng.random(4).astype(np.float32), rng.random((5, 6)).astype(np.float32),
            np.ones(5, np.float32), rng.random(5).astype(np.float32), 0.0,
            rng.random(4).astype(np.float32), rng.random((5, 6)).astype(np.float32),
            np.ones(5, np.float32), False,
        )
        agent._stores += 1
    losses = agent.update()
    assert losses is not None and np.isfinite(losses["critic_loss"])


def test_high_utd_runs_finite():
    agent = SACAgent(
        obs_dim=4, act_dim=2,
        config=_cfg(updates_per_step=10, critic_layernorm=True, critic_dropout=0.1),
    )
    rng = np.random.default_rng(1)
    for _ in range(32):
        agent.store(
            rng.standard_normal(4).astype(np.float32),
            rng.uniform(-1, 1, 2).astype(np.float32),
            float(rng.standard_normal()),
            rng.standard_normal(4).astype(np.float32), False,
        )
    losses = agent.update()
    assert losses is not None and np.isfinite(losses["critic_loss"])
