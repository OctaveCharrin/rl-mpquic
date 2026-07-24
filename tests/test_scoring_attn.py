"""Tests for the Set-Transformer attention-pool scoring critic (Phase 4.4)."""

import numpy as np
import torch

from src.ns3env.realtime_env import PathState
from src.rl.path_agent import PathAgent
from src.rl.sac_agent import SACConfig
from src.rl.scoring_sac_agent import (
    ScoringAttnSACAgent,
    ScoringQNetwork,
    ScoringSACAgent,
    _MaskedSAB,
)

G, F = 4, 6


def _state(n, mask=None, rng=None):
    rng = rng or np.random.default_rng(0)
    m = np.ones(n, dtype=np.float32) if mask is None else np.asarray(mask, np.float32)
    return PathState(
        glob=rng.random(G).astype(np.float32),
        paths=rng.random((n, F)).astype(np.float32),
        mask=m,
    )


def test_mean_pool_has_no_attention_modules():
    crit = ScoringQNetwork(G, F, 32)  # default pool="mean"
    assert crit.attn1 is None and crit.attn2 is None


def test_attn_pool_builds_attention_modules():
    crit = ScoringQNetwork(G, F, 32, pool="attn")
    assert isinstance(crit.attn1, _MaskedSAB) and isinstance(crit.attn2, _MaskedSAB)


def test_attn_critic_permutation_invariance():
    torch.manual_seed(0)
    crit = ScoringQNetwork(G, F, 32, pool="attn")
    n = 5
    glob = torch.rand(1, G)
    paths = torch.rand(1, n, F)
    mask = torch.ones(1, n)
    act = torch.rand(1, n) * 2 - 1
    q1, _ = crit(glob, paths, mask, act)
    perm = torch.randperm(n)
    q1p, _ = crit(glob, paths[:, perm], mask[:, perm], act[:, perm])
    assert torch.allclose(q1, q1p, atol=1e-4)  # still permutation-invariant


def test_attn_critic_ignores_inactive_paths():
    torch.manual_seed(1)
    crit = ScoringQNetwork(G, F, 32, pool="attn")
    glob = torch.rand(1, G)
    paths = torch.rand(1, 5, F)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    act = torch.rand(1, 5) * 2 - 1
    q_a, _ = crit(glob, paths, mask, act)
    # Scribbling on the masked-off rows must not change Q.
    paths2 = paths.clone()
    paths2[:, 3:] = torch.rand(1, 2, F) * 100.0
    q_b, _ = crit(glob, paths2, mask, act)
    assert torch.allclose(q_a, q_b, atol=1e-4)


def test_sab_couples_active_paths():
    # The SAB makes a path's embedding depend on the *other* active paths — the
    # inter-path coupling the masked-mean pool lacks per-path.
    torch.manual_seed(2)
    sab = _MaskedSAB(16, n_heads=4)
    emb = torch.rand(1, 4, 16)
    mask = torch.ones(1, 4)
    out0 = sab(emb, mask)
    emb2 = emb.clone()
    emb2[:, 1] = torch.rand(16)  # perturb path 1 only
    out1 = sab(emb2, mask)
    # Path 0's output row changed because path 1 changed (information flowed).
    assert not torch.allclose(out0[:, 0], out1[:, 0], atol=1e-4)


def test_attn_agent_arch_tag_and_dispatch():
    cfg = SACConfig(start_steps=0, hidden_dim=32, device="cpu")
    agent = ScoringAttnSACAgent(G, F, num_paths=5, config=cfg)
    assert agent.arch == "scoring_attn"
    assert agent.state_dict()["arch"] == "scoring_attn"
    # PathAgent routes the tag to the attention agent.
    pa = PathAgent(0, 5, arch="scoring_attn", global_dim=G, path_dim=F, config=cfg)
    assert isinstance(pa.sac, ScoringAttnSACAgent)
    split, _ = pa.select(_state(5), deterministic=True)
    assert abs(split.sum() - 1.0) < 1e-5


def test_attn_agent_learns_step_and_checkpoint_roundtrip():
    cfg = SACConfig(start_steps=0, update_after=8, batch_size=8, buffer_size=1000,
                    hidden_dim=32, device="cpu")
    agent = ScoringAttnSACAgent(G, F, num_paths=5, config=cfg)
    rng = np.random.default_rng(1)
    for _ in range(40):
        m = (rng.random(5) > 0.3).astype(np.float32)
        m[0] = 1.0
        s, ns = _state(5, m, rng), _state(5, m, rng)
        a = agent.select_action(s.glob, s.paths, s.mask)
        agent.store(s, a, float(rng.random()), ns, False)
    losses = agent.update()
    assert losses is not None and np.isfinite(losses["critic_loss"])

    # A scoring_attn checkpoint loads into a fresh scoring_attn agent.
    a2 = ScoringAttnSACAgent(G, F, num_paths=5, config=cfg)
    a2.load_state_dict(agent.state_dict())
    s = _state(5)
    o1 = agent.select_action(s.glob, s.paths, s.mask, deterministic=True)
    o2 = a2.select_action(s.glob, s.paths, s.mask, deterministic=True)
    assert np.allclose(o1, o2, atol=1e-5)


def test_mean_scoring_still_defaults_and_works():
    # Byte-identical default: plain ScoringSACAgent stays pool="mean", arch="scoring".
    agent = ScoringSACAgent(G, F, num_paths=4, config=SACConfig(start_steps=0, device="cpu"))
    assert agent.arch == "scoring" and agent.pool == "mean"
    assert agent.critic.attn1 is None
