"""Tests for the scoring (dynamic-input) path agent and structured buffer."""

import numpy as np
import torch

from src.rl.replay_buffer import StructuredReplayBuffer
from src.rl.sac_agent import SACConfig
from src.rl.scoring_sac_agent import ScoringGaussianPolicy, ScoringQNetwork, ScoringSACAgent
from src.rl.path_agent import PathAgent
from src.ns3env.realtime_env import PathState

G, F = 4, 6


def _state(n, mask=None, rng=None):
    rng = rng or np.random.default_rng(0)
    m = np.ones(n, dtype=np.float32) if mask is None else np.asarray(mask, np.float32)
    return PathState(
        glob=rng.random(G).astype(np.float32),
        paths=rng.random((n, F)).astype(np.float32),
        mask=m,
    )


def test_policy_variable_n_shapes():
    pol = ScoringGaussianPolicy(G, F, 32)
    for n in (2, 4, 7):
        glob = torch.rand(3, G)
        paths = torch.rand(3, n, F)
        mask = torch.ones(3, n)
        a, logp, mean = pol.sample(glob, paths, mask)
        assert a.shape == (3, n)
        assert logp.shape == (3, 1)
        assert mean.shape == (3, n)


def test_masked_split_zeros_inactive():
    agent = PathAgent(
        obs_dim=0, num_paths=5, arch="scoring", global_dim=G, path_dim=F,
        config=SACConfig(start_steps=0),
    )
    mask = np.array([1, 0, 1, 0, 1], dtype=np.float32)
    split, raw = agent.select(_state(5, mask=mask), deterministic=True)
    assert split.shape == (5,)
    assert abs(split.sum() - 1.0) < 1e-5
    # Inactive paths get exactly zero weight.
    assert split[1] == 0.0 and split[3] == 0.0
    assert (split[[0, 2, 4]] > 0).all()


def test_critic_permutation_invariance():
    torch.manual_seed(0)
    crit = ScoringQNetwork(G, F, 32)
    n = 5
    glob = torch.rand(1, G)
    paths = torch.rand(1, n, F)
    mask = torch.ones(1, n)
    act = torch.rand(1, n) * 2 - 1
    q1, _ = crit(glob, paths, mask, act)
    perm = torch.randperm(n)
    q1p, _ = crit(glob, paths[:, perm], mask[:, perm], act[:, perm])
    assert torch.allclose(q1, q1p, atol=1e-5)


def test_structured_buffer_roundtrip():
    buf = StructuredReplayBuffer(G, F, 5, capacity=10)
    s, ns = _state(5), _state(5)
    act = np.random.uniform(-1, 1, 5).astype(np.float32)
    buf.push(s.glob, s.paths, s.mask, act, 0.7, ns.glob, ns.paths, ns.mask, False)
    assert len(buf) == 1
    b = buf.sample(1)
    assert b["paths"].shape == (1, 5, F)
    assert b["mask"].shape == (1, 5)
    assert np.allclose(b["act"][0], act)
    assert b["rew"][0, 0] == np.float32(0.7)


def test_scoring_agent_learns_step():
    cfg = SACConfig(start_steps=0, update_after=8, batch_size=8, buffer_size=1000)
    agent = ScoringSACAgent(G, F, num_paths=5, config=cfg)
    rng = np.random.default_rng(1)
    # Fill the buffer with varied masks (variable active counts).
    for _ in range(40):
        m = (rng.random(5) > 0.3).astype(np.float32)
        m[0] = 1.0  # keep at least one live
        s, ns = _state(5, m, rng), _state(5, m, rng)
        a = agent.select_action(s.glob, s.paths, s.mask)
        agent.store(s, a, float(rng.random()), ns, False)
    losses = agent.update()
    assert losses is not None
    assert np.isfinite(losses["critic_loss"]) and np.isfinite(losses["policy_loss"])


def test_scoring_checkpoint_roundtrip():
    cfg = SACConfig(start_steps=0)
    a1 = ScoringSACAgent(G, F, num_paths=4, config=cfg)
    a2 = ScoringSACAgent(G, F, num_paths=4, config=cfg)
    a2.load_state_dict(a1.state_dict())
    s = _state(4)
    o1 = a1.select_action(s.glob, s.paths, s.mask, deterministic=True)
    o2 = a2.select_action(s.glob, s.paths, s.mask, deterministic=True)
    assert np.allclose(o1, o2, atol=1e-5)
