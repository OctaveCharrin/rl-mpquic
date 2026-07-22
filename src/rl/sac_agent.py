"""
Generic continuous Soft Actor-Critic (SAC).

A self-contained SAC operating in a **normalized action space** ``[-1, 1]^d``:
the actor is a tanh-squashed diagonal Gaussian, with twin Q critics + Polyak
targets and automatic entropy-temperature tuning. The App and Path agents
(:mod:`src.rl.app_agent`, :mod:`src.rl.path_agent`) wrap this and map the
normalized action into their own range (a target bitrate, or a softmax split).

Off-policy + sample-efficient: each environment frame contributes one transition
and (optionally) one gradient step, which matters because NS-3 frames are
expensive. References: Haarnoja et al., "Soft Actor-Critic" (2018).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

_LOG_STD_MIN = -20.0
_LOG_STD_MAX = 2.0
_EPS = 1e-6


@dataclass
class SACConfig:
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 200_000
    start_steps: int = 1_000      # uniform-random actions before this many stores
    update_after: int = 1_000     # min transitions before gradient updates begin
    updates_per_step: int = 1     # UTD ratio; raise only with critic_layernorm
    auto_entropy: bool = True
    alpha: float = 0.2            # fixed temperature if auto_entropy is False
    device: Optional[str] = None  # "cpu" / "cuda"; auto-selected if None
    # DroQ recipe (P1): normalize + regularize the critic so a high UTD ratio
    # stays stable. Both default off => behavior byte-identical to plain SAC.
    critic_layernorm: bool = False
    critic_dropout: float = 0.0
    # Prioritized Experience Replay (P2). Off => uniform sampling, byte-identical.
    prioritized: bool = False
    per_alpha: float = 0.6        # priority exponent (0 = uniform)
    per_beta0: float = 0.4        # initial IS-correction exponent
    per_beta_steps: int = 100_000  # anneal beta 0->1 over this many updates


def _mlp(
    in_dim: int,
    hidden: int,
    out_dim: int,
    *,
    layernorm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    """MLP with optional per-hidden-layer LayerNorm + Dropout (DroQ critic)."""
    layers: list[nn.Module] = []
    dims = (in_dim, hidden, hidden)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layernorm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Tanh-squashed diagonal-Gaussian policy over ``[-1, 1]^act_dim``."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), _LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """Return (action, log_prob, tanh(mean)) with the tanh log-det correction."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + _EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class QNetwork(nn.Module):
    """Twin Q networks Q1, Q2 over (obs, action)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int,
        *,
        layernorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.q1 = _mlp(obs_dim + act_dim, hidden, 1, layernorm=layernorm, dropout=dropout)
        self.q2 = _mlp(obs_dim + act_dim, hidden, 1, layernorm=layernorm, dropout=dropout)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    """SAC over a normalized ``[-1, 1]`` action space."""

    def __init__(self, obs_dim: int, act_dim: int, config: Optional[SACConfig] = None):
        self.cfg = config or SACConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(
            self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy = GaussianPolicy(obs_dim, act_dim, self.cfg.hidden_dim).to(self.device)
        q_kwargs = dict(
            layernorm=self.cfg.critic_layernorm, dropout=self.cfg.critic_dropout
        )
        self.critic = QNetwork(obs_dim, act_dim, self.cfg.hidden_dim, **q_kwargs).to(self.device)
        self.critic_target = QNetwork(
            obs_dim, act_dim, self.cfg.hidden_dim, **q_kwargs
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

        if self.cfg.auto_entropy:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)
        else:
            self.log_alpha = torch.log(torch.tensor(self.cfg.alpha, device=self.device))

        if self.cfg.prioritized:
            self.buffer = PrioritizedReplayBuffer(
                obs_dim, act_dim, self.cfg.buffer_size, alpha=self.cfg.per_alpha
            )
        else:
            self.buffer = ReplayBuffer(obs_dim, act_dim, self.cfg.buffer_size)
        self._stores = 0
        self._updates = 0  # gradient-step counter (drives PER beta annealing)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # -- interaction -------------------------------------------------------- #

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return an action in ``[-1, 1]^act_dim``.

        Before ``start_steps`` transitions are collected, acts uniformly at
        random (exploration warm-up). Deterministic mode returns ``tanh(mean)``.
        """
        if not deterministic and self._stores < self.cfg.start_steps:
            return np.random.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean = self.policy.sample(obs_t)
            out = mean if deterministic else action
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(obs, act, rew, next_obs, done)
        self._stores += 1

    def ready(self) -> bool:
        return len(self.buffer) >= max(self.cfg.batch_size, self.cfg.update_after)

    def update(self) -> Optional[Dict[str, float]]:
        """Run ``updates_per_step`` gradient steps; return the last losses."""
        if not self.ready():
            return None
        losses: Optional[Dict[str, float]] = None
        for _ in range(self.cfg.updates_per_step):
            losses = self._update_once()
        return losses

    # -- learning ----------------------------------------------------------- #

    def _beta(self) -> float:
        """PER importance-sampling exponent, annealed ``per_beta0`` -> 1.0."""
        frac = min(1.0, self._updates / max(1, self.cfg.per_beta_steps))
        return self.cfg.per_beta0 + frac * (1.0 - self.cfg.per_beta0)

    def _update_once(self) -> Dict[str, float]:
        self._updates += 1
        if self.cfg.prioritized:
            obs, act, rew, next_obs, done, indices, weights = self.buffer.sample(
                self.cfg.batch_size, self._beta()
            )
            weights = torch.as_tensor(weights, device=self.device)
        else:
            obs, act, rew, next_obs, done = self.buffer.sample(self.cfg.batch_size)
            indices, weights = None, None
        obs = torch.as_tensor(obs, device=self.device)
        act = torch.as_tensor(act, device=self.device)
        rew = torch.as_tensor(rew, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        # --- critic ---
        with torch.no_grad():
            next_act, next_logp, _ = self.policy.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target = rew + self.cfg.gamma * (1.0 - done) * q_t

        q1, q2 = self.critic(obs, act)
        if self.cfg.prioritized:
            td1, td2 = q1 - target, q2 - target
            critic_loss = (weights * (td1.pow(2) + td2.pow(2))).mean()
        else:
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.cfg.prioritized:
            with torch.no_grad():
                td = torch.maximum((q1 - target).abs(), (q2 - target).abs())
            self.buffer.update_priorities(indices, td.squeeze(-1).cpu().numpy())

        # --- actor ---
        new_act, logp, _ = self.policy.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_act)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # --- temperature ---
        alpha_loss_val = 0.0
        if self.cfg.auto_entropy:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = float(alpha_loss.item())

        # --- target Polyak update ---
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p)

        return {
            "critic_loss": float(critic_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha_loss": alpha_loss_val,
            "alpha": float(self.alpha.item()),
        }

    # -- checkpoint --------------------------------------------------------- #

    def state_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, sd: Dict[str, object]) -> None:
        self.policy.load_state_dict(sd["policy"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(sd["log_alpha"], device=self.device))

    # -- replay-buffer persistence (P3) ------------------------------------- #

    def save_buffer(self, path: str) -> None:
        """Persist the replay buffer + ``_stores``/``_updates`` counters to ``path``."""
        self.buffer.save(path, stores=self._stores, updates=self._updates)

    def load_buffer(self, path: str) -> None:
        """Restore the replay buffer and the ``_stores``/``_updates`` counters.

        Restoring ``_updates`` keeps PER's beta annealing (see ``_beta``) on its
        original schedule across ``--resume`` instead of restarting from
        ``per_beta0`` — i.e. a fully faithful resume.
        """
        self._stores, self._updates = self.buffer.load(path)
