"""
Scoring (dynamic-input) Soft Actor-Critic for the per-path traffic split.

Where :class:`~src.rl.sac_agent.SACAgent` is a flat-vector SAC locked to a fixed
path count, this variant is **permutation-equivariant** and handles a *variable,
changing* set of paths — the architecture the SCION sibling's path-scoring DQN
uses, adapted from discrete-argmax to continuous SAC.

State is structured: a global context vector ``glob (G,)``, a per-path feature
matrix ``paths (N, F)``, and a liveness ``mask (N,)``. The path count ``N`` is the
candidate cap (fixed for tensor shapes); the mask says which rows are live.

* **Actor** (:class:`ScoringGaussianPolicy`): a *shared* per-path encoder maps
  ``glob ⊕ path_i`` to a tanh-squashed Gaussian latent per path. The latent is the
  SAC action (in ``[-1, 1]^N``, exactly like the flat agent's raw action); the env
  split is a masked softmax of it over the *active* paths. Log-probabilities and
  entropy are summed over active paths only.
* **Critic** (:class:`ScoringQNetwork`, twin): a DeepSets-style encoder consumes
  ``glob ⊕ path_i ⊕ latent_i`` per path, masked-mean-pools across paths, and maps
  the pooled embedding to a scalar Q — permutation-invariant and variable-N.

Inactive paths contribute to neither the log-prob, the pooled critic embedding,
nor the split, so they receive no gradient. The entropy temperature targets
``-active_count`` per sample (not a fixed ``-N``).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import PrioritizedStructuredReplayBuffer, StructuredReplayBuffer
from .sac_agent import SACConfig

_LOG_STD_MIN = -20.0
_LOG_STD_MAX = 2.0
_EPS = 1e-6
# Large negative fill for masked logits (finite, so softmax/backprop stay clean).
_NEG_INF = -1e30


def _encoder(
    in_dim: int, hidden: int, *, layernorm: bool = False, dropout: float = 0.0
) -> nn.Sequential:
    """Two-layer per-path encoder with optional LayerNorm + Dropout.

    The normalization flags are for the *critic only* (DroQ recipe, P1) — the
    shared policy body always builds a plain encoder (flags default off).
    """
    layers: list[nn.Module] = []
    for d_in in (in_dim, hidden):
        layers.append(nn.Linear(d_in, hidden))
        if layernorm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ScoringGaussianPolicy(nn.Module):
    """Shared per-path tanh-Gaussian policy over a variable path set."""

    def __init__(self, global_dim: int, path_dim: int, hidden: int):
        super().__init__()
        self.body = _encoder(global_dim + path_dim, hidden)
        self.mean = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)

    def forward(self, glob: torch.Tensor, paths: torch.Tensor):
        # glob: (B, G); paths: (B, N, F)  ->  mean, log_std: (B, N)
        b, n, _ = paths.shape
        g = glob.unsqueeze(1).expand(b, n, glob.shape[-1])
        h = self.body(torch.cat([g, paths], dim=-1))
        mean = self.mean(h).squeeze(-1)
        log_std = torch.clamp(self.log_std(h).squeeze(-1), _LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std

    def sample(self, glob: torch.Tensor, paths: torch.Tensor, mask: torch.Tensor):
        """Return (action, log_prob, mean_action) with masked tanh-Gaussian.

        ``action`` is the per-path latent in ``[-1, 1]^N``; ``log_prob`` sums the
        tanh-corrected per-path log-densities over *active* paths only (so dead
        paths add neither density nor entropy).
        """
        mean, log_std = self.forward(glob, paths)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        logp = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + _EPS)  # (B, N)
        logp = (logp * mask).sum(dim=-1, keepdim=True)
        return action, logp, torch.tanh(mean)


class ScoringQNetwork(nn.Module):
    """Twin DeepSets critics over (glob, paths, latent action), masked-mean pooled."""

    def __init__(
        self,
        global_dim: int,
        path_dim: int,
        hidden: int,
        *,
        layernorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_dim = global_dim + path_dim + 1  # + per-path action latent
        enc_kw = dict(layernorm=layernorm, dropout=dropout)
        self.enc1 = _encoder(in_dim, hidden, **enc_kw)
        self.head1 = nn.Linear(hidden, 1)
        self.enc2 = _encoder(in_dim, hidden, **enc_kw)
        self.head2 = nn.Linear(hidden, 1)

    def _q(self, enc, head, x, mask):
        emb = enc(x)  # (B, N, H)
        m = mask.unsqueeze(-1)
        pooled = (emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # masked mean
        return head(pooled)

    def forward(
        self,
        glob: torch.Tensor,
        paths: torch.Tensor,
        mask: torch.Tensor,
        act: torch.Tensor,
    ):
        b, n, _ = paths.shape
        g = glob.unsqueeze(1).expand(b, n, glob.shape[-1])
        x = torch.cat([g, paths, act.unsqueeze(-1)], dim=-1)
        return self._q(self.enc1, self.head1, x, mask), self._q(
            self.enc2, self.head2, x, mask
        )


class ScoringSACAgent:
    """SAC over a structured, variable-path-count action space (per-path split)."""

    arch = "scoring"

    def __init__(
        self,
        global_dim: int,
        path_dim: int,
        num_paths: int,
        config: Optional[SACConfig] = None,
    ):
        self.cfg = config or SACConfig()
        self.global_dim = int(global_dim)
        self.path_dim = int(path_dim)
        self.num_paths = int(num_paths)
        self.device = torch.device(
            self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        h = self.cfg.hidden_dim
        # LayerNorm/dropout go to the critic only, never the shared policy body.
        q_kw = dict(layernorm=self.cfg.critic_layernorm, dropout=self.cfg.critic_dropout)
        self.policy = ScoringGaussianPolicy(global_dim, path_dim, h).to(self.device)
        self.critic = ScoringQNetwork(global_dim, path_dim, h, **q_kw).to(self.device)
        self.critic_target = ScoringQNetwork(global_dim, path_dim, h, **q_kw).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

        if self.cfg.auto_entropy:
            # Per-path target entropy; the per-sample target scales with the
            # active-path count in the alpha loss (see _update_once).
            self.target_entropy_per_path = -1.0
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)
        else:
            self.log_alpha = torch.log(torch.tensor(self.cfg.alpha, device=self.device))

        if self.cfg.prioritized:
            self.buffer = PrioritizedStructuredReplayBuffer(
                global_dim, path_dim, num_paths, self.cfg.buffer_size,
                alpha=self.cfg.per_alpha,
            )
        else:
            self.buffer = StructuredReplayBuffer(
                global_dim, path_dim, num_paths, self.cfg.buffer_size
            )
        self._stores = 0
        self._updates = 0  # gradient-step counter (drives PER beta annealing)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # -- interaction -------------------------------------------------------- #

    def select_action(
        self, glob: np.ndarray, paths: np.ndarray, mask: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Per-path latent action in ``[-1, 1]^N`` (uniform-random during warm-up)."""
        if not deterministic and self._stores < self.cfg.start_steps:
            return np.random.uniform(-1.0, 1.0, size=self.num_paths).astype(np.float32)
        g = torch.as_tensor(glob, dtype=torch.float32, device=self.device).unsqueeze(0)
        p = torch.as_tensor(paths, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean = self.policy.sample(g, p, m)
            out = mean if deterministic else action
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    def store(self, state, act, rew, next_state, done) -> None:
        self.buffer.push(
            state.glob, state.paths, state.mask, act, rew,
            next_state.glob, next_state.paths, next_state.mask, done,
        )
        self._stores += 1

    def ready(self) -> bool:
        return len(self.buffer) >= max(self.cfg.batch_size, self.cfg.update_after)

    def update(self) -> Optional[Dict[str, float]]:
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
            batch = self.buffer.sample(self.cfg.batch_size, self._beta())
            indices = batch["indices"]
            weights = torch.as_tensor(batch["weights"], device=self.device)
        else:
            batch = self.buffer.sample(self.cfg.batch_size)
            indices, weights = None, None
        t = lambda k: torch.as_tensor(batch[k], device=self.device)  # noqa: E731
        glob, paths, mask, act = t("glob"), t("paths"), t("mask"), t("act")
        rew, done = t("rew"), t("done")
        n_glob, n_paths, n_mask = t("next_glob"), t("next_paths"), t("next_mask")

        # --- critic ---
        with torch.no_grad():
            next_act, next_logp, _ = self.policy.sample(n_glob, n_paths, n_mask)
            q1_t, q2_t = self.critic_target(n_glob, n_paths, n_mask, next_act)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target = rew + self.cfg.gamma * (1.0 - done) * q_t

        q1, q2 = self.critic(glob, paths, mask, act)
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
        new_act, logp, _ = self.policy.sample(glob, paths, mask)
        q1_pi, q2_pi = self.critic(glob, paths, mask, new_act)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # --- temperature (target entropy scales with active path count) ---
        alpha_loss_val = 0.0
        if self.cfg.auto_entropy:
            target_entropy = self.target_entropy_per_path * mask.sum(dim=1, keepdim=True)
            alpha_loss = -(self.log_alpha * (logp + target_entropy).detach()).mean()
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
            "arch": self.arch,
            "num_paths": self.num_paths,
            "global_dim": self.global_dim,
            "path_dim": self.path_dim,
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
