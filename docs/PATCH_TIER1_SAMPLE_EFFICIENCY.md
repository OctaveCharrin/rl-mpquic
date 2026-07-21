# Patch Plan — Tier 1 Sample-Efficiency Upgrades

**Audience:** a Claude instance (or engineer) implementing these changes later.
This document is a self-contained work order. Read it top to bottom, then
implement the three patches in the stated order. It assumes the code as of the
current `main`; line numbers are anchors, verify before editing.

**Rationale & citations:** see `docs/RL_AGENT_DESIGN.md` §3 Tier 1. Summary:
NS-3 frames are expensive and the current SAC does only `updates_per_step = 1`,
under-using its off-policy advantage. These three patches extract more learning
per frame (P1 UTD + critic normalization), focus learning on rare critical
events (P2 prioritized replay), and stop discarding warm data on resume (P3).

**Scope:** `src/rl/sac_agent.py`, `src/rl/scoring_sac_agent.py`,
`src/rl/replay_buffer.py`, `src/train/config.py`,
`src/train/hierarchical_train.py`, `configs/*.yaml`, and tests under `tests/`.

**Contract to preserve:** both agents must stay behaviorally identical when the
new config flags are left at their defaults that reproduce today's behavior
(`critic_layernorm: false`, `critic_dropout: 0.0`, `prioritized: false`,
`updates_per_step: 1`, buffer persistence off). Every patch is gated so a stock
config is byte-compatible. `uv run pytest` must stay green throughout.

**Implementation order:** P1 → P3 → P2. P1 and P3 are localized and low-risk; P2
(prioritized replay) is the most invasive and depends on the `_update_once` loss
structure P1 leaves in place. **Do not raise UTD without P1's critic
normalization** — high UTD on an unnormalized critic diverges.

---

## Patch P1 — Critic normalization + higher update-to-data ratio

**Goal.** Add optional LayerNorm (and optional Dropout) to the *critic* networks
only, so training is stable at `updates_per_step` (UTD) of 10–20. This is the
DroQ recipe (Hiraoka et al., 2022). `updates_per_step` already loops in
`update()` (`sac_agent.py:170`), so raising UTD needs no new loop — only the
critic must be regularized.

### P1.1 — `SACConfig` new fields (`sac_agent.py:33`)

Add three fields (keep defaults reproducing current behavior):

```python
    critic_layernorm: bool = False   # DroQ-style LayerNorm on critic hidden layers
    critic_dropout: float = 0.0      # DroQ dropout prob on critic hidden layers (0 = off)
    # updates_per_step already exists; raise it in YAML (e.g. 10 or 20) with the above on.
```

### P1.2 — Make `_mlp` normalization-aware (`sac_agent.py:48`)

`_mlp` is used **only** by `QNetwork` (the critic) — `GaussianPolicy` builds its
own body — so editing `_mlp` touches critics only. Replace with:

```python
def _mlp(in_dim, hidden, out_dim, *, layernorm=False, dropout=0.0):
    def block(i, o):
        layers = [nn.Linear(i, o)]
        if layernorm:
            layers.append(nn.LayerNorm(o))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        return layers
    return nn.Sequential(*block(in_dim, hidden), *block(hidden, hidden),
                         nn.Linear(hidden, out_dim))
```

### P1.3 — Thread flags into `QNetwork` and `SACAgent` (`sac_agent.py:87`, `:112`)

- `QNetwork.__init__` takes `layernorm=False, dropout=0.0` and forwards both to
  the two `_mlp(...)` calls.
- In `SACAgent.__init__`, pass `layernorm=self.cfg.critic_layernorm,
  dropout=self.cfg.critic_dropout` when building **both** `self.critic` and
  `self.critic_target`.

### P1.4 — Scoring agent critic normalization (`scoring_sac_agent.py:46`, `:90`)

`_encoder` is shared by the actor (`ScoringGaussianPolicy.body`) *and* the critic
(`ScoringQNetwork.enc1/enc2`), so **do not** blanket-edit it. Instead:

- Add `layernorm=False, dropout=0.0` params to `_encoder`, applied after each
  ReLU (same block pattern as P1.2).
- Pass them **only** from `ScoringQNetwork` (both `enc1`/`enc2`), never from the
  policy body.
- `ScoringSACAgent.__init__` forwards `self.cfg.critic_layernorm/critic_dropout`
  when constructing `self.critic` and `self.critic_target`.

### P1.5 — Dropout / train-mode subtlety (only if `critic_dropout > 0`)

DroQ keeps dropout **active during the target computation** (it is part of the
regularization, not just train-time). The critic modules are created in default
`train()` mode and never switched to `eval()`, and the target Q is computed under
`torch.no_grad()` but with modules still in train mode — so dropout is already
active there. **Action:** do nothing special, but add a one-line comment in
`_update_once` noting the target uses the critic in train mode by design. If
`critic_dropout == 0.0` (the recommended default), this is moot; LayerNorm behaves
identically in train/eval.

### P1.6 — Config loader (`config.py:189`)

Add to the `SACConfig(...)` call:

```python
        critic_layernorm=bool(sac.get("critic_layernorm", False)),
        critic_dropout=float(sac.get("critic_dropout", 0.0)),
```

### P1.7 — Config YAML

Leave `configs/default.yaml`, `smoke.yaml`, `four_path.yaml` unchanged (defaults
off = current behavior). In `configs/dynamic.yaml` `sac:` block, enable the
recipe:

```yaml
  updates_per_step: 10       # UTD; raise learning-per-frame (was 1)
  critic_layernorm: true     # DroQ-style stabilizer, required at high UTD
  # critic_dropout: 0.01     # optional; leave 0 unless instability persists
```

### P1.8 — Wall-clock note

UTD=10 means 10× the gradient steps per frame. On **mock** this multiplies
training time roughly linearly; on **NS-3** the simulator dominates, so extra
gradient steps are comparatively cheap — the intended regime. Expose UTD via YAML
so it can be tuned per backend.

### P1 acceptance

- `uv run pytest` green with defaults (flags off ⇒ identical nets: assert
  `QNetwork` with `layernorm=False` has the same module list length as before).
- New unit test: build a `SACAgent` with `critic_layernorm=True`, assert critic
  `state_dict` contains `LayerNorm` params and the policy `state_dict` does **not**
  (normalization stayed critic-only).
- Smoke train: `uv run python train.py --config configs/dynamic.yaml --backend
  mock --episodes 5` runs without NaNs at UTD=10.

---

## Patch P3 — Persist replay buffers across `--resume`

**Goal.** `hierarchical_train.py:121` currently reloads network weights on resume
but the replay buffer "restarts empty," and it papers over this by forcing
`_stores = start_steps` to skip warm-up — so a resumed run learns from a cold,
empty buffer. Persist and restore the buffers so chunked/interruptible runs keep
their experience. Make it **opt-in** (`--persist-buffer`) because the buffer is
large and slow to write.

### P3.1 — Buffer save/load (`replay_buffer.py`)

Add to `ReplayBuffer`:

```python
    def save(self, path):
        np.savez(path, obs=self.obs, next_obs=self.next_obs, act=self.act,
                 rew=self.rew, done=self.done,
                 ptr=self._ptr, size=self._size)

    def load(self, path):
        d = np.load(path)
        for k in ("obs", "next_obs", "act", "rew", "done"):
            getattr(self, k)[...] = d[k]
        self._ptr, self._size = int(d["ptr"]), int(d["size"])
```

Add the analogous `save`/`load` to `StructuredReplayBuffer` covering
`glob, paths, mask, act, rew, next_glob, next_paths, next_mask, done` + `ptr/size`.
Guard `load` against a capacity/shape mismatch (raise a clear error if the saved
arrays don't match the constructed buffer, so a config change fails loudly rather
than corrupting state).

### P3.2 — Expose the buffer on the agents

Both `SACAgent` and `ScoringSACAgent` already hold `self.buffer` and `self._stores`.
Add thin passthroughs (or just access `agent.sac.buffer` from the trainer):
`save_buffer(path)` / `load_buffer(path)` that also persist `_stores` (put it in
the npz, or save a tiny sidecar). Restoring `_stores` is what lets us **drop the
`_stores = start_steps` hack** and resume warm-up state honestly.

### P3.3 — Trainer wiring (`hierarchical_train.py`)

- Add a `persist_buffer: bool = False` parameter to `run_training` and a
  `--persist-buffer` flag in `train.py`.
- In `_save_ckpts()` (`:116`): if `persist_buffer`, also write
  `app_buffer.npz` / `path_buffer.npz` in `out_dir`. **Do not** write these every
  episode by default (200k×dims floats is slow) — write them only at the
  `finally`/end and on the *last* checkpoint, or every `K` episodes behind a
  counter. Document the chosen cadence in the function docstring.
- In the resume block (`:123`): if `persist_buffer` and the `*_buffer.npz` exist,
  `load` them into the agents and restore `_stores`; **only then** skip the
  `_stores = start_steps` fallback. If the buffers are absent, keep today's
  behavior (empty buffer + skip warm-up) for backward compatibility.

### P3 acceptance

- Round-trip test: fill a buffer, `save`, construct a fresh buffer, `load`, assert
  arrays and `_stores`/`ptr`/`size` are identical; `sample()` works.
- Resume test: train 3 episodes with `--persist-buffer`, resume 3 more, assert the
  path agent's `len(buffer) > 0` immediately after resume (was 0 before).
- Shape-mismatch test: saving with `num_paths=3` then loading into a `num_paths=6`
  buffer raises a clear error.

---

## Patch P2 — Prioritized Experience Replay (PER)

**Goal.** Sample transitions with probability ∝ TD-error^α so the rare critical
events under `configs/dynamic.yaml` (churn-out, regime swaps, correlated
failures) are replayed more often (Schaul et al., 2016). Most invasive patch —
touches both buffers and both `_update_once`. Gated by `prioritized: false`
default, so stock runs are unchanged.

### P2.1 — Config fields (`sac_agent.py:33`, loader `config.py:189`, YAML)

```python
    prioritized: bool = False
    per_alpha: float = 0.6       # priority exponent (0 = uniform)
    per_beta0: float = 0.4       # initial importance-sampling correction
    per_beta_steps: int = 100_000  # linear anneal beta 0.4 -> 1.0 over N updates
```
Add matching `sac.get(...)` lines in the loader; enable in `dynamic.yaml` only.

### P2.2 — Sum-tree prioritized buffer (`replay_buffer.py`)

Implement priorities with a **sum-tree** for O(log n) proportional sampling
(a linear `np.random.choice(p=...)` over a 200k buffer per update is too slow at
high UTD). Two options — pick one and document it:

- **(Recommended) Add a `PrioritizedReplayBuffer` and a
  `PrioritizedStructuredReplayBuffer`** subclassing the existing buffers, adding a
  sum-tree of size `capacity`. Keep the uniform classes intact so `prioritized:
  false` uses them unchanged.

New/overridden methods on the prioritized variants:

```python
    def push(self, *args):
        # store transition as today, then set new leaf priority = current max
        # priority (so fresh transitions are seen at least once).
    def sample(self, batch_size, beta):
        # sample `batch_size` leaves ∝ priority^alpha via the sum-tree;
        # return (batch..., indices, is_weights) where
        #   P(i) = p_i^alpha / sum_j p_j^alpha
        #   w_i  = (N * P(i))^(-beta),  normalized by max w  -> in (0, 1]
    def update_priorities(self, indices, td_errors):
        # p_i = |td_error_i| + eps    (eps ~ 1e-6)
```

For the structured buffer, `sample` returns the same dict as today plus
`indices` and `weights` keys.

### P2.3 — Agent selection of buffer + beta schedule

- In each agent `__init__`, choose the prioritized buffer when
  `self.cfg.prioritized`, else the existing uniform buffer.
- Add an update counter (e.g. `self._updates`) and a `_beta()` helper that
  linearly anneals `per_beta0 → 1.0` over `per_beta_steps`.

### P2.4 — Weighted critic loss (`sac_agent.py:176`, `scoring_sac_agent.py:207`)

This is the core change. Replace the unweighted critic loss with a per-sample,
IS-weighted loss and feed TD errors back:

```python
# uniform path (prioritized: false) keeps today's code exactly.
# prioritized path:
q1, q2 = self.critic(obs, act)                    # (B,1)
td1 = q1 - target                                 # per-sample errors
td2 = q2 - target
w = torch.as_tensor(is_weights, device=self.device)  # (B,1)
critic_loss = (w * (td1.pow(2) + td2.pow(2))).mean()
...
# after critic step, update priorities with the (detached) larger error:
new_p = torch.max(td1.abs(), td2.abs()).detach().squeeze(-1).cpu().numpy()
self.buffer.update_priorities(indices, new_p)
```

Keep the actor and temperature losses unchanged (PER corrects the critic
regression only). Branch on `self.cfg.prioritized` so the uniform code path is
untouched. Do the identical change in `ScoringSACAgent._update_once`, threading
`indices`/`weights` out of the dict returned by the structured `sample`.

### P2.5 — Subtleties to get right

- **IS-weight normalization:** divide weights by their max in the batch so they
  scale the loss *down*, never up (stability).
- **New-leaf priority = running max** so every transition is sampled at least
  once before its true TD error is known.
- **`eps`** added to priorities so a zero-TD transition can still be resampled.
- **P1 interaction:** with `critic_dropout > 0`, the TD error used for priority is
  noisy across passes — acceptable, but prefer `critic_dropout = 0` (LayerNorm
  only) when PER is on to keep priorities stable.

### P2 acceptance

- Unit: after pushing transitions with a known large error on one index and
  calling `update_priorities`, that index's sampling frequency over many draws is
  significantly above uniform.
- Unit: IS weights are in `(0, 1]` and equal `1` for the max-priority sample.
- `prioritized: false` ⇒ `_update_once` numerically identical to today (seed a
  batch, compare critic loss before/after the refactor).
- Smoke train on `dynamic.yaml` with `prioritized: true` runs without NaNs.

---

## Final step — Update the documentation

After P1/P2/P3 land and validation passes, update the docs so the repo describes
the *new* behavior. Do this last (once the code is settled) so docs don't drift
from a half-finished patch. Concretely:

- **`CLAUDE.md`** — the layering/`src/rl/` bullet and the "Dev loop" section:
  note the new `sac:` knobs (`updates_per_step` UTD, `critic_layernorm`,
  `critic_dropout`, `prioritized` + `per_*`) and, if P3 shipped, that
  `--resume` now restores the replay buffer with `--persist-buffer`. Keep the
  "off by default = byte-identical" contract language consistent with the rest of
  the file.
- **`docs/RL_AGENT_DESIGN.md`** — move the implemented items in §3 Tier 1 from
  "suggested" to "done": either strike them through with a short "✅ implemented
  in P1/P2/P3 — see config keys" note, or add an "Implemented" subsection. Update
  §2.1 (SAC justification) to mention the critic normalization now in place, and
  §4 "Recommended first step" so it no longer points at work already done.
- **`docs/IMPLEMENTATION.md`** — §2.2 "Generic SAC core" and the
  `updates_per_step`/replay-buffer description (around the lines mentioning
  `update_after`/`updates_per_step` and the buffer): document the normalization
  layers, the UTD default, the prioritized buffer path, and buffer persistence.
- **`docs/ARCHITECTURE.md`** — §7.1 "Why SAC" and the replay-buffer mentions:
  add a sentence on DroQ-style critic normalization + PER as the sample-efficiency
  measures, with the same citations.
- **`configs/dynamic.yaml`** — ensure the inline comments on the new `sac:` keys
  explain what each does (UTD, LayerNorm, PER), matching the style of the existing
  reward-block comments.
- **This file** — tick the "Files touched" checklist below, and add a short
  "Status: implemented on <branch/commit>" line at the top so a later reader knows
  the plan was executed.

Do **not** edit `docs/REWARD_TUNING.md` or `docs/SLIDES.md` (unrelated), and only
touch `docs/TUNING_DYNAMICS.md` if a new knob changes dynamics-tuning guidance.

## Combined validation protocol

1. **Unit tests:** all of the P1/P2/P3 acceptance tests above, added under
   `tests/`. `uv run pytest` green.
2. **Backward-compat proof:** run `uv run python train.py --config
   configs/default.yaml --backend mock --episodes 5` with all flags at defaults
   and confirm results match pre-patch (same seed ⇒ same first-episode stats;
   the default code paths are unchanged).
3. **Ablation A/B:** on `configs/dynamic.yaml`, mock backend, ≥3 seeds each:
   - baseline (`updates_per_step=1`, all flags off),
   - +P1 (`updates_per_step=10`, `critic_layernorm=true`),
   - +P1+P2 (`prioritized=true`).
   Compare `app_reward_mean` / `path_reward_mean` / `loss_mean` from
   `stats.json`, and run `uv run python evaluate.py --config configs/dynamic.yaml
   --backend mock --app runs/<run>/app.pth --path runs/<run>/path.pth --ablation`.
   Expectation: P1 improves sample efficiency (higher reward at equal episodes);
   P2 further reduces `loss_mean` under churn/correlated failures.
4. **NS-3 sanity (optional):** one short `--backend ns3` run to confirm the higher
   UTD does not stall the bridge (it should not — gradient steps are Python-side,
   between frames).

## Files touched (checklist)

- [ ] `src/rl/sac_agent.py` — `SACConfig` fields; `_mlp` norm; `QNetwork`/agent wiring; weighted loss (P2).
- [ ] `src/rl/scoring_sac_agent.py` — `_encoder` norm (critic-only); agent wiring; weighted loss (P2).
- [ ] `src/rl/replay_buffer.py` — `save`/`load` (P3); prioritized buffers + sum-tree (P2).
- [ ] `src/train/config.py` — load new `sac:` keys (P1, P2).
- [ ] `src/train/hierarchical_train.py` — buffer persistence + resume wiring (P3).
- [ ] `train.py` — `--persist-buffer` flag (P3).
- [ ] `configs/dynamic.yaml` — enable UTD/LayerNorm (P1), optionally PER (P2); comment the new keys.
- [ ] `tests/` — acceptance tests for P1, P2, P3.
- [ ] **Docs (final step):** `CLAUDE.md`, `docs/RL_AGENT_DESIGN.md`,
      `docs/IMPLEMENTATION.md`, `docs/ARCHITECTURE.md` — reflect the shipped
      behavior; add a "Status: implemented" line to this file.

## References

See `docs/RL_AGENT_DESIGN.md` §References. Directly relevant here:
- T. Hiraoka et al., "Dropout Q-Functions for Doubly Efficient RL" (DroQ), ICLR
  2022. <https://arxiv.org/abs/2110.02034>
- A. Bhatt et al., "CrossQ," ICLR 2024 (BatchNorm alternative, no target net).
  <https://arxiv.org/abs/1902.05605>
- X. Chen et al., "REDQ," ICLR 2021 (high-UTD reference).
  <https://arxiv.org/abs/2101.05982>
- T. Schaul et al., "Prioritized Experience Replay," ICLR 2016.
  <https://arxiv.org/abs/1511.05952>
