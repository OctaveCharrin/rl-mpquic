# RL Agent Architecture & Training Strategy: Justification and Improvements

Scientific grounding for the agent design in `src/rl/` and the training loop in
`src/train/hierarchical_train.py`, plus a prioritized list of improvements to
optimize results. Companion to `docs/REWARD_TUNING.md` (the reward side).

## 1. What is being justified

| Component | Where | Current design |
|-----------|-------|----------------|
| Base algorithm | `sac_agent.py` | Soft Actor-Critic: tanh-Gaussian actor, twin Q critics, Polyak targets, auto entropy temperature, normalized `[-1,1]` actions |
| Hierarchy | `hierarchical_train.py` | Two SAC agents on two timescales — **App** (bitrate, every `app_period_s=1 s`) over **Path** (per-path split, every frame) |
| Coupling | `path_agent.py`, `realtime_env.py` | Path observation includes the App's current target bitrate (the "goal") |
| Credit assignment | `hierarchical_train.py:189` | App action credited with QoE accumulated over the *window* of frames it governed; Path credited per frame |
| Path arch A (`flat`) | `sac_agent.py` | Fixed-dim MLP; `num_paths`-D action softmaxed into a split |
| Path arch B (`scoring`) | `scoring_sac_agent.py` | Permutation-equivariant: shared per-path encoder actor + masked-softmax split; DeepSets masked-mean twin critic; variable/masked path set |
| SAC hyperparams | `configs/*.yaml` | `hidden=256, γ=0.99, τ=0.005, lr=3e-4, batch=256, buffer=200k, start_steps=1000, updates_per_step=1, auto_entropy` |

## 2. Justification of the design choices

### 2.1 Why SAC (the base learner)

SAC (Haarnoja et al., 2018) is the right base for this problem for four
reasons, each matching a property of the environment:

- **Off-policy ⇒ sample efficiency.** NS-3 frames are expensive to generate, and
  a real-time system produces one transition per frame. SAC reuses every
  transition from a replay buffer many times, and empirically exceeds DDPG/TD3/PPO
  in sample efficiency on continuous-control benchmarks — the dominant selection
  criterion here.
- **Continuous actions.** Both levers are continuous — a target bitrate and a
  simplex-valued split. SAC's tanh-squashed Gaussian is a native fit; the code
  keeps a normalized `[-1,1]` action and maps outward (`AppAgent.to_kbps`,
  `PathAgent.to_split`), a clean separation of policy from actuation.
- **Maximum-entropy ⇒ robust exploration.** The max-entropy objective yields
  policies "robust in the face of modeling and estimation errors" (Haarnoja) —
  valuable under the non-stationary dynamics (churn/regime/burst), where a
  brittle deterministic policy would overfit one regime.
- **Twin critics + auto temperature ⇒ stability.** Clipped double-Q (from TD3;
  Fujimoto et al., 2018) mitigates value overestimation (`sac_agent.py:200`), and
  automatic entropy tuning (Haarnoja et al., "Algorithms and Applications", 2018)
  removes the hardest hyperparameter (`sac_agent.py:209`). The scoring agent
  correctly scales the entropy target to the *active* path count
  (`scoring_sac_agent.py:239`) rather than a fixed `-N`.

### 2.2 Why a two-timescale hierarchy

The App/Path split is a **feudal / two-timescale hierarchy**, grounded in:

- **Feudal RL** (Dayan & Hinton, 1993) and **FeUdal Networks** (Vezhnevets et
  al., 2017): a *Manager* operating at a coarser temporal resolution sets a goal
  that a *Worker* enacts at every environment tick. Here the App agent is the
  Manager (slow, sets the bitrate "goal") and the Path agent is the Worker (fast,
  moves bytes). The decoupling across timescales is exactly FuN's mechanism for
  long-horizon credit assignment.
- **Options / semi-MDP framework** (Sutton, Precup & Singh, 1999): the App
  action persists over a window of frames and is credited with the reward
  accumulated across that window — a temporally-extended action (an "option").
  `pop_app_window_reward()` (`hierarchical_train.py:190`) implements the SMDP
  reward accumulation.
- **The coupling makes it hierarchical, not two independent agents.** The Path
  observation includes the App's target bitrate (`path_agent.py` docstring), so
  the Worker conditions on the Manager's goal — the defining feature of feudal
  control.

This decomposition is also empirically motivated by the repo's own finding
(`CLAUDE.md`): on the static topology an App-only ablation reaches ~98%, but on
the dynamic scenario it collapses — i.e. the two levers genuinely need separate,
differently-paced controllers.

### 2.3 Why a permutation-equivariant `scoring` path agent

When the path set is variable/changing (churn), a fixed-dim MLP is mis-specified:
it hard-codes path identity by input position and cannot handle appearance/
disappearance. The `scoring` agent uses the **Deep Sets** construction (Zaheer et
al., 2017): a shared per-element encoder + a permutation-invariant pooling
(masked mean) so the policy is invariant to path ordering and handles variable
`N`. This is the same inductive bias used for exchangeable-object RL (Mern et
al., 2020) and mirrors the SCION path-scoring sibling referenced in the code. The
masked softmax over active paths and the summed-over-active-paths log-prob
(`scoring_sac_agent.py:86`) keep dead paths gradient-free — correct handling of
the liveness mask.

### 2.4 Domain fit vs. the MPQUIC-RL literature

DRL for multipath-QUIC scheduling is an established line — DQN schedulers (Sensors
2022), the **MARS** multi-agent per-path scheduler (ACM TOMM 2024), and
PPO+LSTM designs (PRISM). Our design is consistent with, and in places ahead of,
this literature: most use *discrete* per-packet DQN, whereas we use *continuous*
SAC over a simplex split with a hierarchical rate controller on top. MARS's
per-path network mirrors our shared per-path encoder in the scoring agent.

## 3. Suggested improvements (prioritized)

Ordered by expected return-on-effort for *this* codebase.

### Tier 1 — high impact, low effort

**Status: Tier-1 #1, #2, #3 are implemented** (DroQ-style critic LayerNorm +
higher UTD, Prioritized Experience Replay, and replay-buffer persistence across
`--resume`). All are off by default and enabled in `configs/dynamic.yaml`; see
`docs/PATCH_TIER1_SAMPLE_EFFICIENCY.md`.

1. **Raise the update-to-data ratio (UTD) with critic normalization.** *(done —
   `critic_layernorm` + `updates_per_step: 10` in `dynamic.yaml`; LayerNorm is
   threaded into the critic `_mlp`/`_encoder` only, never the policy body.)*
   `updates_per_step = 1` under-exploits SAC's off-policy advantage precisely
   where samples are most expensive (NS-3). Increasing UTD to 4–20 extracts far
   more learning per frame, but naive high-UTD SAC diverges from Q-overestimation.
   The fix is cheap and well-established:
   - **DroQ** (Hiraoka et al., 2022): small critic ensemble + **Dropout +
     LayerNorm**, stable at UTD 20.
   - **CrossQ** (Bhatt et al., ICLR 2024): **BatchNorm** in the critic, *removes
     target networks*, matches ensemble methods at **UTD 1** — the simplest,
     lightest option.
   - **REDQ** (Chen et al., 2021): 10-critic ensemble, the reference high-UTD
     method (heavier).
   Concretely: add `nn.LayerNorm` to `_mlp`/`_encoder` (`sac_agent.py:48`,
   `scoring_sac_agent.py:46`) and bump `updates_per_step`. This is the single
   biggest expected win.

2. **Prioritized Experience Replay** (Schaul et al., 2016). *(done — sum-tree
   `PrioritizedReplayBuffer`/`PrioritizedStructuredReplayBuffer`, IS-weighted
   critic loss, beta annealed `per_beta0`→1; enable with `prioritized: true`.)*
   The critical events
   under `configs/dynamic.yaml` — churn-out, regime swaps, correlated failures —
   are *rare*, so uniform sampling from `ReplayBuffer`/`StructuredReplayBuffer`
   under-trains exactly the transitions that matter. TD-error prioritization
   upweights them. Drop-in change to `replay_buffer.py` (add a sum-tree and
   importance-sampling weights into the critic loss).

3. **Persist the replay buffer across `--resume`.** *(done — `--persist-buffer`
   writes `app_buffer.npz`/`path_buffer.npz`; resume restores them and skips the
   cold-start warm-up hack.)* The loop notes the buffer
   "restarts empty" on resume (`hierarchical_train.py:121`), discarding warm data
   and re-incurring a cold refill each chunk. Serializing the buffer alongside the
   checkpoints removes a real regression in chunked/interruptible runs.

### Tier 2 — medium impact

4. **Handle partial observability with short history or recurrence.** The path
   capacity is time-varying and *hidden* (only its effects are observed), so a
   single-frame snapshot makes this a POMDP. Frame-stacking (last k observations)
   or a GRU encoder (as in DRQN, Hausknecht & Stone 2015; R2D2, Kapturowski et
   al., 2019; and PRISM's LSTM in the MPQUIC setting) lets the agent infer the
   regime rather than react blindly. Start with cheap frame-stacking before
   recurrence.

5. **Attention pooling in the scoring critic.** The DeepSets **masked-mean** pool
   (`scoring_sac_agent.py:104`) is permutation-invariant but cannot represent
   *interactions* between paths — yet `corr_groups` (shared-bottleneck correlated
   failures) are exactly inter-path structure. A **Set Transformer** (Lee et al.,
   2019) self-attention pool captures "these two paths fail together," directly
   targeting the correlated-failure dynamics the mean-pool is blind to.

6. **Off-policy correction for the manager (HIRO).** As the Path worker's policy
   changes, stored App transitions become stale (the same bitrate goal now yields
   a different windowed outcome). HIRO (Nachum et al., 2018) relabels manager
   transitions to stay consistent with the current worker — the standard fix for
   non-stationary HRL credit assignment. Relevant because the App reward is a
   Monte-Carlo window sum with no within-window bootstrap.

7. **Domain randomization over `DynamicsConfig` during training.** The dynamics
   parameters (churn/regime/burst rates) are fixed per config; randomizing them
   per episode (Tobin et al., 2017) trains a policy robust to a *distribution* of
   networks rather than one, improving generalization to unseen NS-3 conditions.
   The machinery already exists — just sample the config fields per `reset`.

### Tier 3 — research-grade / exploratory

8. **N-step returns for the fast Path agent** to accelerate credit assignment
   over the per-frame horizon (with the usual off-policy caveats).
9. **Reward/return scaling (PopArt)** so the App (windowed) and Path (per-frame)
   magnitudes don't need hand-balanced learning rates.
10. **Distributional critics** (QR-DQN/IQN-style) to better model the heavy-tailed
    latency under bursts.

### Correctness notes worth a look

- **Double squashing (flat path agent).** The flat action is tanh-squashed *then*
  softmaxed into a split (`path_agent.py:65`), while the entropy target is `-N`
  in raw action space — a mild mismatch between the entropy the agent regularizes
  and the split it actually emits. The scoring agent's masked-softmax formulation
  is cleaner; consider standardizing on it.
- **UTD note interacts with Tier 1**: bumping `updates_per_step` without the
  critic normalization in (1) will likely destabilize — apply them together.

## 4. Recommended first step

Do Tier-1 #1 (critic LayerNorm + higher UTD, DroQ/CrossQ-style) and #2 (PER)
first: both are localized changes to `sac_agent.py` / `replay_buffer.py`, both
directly target sample efficiency on expensive NS-3 frames, and both are
strongly evidenced in the literature. Validate with the existing
`evaluate.py --ablation` on `configs/dynamic.yaml` across multiple seeds before
and after.

## References

**Core RL algorithm**
- T. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with
  a Stochastic Actor," ICML 2018. <https://arxiv.org/abs/1801.01290>
- T. Haarnoja et al., "Soft Actor-Critic Algorithms and Applications," 2018
  (automatic temperature). <https://arxiv.org/abs/1812.05905>
- S. Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic
  Methods" (TD3, clipped double-Q), ICML 2018. <https://arxiv.org/abs/1802.09477>

**Hierarchical RL**
- P. Dayan & G. Hinton, "Feudal Reinforcement Learning," NeurIPS 1993.
- A. Vezhnevets et al., "FeUdal Networks for Hierarchical RL," ICML 2017.
  <https://arxiv.org/abs/1703.01161>
- R. Sutton, D. Precup, S. Singh, "Between MDPs and semi-MDPs: A framework for
  temporal abstraction" (options), Artificial Intelligence, 1999.
- O. Nachum et al., "Data-Efficient Hierarchical RL" (HIRO), NeurIPS 2018.
  <https://arxiv.org/abs/1805.08296>

**Permutation-invariant / set architectures**
- M. Zaheer et al., "Deep Sets," NeurIPS 2017. <https://arxiv.org/abs/1703.06114>
- J. Lee et al., "Set Transformer," ICML 2019. <https://arxiv.org/abs/1810.00825>
- J. Mern et al., "Object Exchangeability in Reinforcement Learning," AAMAS 2020.
  <https://arxiv.org/abs/1905.02698>

**Sample-efficient SAC variants**
- X. Chen et al., "Randomized Ensembled Double Q-Learning" (REDQ), ICLR 2021.
  <https://arxiv.org/abs/2101.05982>
- T. Hiraoka et al., "Dropout Q-Functions for Doubly Efficient RL" (DroQ), ICLR
  2022. <https://arxiv.org/abs/2110.02034>
- A. Bhatt et al., "CrossQ: Batch Normalization in Deep RL for Greater Sample
  Efficiency and Simplicity," ICLR 2024. <https://arxiv.org/abs/1902.05605>
- T. Schaul et al., "Prioritized Experience Replay," ICLR 2016.
  <https://arxiv.org/abs/1511.05952>

**Partial observability & robustness**
- M. Hausknecht & P. Stone, "Deep Recurrent Q-Learning for POMDPs" (DRQN), 2015.
  <https://arxiv.org/abs/1507.06527>
- S. Kapturowski et al., "Recurrent Experience Replay in Distributed RL" (R2D2),
  ICLR 2019.
- J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks
  from Simulation to the Real World," IROS 2017.
  <https://arxiv.org/abs/1703.06907>

**Multipath-QUIC RL (domain)**
- H. Wu et al., "Multi-agent DRL-based Multipath Scheduling for Video Streaming
  with QUIC" (MARS), ACM TOMM 2024. <https://dl.acm.org/doi/10.1145/3649139>
- "Reinforcement Learning Based Multipath QUIC Scheduler for Multimedia
  Streaming," Sensors 22(17):6333, 2022. <https://www.mdpi.com/1424-8220/22/17/6333>
- "PRISM: PPO with deep RL for Intelligent Scheduling in Multipath QUIC," 2026.
