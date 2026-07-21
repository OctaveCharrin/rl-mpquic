---
marp: true
title: "RL-MPQUIC — Dual-Agent Hierarchical RL for Real-Time Multipath Video"
paginate: true
theme: default
---

<!--
Slide deck for the lab implementation team. Target: ~10 minutes (15 core slides
+ 1 backup). Renders with Marp (`marp SLIDES.md` / VS Code Marp extension),
reveal-md, or pandoc. Slides separated by `---`; presenter notes in HTML
comments with a per-slide time budget that sums to ~10:00.
Source of truth: docs/IMPLEMENTATION.md.
-->

# RL-MPQUIC
## Dual-agent hierarchical RL for real-time video over multipath QUIC

Octave Charrin — implementation walkthrough

**Agenda:** ① NS-3 network body · ② RL agents (I/O · architecture · training · reward) · ③ Baselines · ④ Results

<!-- 0:25 — One line: a Python "brain" (PyTorch SAC) drives a thin NS-3 "body"
over a shared-memory bridge. Two agents, two timescales, one bridge exchange per
video frame. Today: how each piece is actually implemented. -->

---

# The big picture

- **Goal:** control a WebRTC-like real-time video call over **N abstracted MPQUIC subflows** — pick the encoder **bitrate** *and* how to **split each frame** across paths.
- **Brain / body split:**
  - **Brain** = Python + PyTorch — two SAC agents. All RL logic.
  - **Body** = NS-3 C++ scenario — topology, frame push, byte striping, transport-state reporting. **No RL logic.**
  - Bridged by **`ns3-ai` struct-based shared memory**.
- **Two interchangeable bodies** behind one `DataPlane` ABC:
  - `Ns3DataPlane` — real packet-level NS-3.
  - `MockRealtimeDataPlane` — pure-Python fluid/queue surrogate for tests, CI, fast RL iteration (deterministic per seed).
- Everything downstream (env, reward, agents, training) is **backend-agnostic**.

<!-- 0:45 — Emphasize: mock and NS-3 expose the *identical* method contract, so
we develop against the mock and validate on NS-3. The contract is two POD structs. -->

---

# The decision epoch — the two-timescale contract

- Clocked by **one video frame** (`1/fps`, default **33.3 ms @ 30 fps**).
- **Exactly one** observation→action exchange per frame. Body **leads with a send**, then blocks for the action; sim time frozen during the exchange.
- **Path agent** acts **every frame** → chooses `splitRatio[]`.
- **App agent** acts only on an **app boundary** (`app_period_s` = 1 s = every 30 frames), gated by `EnvStruct.appDecisionDue`. Its `targetBitrateKbps` **persists** between boundaries.

Wire format — two POD C structs (`ns3/realtime_mpquic.h`), mirrored by Python dataclasses:

| C++ → Python | Python → C++ |
|---|---|
| `EnvStruct` ↔ `FrameObs` | `ActStruct` ↔ (bitrate, split) |

`kMaxPaths = 8` fixed arrays → trivially copyable; only `numPaths` slots read. One NS-3 process serves the whole run; episodes reset **in-band** (`ACT_RESET`).

<!-- 0:45 — The hierarchy lives here: the Path agent's observation *includes* the
App bitrate, so it schedules the load the App agent chose. -->

---

# ① NS-3 body — topology & transport

`RealtimeController::Build()` — a **dumbbell of N parallel paths**, client ↔ server. Per path `i`:

- **`PointToPointHelper`** link: configured **bottleneck `DataRate`** + **one-way `Delay`**.
- **FqCoDel AQM** (`TrafficControlHelper`) → queueing delay/loss emerge endogenously.
- Dedicated **`/24` subnet**; a **`RealtimeSource` (server) + `RealtimeSink` (client)** = one long-lived subflow.
- **On/off UDP cross-traffic** — mean rate `cross_frac × link_rate`, 1200 B packets, exponential on/off with per-path phase.

**Selectable per-path transport** (`transport: tcp | udp`):
- **`tcp`** (default): stock single-path NS-3 TCP; loss → added latency (reliable/in-order).
- **`udp`**: unreliable datagrams (≤1400 B + 24 B `UdpFrameHeader`) with **explicit app-layer deadline drop** — the closer analogue to QUIC unreliable datagrams.

One YAML `topology.paths` (`{rate, delay, cross_frac}`) drives **both** NS-3 links and the mock's per-path baselines.

<!-- 0:55 — cross_frac drives congestion; FqCoDel means we don't hand-model
queues. TCP is byte-identical to the historical scenario; UDP is the realtime
analogue. -->

---

# ① NS-3 body — the life of one frame

1. **Generate** (`GenerateFrame`): bytes/frame = `(bitrate·1000/8)/fps`, ×2.5 on keyframes (1/s), ±25 % P-frame jitter.
2. **Stripe**: normalize split → per-path byte shares → **reconcile rounding onto the largest share** (shares sum *exactly* to frame bytes).
3. **Deliver + track** — TCP is one reliable in-order stream carrying many frames, so per-frame latency uses a **byte-watermark** scheme:
   - each path keeps `m_pathEnq[i]` (bytes ever enqueued);
   - a share records `{watermark, frameId, bytes}`; **delivered when cumulative-received ≥ watermark**;
   - a frame completes when all its non-empty shares complete.
4. **Latency = max over shares of (arrival − generation)**.

**Application-layer "loss" ≡ deadline miss:** last byte after **`deadline_ms` (180 ms)** ⇒ `loss = 1.0`, `bytes_delivered = 0` (swept by `ExpireLateFrames`). Distinct from packet loss (which under TCP just adds latency).

<!-- 0:45 — The watermark scheme is the trick for per-frame latency without
per-packet tagging. Deadline-miss = the realtime "loss" the App loss EWMA tracks. -->

---

# ① NS-3 body — non-stationary dynamics (optional)

Off by default (byte-identical static scenario). Four coupled mechanisms make *which path to send on* time-varying. Deterministic per seed, once/frame, `P(event)=1−e^{−rate·Δt}`, fixed draw order:

| Mechanism | Effect |
|---|---|
| **Churn** | per-path on/off Markov chain (≥ `min_active`); bytes on a dead path are **lost** |
| **Regime** | capacity ×`U(lo,hi)` at Poisson change-points → abrupt best-path swaps |
| **Burst** | Poisson transient capacity collapse |
| **Corr** | groups of paths degrade together (shared bottleneck) |

**Faithful NS-3 port (not a naive one):**
- **Churn = drop-all receive error models on both directions** — NOT a `DataRate` collapse (that strands a packet mid-serialization for ~10 s). TCP subflow torn down / reconnected in lockstep with the sink.
- **Regime/burst/corr** scale bottleneck `DataRate` **and** cross-traffic together (rescale-safe `CrossTrafficApp`); seasonal envelope + noise mirrored from the mock.

**Parity guard:** `scripts/parity_check.py` runs an even-split policy through both backends and asserts loss regimes match.

<!-- 0:45 — The churn detail is the one that bit us. Mock and C++ share
amp/period formulas — keep in sync. -->

---

# ② App agent — observation, action, SAC

**App observation** (`build_app_obs`, **dim 5**) — aggregate, **sender-observable only**:
`[bitrate_norm, rtt/200ms, jitter/50ms, loss, throughput/cap]`
→ no per-path detail, **no episode progress** (horizon-agnostic — a real call has unknown length).

**Action:** dim **1**, `a∈[−1,1]` → `kbps = min + (a+1)/2·(max−min)`, **[300, 6000] kbps**.

**Generic SAC core** (`sac_agent.py`), normalized `[−1,1]^d` action space:
- **Actor** = tanh-squashed diagonal Gaussian, MLP `256×256` → `{mean, log_std∈[−20,2]}`, tanh log-det correction.
- **Critics** = **twin Q** (clipped double-Q) + Polyak target (`τ=0.005`).
- **Auto-entropy** temperature, target `H̄ = −d_act`.
- Off-policy: replay buffer (200k) amortizes expensive NS-3 frames; `γ=0.99`, lr `3e-4`, batch 256; uniform-random warm-up `start_steps=1000`.

<!-- 0:50 — SAC because it's sample-efficient (1 frame = 1 transition = 1 grad
step) and max-entropy exploration suits continuous bitrate/split. The raw [-1,1]
action is stored & scored, not the kbps. -->

---

# ② Path agent — observation + flat vs. scoring

**Path observation** — bitrate is element 0 ⇒ **the hierarchical coupling**:
- **Flat** (`build_path_obs`, **dim `4 + 5N`**): global `[bitrate, rtt, loss, thr]` + per path `[cwnd, srtt, bufferOcc, path_thr, path_loss]`.
- **Structured** (`build_path_state`): `PathState(glob (4,), paths (N,6), mask (N,))` — extra per-path feature = **liveness flag**.

`PathAgent` dispatches on `arch`:
- **`flat`** (default): `SACAgent`, `d_act=N`, split = **softmax(a/τ)**. Fixed path count/order.
- **`scoring`** — **permutation-equivariant** SAC, handles a **variable/changing** path set (churn):
  - **Actor** = a **shared per-path encoder** → per-path latent → **masked softmax over active paths**; log-prob/entropy over active paths only.
  - **Critic** = **DeepSets** — encode `glob ⊕ path_i ⊕ latent_i`, **masked-mean-pool** → scalar Q (permutation-invariant, variable-N).
  - Entropy target scales with **active-path count** (`H̄ = −mask.sum()`). Dead paths → no density, no gradient, no split.

<!-- 1:00 — The research contribution on the transport side. A flat MLP can't
represent "path 4 just disappeared." Shared encoder + DeepSets + mask = invariant
to how many paths are live and their order. Adapted from the SCION sibling's DQN. -->

---

# ② Reward design — clean decomposition

**Perceptual quality = VMAF (0–100)**, default log rate-quality curve through anchors **(300 kbps, 25)** and **(4300 kbps, 92)** — concave, diminishing returns. Pluggable `vmaf_fn` → optional **WebRTC-grounded learned surrogate** `(bitrate, loss, delay, jitter) → VMAF`.

**App QoE** (per window, clipped `[−2,1]`):
```
QoE = a·VMAF(bitrate)/100 − b·lat − c·jit − d·loss
```
weights `a=1, b=0.5, c=0.5, d=1`; `lat=clip(latency/200ms)`, `jit=clip(jitter/50ms)`.

**Path reward** (per frame, clipped `[−2,1]`):
```
R_p = (1 − loss) − b·lat − c·jit          # NO quality term
```

The Path agent doesn't set bitrate, so rewarding it for quality would be **miscredited** — it's paid purely to **deliver frames fast & intact**. This attribution gives each agent a clean gradient.

<!-- 0:45 — Two separate reward functionals, one per agent. Learned-VMAF variant
drops −d·loss (already folded in) and adds a utilization term. -->

---

# ② Training — two-timescale credit assignment

**Reward window** (`realtime_env.py`): per-frame `latency/jitter/loss` accumulate for the frames the *current* bitrate governs. At an app boundary, `pop_app_window_reward()` aggregates + scores + clears, crediting it to the **previous** App action — a correct temporally-extended reward. Path agent is rewarded **immediately** each frame.

**Dual-cadence loop** (`hierarchical_train.py`):
```
while not done:
    if app_decision_due:                      # slow: credit closed window,
        r_app = pop_app_window_reward()       #       store, update, pick bitrate
        app.store(...); app.update(); target_kbps = app.select(app_obs)
    p_obs = path_obs(obs, target_kbps, arch)  # fast: every frame
    split = path_agent.select(p_obs)
    next_obs, r_p, done = env.step(target_kbps, split)
    path_agent.store(...); path_agent.update()
```
- **Truncation, not terminal:** every transition stored `done = False` → bootstrap off next state (horizon-agnostic).
- Checkpoints **every episode**; `--resume` continues. `finally` → `dp.close()` (ACT_TERMINATE).

<!-- 0:45 — The window is the crux of hierarchical credit: the App action is
judged on the QoE it actually produced over its 30 frames. -->

---

# ③ Baselines — how each makes its decision

All **mask-aware** (dead path → zero weight). `even/single/proportional/random` share a **reactive bitrate rule**: `target = 0.9 × recent aggregate goodput` — so they isolate **scheduling quality**.

| Baseline | Bitrate | Split rule |
|---|---|---|
| `even` | reactive goodput | uniform over active paths |
| `single` | reactive goodput | whole frame on highest-throughput active path (argmax) |
| `proportional` | reactive goodput | ∝ recent per-path throughput |
| `random` | reactive goodput | fresh Dirichlet-uniform split (seeded) |
| **`webrtc`** | **GCC** | ∝ recent per-path throughput |
| `learned` | trained App agent | trained Path agent (deterministic) |

**`webrtc` = stateful WebRTC Google Congestion Control** (`_GccBitrate`), the realistic reference:
- **loss-based:** `loss>10%` → `est×=(1−0.5·loss)`; `loss<2%` → may increase
- **delay-based:** queuing delay (RTT over min-RTT) > 50 ms → `est = 0.85 × recv rate`; else `est ×= 1.08`
- Carries its own estimate → **probes** for capacity, never spirals to the floor.

<!-- 0:50 — Key: the reactive rule reads *achieved* goodput; GCC only reads
goodput on the back-off branch. That difference is the next slide. -->

---

# ③ Ablation — isolating each agent

`--ablation` disables exactly one learned agent by swapping in a heuristic:

| Variant | Bitrate | Split | Isolates | vs. |
|---|---|---|---|---|
| `app_only` | learned | **even** (Path off) | value of learned **bitrate** | `even` |
| `path_only_gcc` | **GCC** (App off) | learned | value of learned **scheduler** | `webrtc` |
| `learned` | learned | learned | full system | — |

**Why GCC and not the reactive rule for the App-off case?**
The reactive `0.9 × goodput` rule is a **feedback loop**: a split that under-uses the network → low goodput → low bitrate → fewer bytes → … spirals to the **300 kbps floor** (VMAF floor). Paired with the learned split it **traps the variant at the floor and masks the scheduler**. GCC probes for capacity and holds a realistic load → the QoE gap vs. the matched `webrtc` reference reflects **scheduling quality alone**.

<!-- 0:40 — An ablation must not confound the thing you removed with a degenerate
bitrate. GCC fixes it. -->

---

# ④ Results — static topology (`four_path`, N=4)

<div style="display:flex; gap:1em;">

![w:560](runs/a_fig_20260710/static_figure5_quality_vs_cost.png)
![w:560](runs/a_fig_20260710/static_figure12_ablation.png)

</div>

- **Left — QoE vs. compute:** the hierarchical controller (★) sits **top-right** — best QoE, at a small compute premium (SAC forward pass + obs build, still far inside a 33 ms frame budget).
- **Right — ablation:** `app_only` reaches **~98 %** of the full system. On a static topology the optimal split barely moves, so the scheduler earns only a **modest** gain — expected, and the reason we built the dynamic scenario.

<!-- 0:45 — Honest framing: on a fixed network, bitrate control does most of the
work. Sets up the payoff on the next slide. -->

---

# ④ Results — dynamic topology (`dynamic`, N=6, churn+regime+burst+corr)

![w:620 center](runs/a_fig_20260710/dyn_figure12_ablation.png)

- **`app_only` collapses** — an even split shoves the encoder's bitrate onto congested/dead paths, so deadline misses spike and VMAF craters.
- The **full learned pair stays far ahead**; `path_only_gcc` vs. `webrtc` isolates the scheduler's now-**large** contribution (both at the same GCC bitrate).
- **Headline:** non-stationary paths + a Path agent that ingests the changing path set is what makes scheduling decisive.

<!-- 0:45 — This is the payoff vs. the static ~98% slide. Next slide shows *why*:
the split actually moves. -->

---

# ④ Split behaviour — static vs. dynamic

<div style="display:flex; gap:1em;">

![w:560](runs/a_fig_20260710/static_figure7_split_behavior.png)
![w:560](runs/a_fig_20260710/dyn_figure7_split_behavior.png)

</div>

- **Left — static (`four_path`):** the per-path split is **essentially flat** over the episode — the best allocation never changes, so a fixed/heuristic split is already near-optimal (why `app_only ≈ 98 %`).
- **Right — dynamic (`dynamic`, scoring agent):** the stacked area **shifts continuously** — the agent reallocates as paths **churn out** (a band collapses to zero), **regimes swap** the best path, and **bursts** hit. It tracks live, fast paths and abandons dead ones **frame-by-frame**.

The contrast *is* the argument: the scheduler earns its keep only when "which path" is time-varying — and the permutation-equivariant model is what lets it follow a changing path set.

<!-- 0:45 — Point at a band going to zero on the right (a churn-out) and the
others swelling to absorb it. On the left, nothing moves. -->

---

# Takeaways for the implementation team

- **One bridge, one struct pair, one exchange/frame.** Keep C++ thin; all logic in Python. Edit `EnvStruct`/`ActStruct` + both dataclasses **together**.
- **Two agents, two timescales:** App = bitrate (1 s), Path = split (per frame); Path is conditioned on the App bitrate. Rewards are **decomposed & window-credited**.
- **Scoring Path agent** (shared encoder + DeepSets + mask) handles a **changing path set**.
- **Mock ⇄ NS-3 parity** is a hard contract (`parity_check.py`) — dynamics ported faithfully (churn = drop-all error models, not DataRate collapse).
- **Evaluation:** baselines share a bitrate rule to isolate scheduling; `webrtc` (GCC) is the realistic reference; the ablation uses GCC to avoid the reactive-bitrate collapse.
- **Result:** static → scheduler earns little (`app_only ≈ 98 %`); **dynamic → `app_only` collapses, the hierarchical controller wins**.

**Deep-dive reference:** `docs/IMPLEMENTATION.md` (exact shapes, formulas, file map).

<!-- 0:25 — Close: the dynamic scenario is the reason the whole architecture
exists. Questions → point people at IMPLEMENTATION.md §-numbers. -->

---

# Backup — Quality vs. compute, dynamic scenario

![w:640 center](../runs/a_fig_20260710/dyn_figure5_quality_vs_cost.png)

- Same Pareto view under dynamics: **Hierarchical RL (★) dominates on QoE**.
- Decision cost = SAC forward pass + observation build — **sub-ms to a few ms/frame**, comfortably within the 30 fps budget.
- A small, bounded compute premium buys a **large QoE gain** exactly where scheduling is hard.

<!-- Backup slide — show only if asked about compute cost under dynamics. -->
