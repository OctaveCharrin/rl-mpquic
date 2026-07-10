# IMPLEMENTATION.md — Design & Implementation Reference

A precise, implementation-level description of the `rl-mpquic` system for the lab
implementation team. Where `ARCHITECTURE.md` is the narrative/rationale document,
this file is the **spec you build and modify against**: exact shapes, exact
formulas, exact control flow, and the file/function each lives in.

The system is a **dual-agent hierarchical SAC controller** for real-time video
(WebRTC-like conferencing) over an abstracted **multipath QUIC** transport. Python
(PyTorch) is the "brain"; an NS-3 C++ scenario is the "body". A pure-Python mock
body mirrors the same interface for fast, NS-3-free iteration.

Contents:
1. [The network simulation](#1-the-network-simulation)
2. [The agents (SAC + baselines) and their I/O](#2-the-agents-sac--baselines-and-their-io)
3. [The training pipeline](#3-the-training-pipeline)
4. [The evaluation experiments](#4-the-evaluation-experiments)
5. [Appendix: exact dimensions, constants, file map](#5-appendix-exact-dimensions-constants-file-map)

---

## 0. The decision epoch and the two-timescale contract

Everything is clocked by **one video frame** (`1/fps`, default 33.3 ms at 30 fps).
Per frame there is exactly **one** observation→action exchange between brain and
body:

- The **Path agent** acts **every frame** — it chooses the per-path byte
  split `splitRatio[]`.
- The **App agent** acts only on an **app boundary** (every `app_period_s`, default
  1 s = every 30 frames), signalled by `EnvStruct.appDecisionDue`. Its chosen
  `targetBitrateKbps` **persists** in the action struct between boundaries.

The two agents share a single synchronous bridge. The body always **leads with a
send** (fills the observation, sends it, then blocks waiting for the action). Sim
time is frozen during the exchange and only advances while the frame's bytes are
delivered.

The contract is two POD C structs (`ns3/realtime_mpquic.h`), mirrored exactly by
two Python dataclasses (`src/ns3env/dataplane.py`): `EnvStruct`↔`FrameObs`,
`ActStruct`↔(bitrate, split). Fixed-size arrays `kMaxPaths = 8` keep them trivially
copyable for the shared-memory interface; only `numPaths` slots are read.

---

## 1. The network simulation

There are **two interchangeable backends** behind one abstract base class
`DataPlane` (`src/ns3env/dataplane.py`). Both implement the identical method
contract, so the env/reward/agents/training code is backend-agnostic:

```python
class DataPlane(ABC):
    num_paths: int          # candidate path count (upper bound; active count can vary)
    cap_mbps: float         # aggregate throughput normalizer for observations
    def reset(seed) -> FrameObs
    def current_obs() -> FrameObs
    def step_frame(target_bitrate_kbps, split_ratio) -> FrameResult
    def is_done() -> bool
    @property clock_s -> float
    def app_decision_due() -> bool
    def close()
```

- `reset(seed)` starts an episode and returns the initial `FrameObs`.
- `step_frame(bitrate, split)` **applies the action, advances the clock by
  `1/fps`**, and returns the realized `FrameResult` of the frame just delivered
  (`latency_ms`, `jitter_ms`, `loss`, `bytes_delivered`). The *next* observation is
  available via `current_obs()`.

### 1.1 The observation/result payload

`FrameObs` (mirror of `EnvStruct`) carries:

| Group | Fields |
|---|---|
| Framing | `num_paths`, `clock_s`, `done`, `app_decision_due` |
| App aggregate | `current_bitrate_kbps`, `rtt_ms`, `jitter_ms`, `loss`, `throughput_mbps` |
| Per-path (len `num_paths`) | `cwnd[]`, `srtt_ms[]`, `buffer_occ[]`, `path_throughput_mbps[]`, `path_loss[]` |
| Liveness | `path_active[]` — `1.0` live / `0.0` churned out (all-ones under static dynamics) |
| Last frame result | `last_latency_ms`, `last_jitter_ms`, `last_loss`, `last_bytes` |

`FrameResult` = `(latency_ms, jitter_ms, loss, bytes_delivered)` for one frame.

**Application-layer "loss" ≡ deadline miss.** A frame whose last byte arrives after
`deadline_ms` (default 180 ms) counts as lost (`loss = 1.0`, `bytes_delivered = 0`).
This is the real-time-conferencing semantics — distinct from network packet loss,
which under TCP surfaces as added latency.

### 1.2 The video source model (identical on both backends)

`src/ns3env/video_source.py::frame_bytes` (mirrored by
`RealtimeController::GenerateFrame` in C++). One frame every `1/fps` on the wall
clock regardless of whether the previous frame finished (this is what produces
bufferbloat under overdrive):

```
base   = (bitrate_kbps * 1000 / 8) / fps                 # bytes/frame at target rate
kf     = 2.5 if (frame_idx % keyframe_interval == 0) else 1.0    # I-frame burst
jitter = 1 + frame_size_jitter * U(-1, 1)                # P-frame variability
frame_bytes = max(1, round(base * kf * jitter))
```

Defaults: `fps = 30`, `keyframe_interval = 30` (one I-frame/sec),
`frame_size_jitter = 0.25`, bitrate clamped to `[300, 6000]` kbps.

### 1.3 Byte striping and per-frame completion (shared logic)

Given `split_ratio` and `frame_bytes`, both backends:
1. Normalize the split: clamp negatives, renormalize to sum 1, fall back to uniform
   if degenerate (`_normalize_split` / the C++ equivalent).
2. Split bytes across paths by ratio, then **reconcile rounding onto the largest
   share** so per-path shares sum *exactly* to `frame_bytes` (`_split_bytes`).
3. Deliver each non-empty share and record its arrival time. The frame's
   **latency = max over its shares of (arrival − generation time)**; a share routed
   onto a dead path never arrives (counts as dropped/lost).

### 1.4 Backend A — `MockRealtimeDataPlane` (fluid/queue surrogate)

Pure Python, **deterministic per seed**, no NS-3 dependency. It is *not* a packet
simulator — it is a standing-queue fluid model calibrated to the same topology.
This is the backend used for all unit tests, CI, and rapid RL iteration.

**Per-path time-varying capacity** (`_PathTrace.capacity_mbps`):

```
season = 1 + amp * sin(2π t / period + phase)
cross  = cross_frac * (0.5 + 0.5 * sin(2π t / cross_period + phase))
cap    = max(0.05, base_mbps * season * max(0.05, 1 − cross) * (1 + N(0, noise_std)))
```

i.e. a sinusoidal capacity envelope, a phase-shifted bursty cross-traffic
reduction, and multiplicative noise (`noise_std = 0.05`).

**Standing-queue delivery** (reproduces bufferbloat). Each path holds a `busy_until`
serialization clock; delivering `b` bytes:

```
ser_s   = (b * 8) / (cap * 1e6)                  # serialization at current capacity
ready   = max(now, busy_until[i])                # queue ahead of us if path busy
finish  = ready + ser_s ;  busy_until[i] = finish
arrival = finish + (base_rtt_ms/1000)/2          # + one-way propagation
```

The clock then advances by exactly `1/fps` **regardless of how long delivery took**
— the real-time cadence. Overdriving a path makes `busy_until` run ahead of `now`,
so subsequent frames inherit a growing standing queue and latency diverges.

**Synthesized per-path observation fields** (kept consistent with the queue state):
`srtt = base_rtt + standing-queue delay`, `buffer_occ = queue_s * cap * 1e6 / 8`,
`cwnd ≈ cap * 1e6 / 8 * base_rtt/1000` (bandwidth-delay product). Light network
loss when a path drops below ~50% of its baseline capacity (`_PathTrace.network_loss`).

**Aggregate EWMAs** (updated each `step_frame`; these gains are the canonical
reference the C++ body also uses):

| Quantity | Update rule |
|---|---|
| jitter EWMA | `0.7·old + 0.3·|Δlatency|` |
| app-loss EWMA | `0.9·old + 0.1·[late]` |
| throughput EWMA | `0.7·old + 0.3·goodput` |
| aggregate RTT EWMA | `0.8·old + 0.2·latency` (split-weighted srtt when available) |
| per-path goodput EWMA | `0.6·old + 0.4·share_goodput` |

**Default mock topology** (`configs/default.yaml`, legacy 3-path): base rates
`8/4/2 Mbps`, one-way delays `10/17/30 ms` (base RTT = `2×` one-way),
`cross_frac 0.45/0.65/0.35`.

### 1.5 Backend B — `Ns3DataPlane` (real packet-level NS-3)

Drives `ns3/realtime_mpquic.cc` over the **ns3-ai struct-based shared-memory
bridge** (Linux/WSL2 only; the pybind `.so` is `cpython-312`). One long-lived NS-3
process serves the whole run; episodes reset **in-band** (ns3-ai allows only one
shared-memory creator per process).

**Topology.** `RealtimeController::Build()` constructs a dumbbell of `N` parallel
paths between one client and one server. Per path `i`:
- a `PointToPointHelper` link with configured **bottleneck DataRate** and **one-way
  propagation Delay**;
- a **FqCoDel** AQM (`TrafficControlHelper`) so queueing delay/loss emerge
  endogenously;
- a dedicated `/24` subnet;
- a `RealtimeSource` (server-side sender) + `RealtimeSink` (client-side receiver)
  forming one long-lived subflow;
- an **on/off UDP cross-traffic** flow (`BuildCrossTraffic`), mean rate
  `cross_frac × link_rate`, 1200 B packets, exponentially-distributed on/off times
  with per-path phase shifts (on-mean `0.6 + 0.2·i` s, off-mean `0.8 + 0.3·i` s).

**Selectable per-path transport** (`transport: tcp | udp`, config knob forwarded to
the body; `--ns3-transport` overrides):
- `tcp` (default): a stock single-path NS-3 TCP connection (`TcpSocketFactory`,
  build-default congestion control). Reliable/in-order; network loss surfaces as
  added latency. Byte-identical to the historical scenario.
- `udp`: unreliable datagrams with **explicit app-layer deadline drop**
  (`RealtimeSource::DrainUdp`): a frame's share is fragmented into ≤1400 B
  datagrams (each with a 24-byte `UdpFrameHeader`: frame id, gen time, offsets);
  any share already past its deadline is *written off* rather than retransmitted —
  the closer analogue to QUIC's unreliable datagrams.

**Per-frame completion via byte-watermark tracking.** TCP delivers a single
reliable in-order stream carrying many pipelined frames, so per-frame latency needs
a watermark scheme (no per-packet tagging):
- each path keeps a monotonically increasing `m_pathEnq[i]` (bytes ever handed to
  the subflow);
- a frame's `b`-byte share enqueues a record `{watermark = m_pathEnq[i], frameId,
  bytes = b}`;
- the sink reports its cumulative received byte count; a share is delivered when
  cumulative-received ≥ its watermark;
- a frame completes when all its non-empty shares complete
  (`CompleteFrame`); frames past the deadline are swept out by `ExpireLateFrames`
  (loss).

**Transport-state signals per path** (read from the real sockets): `cwnd` from the
`CongestionWindow` trace; `srtt` = EWMA (`0.85·old + 0.15·sample`) of the `RTT`
trace; `bufferOcc` = app bytes queued but not yet handed to TCP; per-path goodput
EWMA; per-path network loss = `lost/(tx+lost)` from `FlowMonitor` — computed only
on the **1 s app cadence** (a full sweep is expensive) and cached between sweeps.

**Decision loop** (`RealtimeController::Decide`, once per frame):
1. `ExpireLateFrames(now)` — retire deadline-past frames.
2. On an app boundary: `RefreshNetworkLoss()` (FlowMonitor sweep).
3. `FillObservation(env)` — populate `EnvStruct`; set `done` if the frame budget is
   exhausted.
4. **Send** `EnvStruct`, **receive** `ActStruct` (or, in `--selftest`, synthesize an
   even-split fixed-bitrate action with no bridge).
5. Dispatch: `ACT_TERMINATE` → stop; `ACT_RESET` or `done` → reset episode counters
   & EWMAs (network stays warm — sockets/queues/CC persist across episodes);
   `ACT_STEP` → `GenerateFrame(act)`, increment counter, schedule next `Decide` at
   `now + 1/fps`.

Episode length = `round(fps · episode_seconds)` frames; a warm-up delay (default
1 s) precedes the first decision so TCP establishes and RTT estimates go live.

### 1.6 Optional non-stationary dynamics (both backends)

Off by default (`DynamicsConfig.enabled = False`); when disabled the dynamics RNG
is never drawn and behavior is byte-identical to the static scenario. Four coupled
mechanisms make *which path to send on* a genuinely time-varying decision. All are
**deterministic per seed**, advanced **once per frame**, with event probability
`1 − e^{−rate·Δt}` and a **fixed draw order (regime, burst, corr, churn)** so mock
and C++ stay behaviorally comparable.

| Mechanism | Parameters | Effect |
|---|---|---|
| **Churn** | `churn_up_rate`, `churn_down_rate`, `min_active` | Per-path on/off Markov chain (floored at `min_active`). A dead path reports a dead row + `path_active = 0`; bytes routed onto it are **lost**. |
| **Regime** | `regime_rate`, `regime_lo/hi` | Per-path capacity multiplier resampled `~U(lo, hi)` at Poisson change-points → abrupt best-path swaps. |
| **Burst** | `burst_rate`, `burst_intensity`, `burst_duration_s` | Poisson per-path capacity collapse to `burst_intensity` for `burst_duration_s`. |
| **Corr** | `corr_groups`, `corr_rate`, `corr_intensity`, `corr_duration_s` | Groups of path indices degrade together (shared bottleneck). |

Effective per-path capacity multiplier = `regime_mult × burst_mult × corr_mult`
(`_cap_mult` in mock; `CapMult` in C++).

**How the NS-3 body implements each (important, differs from a naive port):**
- **Churn is NOT a `DataRate` collapse.** A dead path is silenced by **drop-all
  receive error models on both link directions** (data + ACKs): packets keep
  serializing normally and die at the receiver, so a revived path is usable
  instantly (no packet stranded mid-serialization for seconds). On churn-out the
  TCP subflow is **torn down** (queued shares written off as dropped via
  `WriteOffPathShares`), and **reconnected** on revival with send-buffer/watermark
  reset to zero in lockstep with the sink. (UDP mode needs no connection handling.)
- **Regime/burst/corr** scale the per-path bottleneck `DataRate` **and the
  cross-traffic rate** by the same multiplier (`ApplyPathRate`), so cross traffic
  stays a fixed fraction of current capacity (multiplicative, like the mock). Under
  dynamics the stock `OnOffApplication` is replaced by a rescale-safe
  `CrossTrafficApp` (the stock app `NS_FATAL`s when rescaled every frame).
- **Seasonal envelope.** The C++ also mirrors the mock's sinusoidal capacity
  envelope + noise (`EnvelopeMult`, folded into `ApplyPathRate`); amp/period/phase
  formulas are duplicated from `ExperimentConfig.mock_dataplane` in
  `src/train/config.py` — **keep them in sync**.
- **Backlog bound.** Every path carries a WebRTC-style bound
  (`m_maxBufferBytes` = one deadline's worth of bytes at nominal rate): a stalled
  path drops fresh shares rather than accumulating stale backlog.
- Under dynamics, FlowMonitor's `MaxPerHopDelay` is lowered to the deadline so loss
  is counted at real-time timescales, and a path gone quiet > 1 deadline decays its
  goodput EWMA (`0.9×`) so stalls are visible.

**Parity guard:** `scripts/parity_check.py` runs the same neutral even-split policy
through both backends (mock in-process, NS-3 via `--selftest`) and asserts the loss
regimes match. Residual approximations: independent RNG streams (behaviorally
equivalent, not frame-identical); real TCP/UDP transport vs. the mock's analytical
queue; corr-group indices ≥ path count are dropped.

### 1.7 Config → topology wiring

One YAML drives both backends (`src/train/config.py`). The `topology.paths` list
(`{rate, delay, cross_frac}`) becomes NS-3 link attributes **and** the mock's
per-path baselines (`rate → base_mbps`, `delay → base_rtt = 2×delay`,
`cross_frac → mean cross`). The path list is serialized to the NS-3 CLI (an empty
arg keeps the C++ built-in default). `dynamics:` and `transport:` are forwarded the
same way.

---

## 2. The agents (SAC + baselines) and their I/O

### 2.1 Observation builders (`src/ns3env/realtime_env.py`)

Pure numpy, no torch. Features are normalized to roughly `[0, 1]` and clipped.

**App observation** — `build_app_obs`, **dim 5**:

| Idx | Feature | Normalization |
|---|---|---|
| 0 | current bitrate | `(kbps − min)/(max − min)` |
| 1 | aggregate RTT | `rtt_ms / latency_norm_ms` (200) |
| 2 | aggregate jitter | `jitter_ms / jitter_norm_ms` (50) |
| 3 | aggregate loss | `loss` (∈[0,1]) |
| 4 | aggregate throughput | `throughput_mbps / cap_mbps` |

The App agent sees only **aggregate, sender-observable** state. It deliberately
does **not** see per-path detail (that's the Path agent's job) and **not** episode
progress/time-to-go — a deployable policy must be horizon-agnostic (a real call has
unknown duration). The fixed training horizon is handled as a *truncation* (§3.4).

**Flat Path observation** — `build_path_obs`, **dim `4 + 5N`** (24 for
N=4, 34 for N=6):
- Global block (4): `[bitrate_norm, rtt/latency_norm, loss, throughput/cap]`. The
  first element is the **App agent's current target bitrate** — this is the
  hierarchical coupling (the Path agent is conditioned on the load it must schedule).
- Per-path block (5 × N): for each path `[cwnd/200000, srtt/latency_norm,
  bufferOcc/200000, path_throughput/cap, path_loss]`.

**Structured Path state** — `build_path_state`, for the scoring agent.
The flat vector hard-codes path count/order and cannot represent a *changing* set.
The same features are instead a `PathState(glob, paths, mask)` triple:
- `glob`: `(4,)` — the same global block.
- `paths`: `(N, 6)` — the five flat per-path features **plus the liveness flag**.
- `mask`: `(N,)` — `1.0` live / `0.0` churned out.

`N` is fixed at the candidate cap for tensor shapes; the *mask*, not the array
width, says which rows are live.

### 2.2 Generic SAC core (`src/rl/sac_agent.py`)

Continuous Soft Actor-Critic operating entirely in a **normalized action space
`[−1, 1]^d`**. Off-policy (replay buffer amortizes expensive NS-3 frames) +
max-entropy objective. Reused by both agents through thin wrappers.

**Actor — `GaussianPolicy`** (tanh-squashed diagonal Gaussian):
```
obs → MLP(256,256, ReLU) → { mean; log_std (clamped [−20, 2]) }
x ~ Normal(mean, exp(log_std)) ;  a = tanh(x)                       # a ∈ (−1,1)^d
log π(a|s) = Σ_j [ log Normal(x_j) − log(1 − a_j² + 1e-6) ]         # tanh correction
```
Deterministic mode (eval) returns `tanh(mean)`.

**Critics — `QNetwork`** (twin, clipped-double-Q): two MLPs `Q1,Q2 : (obs⊕act)→ℝ`,
each `Linear(d_obs+d_act,256)→ReLU→Linear(256,256)→ReLU→Linear(256,1)`. A Polyak
target copy is held for bootstrapping.

**One gradient step** (`_update_once`, batch 256):
```
# critic (γ=0.99)
a', logπ' ← actor(s') ; y = r + γ(1−done)[ min(Q1ᵗ,Q2ᵗ)(s',a') − α·logπ' ]
L_critic = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
# actor
ã, logπ ← actor(s) ; L_actor = E[ α·logπ − min(Q1,Q2)(s,ã) ]
# temperature (auto entropy), target H̄ = −d_act
L_α = − E[ log α · (logπ + H̄) ]
# Polyak target: θᵗ ← (1−τ)θᵗ + τθ,  τ=0.005
```
Adam lr `3e-4` for all three losses. `α = exp(log_alpha)`.

**Exploration / warm-up.** `select_action` returns **uniform random** in `[−1,1]^d`
until `start_steps` (1000) transitions are stored; updates begin once the buffer
holds `≥ max(batch_size, update_after)` (1000); `updates_per_step` (1) gradient
steps per frame. `ReplayBuffer` is a preallocated ring of
`(obs, act, rew, next_obs, done)`, capacity 200 000, storing normalized actions.

### 2.3 App agent wrapper (`src/rl/app_agent.py`)

`d_act = 1`. Maps the normalized action affinely to a bitrate:
```
kbps = min_kbps + (a+1)/2 · (max_kbps − min_kbps)      # defaults 300–6000
```
`select(obs) → (kbps, raw_action)`; the **raw `[−1,1]` action is what is stored**
in the buffer / scored by the critic.

### 2.4 Path agent wrapper (`src/rl/path_agent.py`)

Dispatches on `arch`:

- **`"flat"` (default)**: wraps `SACAgent` with `d_act = N`. The normalized action
  becomes a split via **temperature-scaled softmax** `split = softmax(a/τ)` (τ=1).
- **`"scoring"`**: wraps `ScoringSACAgent` (below); the per-path latent becomes a
  **masked softmax** over active paths only.

Both store the **pre-softmax `[−1,1]` vector** as the action. The observation
includes the App bitrate → hierarchy.

### 2.5 Scoring (dynamic-input) SAC (`src/rl/scoring_sac_agent.py`)

Permutation-equivariant SAC for a **variable/changing** path set (handles churn;
no relearning per path ordering). Adapts the SCION sibling's path-scoring DQN from
discrete argmax to continuous SAC. Consumes `(glob, paths, mask)`.

- **Actor — `ScoringGaussianPolicy`.** A **shared** per-path encoder maps
  `glob ⊕ path_i` → a tanh-squashed Gaussian latent per path → SAC action in
  `[−1,1]^N` (exactly the flat agent's raw action). The env split is a **masked
  softmax** over active paths. Log-prob/entropy summed over **active paths only**
  (`logp * mask`), so dead paths contribute no density or gradient.
- **Critic — `ScoringQNetwork` (twin, DeepSets).** Encodes `glob ⊕ path_i ⊕
  latent_i` per path, **masked-mean-pools** across paths, maps the pooled embedding
  to a scalar Q — permutation-invariant, variable-N.
- **Entropy target scales with active-path count** per sample
  (`H̄ = −1 · mask.sum()`), not a fixed `−N`, since the effective action dimension
  shrinks as paths churn.
- **`StructuredReplayBuffer`**: rectangular fixed-`N` arrays `(glob, paths, mask,
  act, rew, next_*, done)` — the mask, not ragged padding, carries liveness.

Checkpoints are tagged `arch: "scoring"`; the critic/actor/target/temperature
update math is otherwise identical to §2.2.

### 2.6 Heuristic baselines (`src/train/evaluate.py`)

Non-learned scheduling policies, all **mask-aware** (a churned-out path gets zero
weight) so they are not handicapped by churn they can't see. The `even` / `single`
/ `proportional` / `random` baselines pair with a common **reactive bitrate rule**
(`target = 0.9 × recent aggregate goodput`, clamped, `_heuristic_bitrate`) so
comparisons isolate *scheduling* quality:

| Baseline | Bitrate rule | Split rule |
|---|---|---|
| `even` | reactive goodput | uniform over active paths |
| `single` | reactive goodput | whole frame on the highest recent-throughput active path (argmax) |
| `proportional` | reactive goodput | ∝ recent per-path throughput (active only) |
| `random` | reactive goodput | fresh Dirichlet-uniform split over active paths each frame (seeded) |
| `webrtc` | **GCC** (`_GccBitrate`) | ∝ recent per-path throughput (active only) |
| `learned` | trained App agent | trained Path agent (**deterministic**) |

**`webrtc`** is a realistic reference: a stateful WebRTC-style **Google Congestion
Control** bandwidth estimator (`_GccBitrate`) driving a proportional multipath
split. Each App tick (≈ `app_period_s`, matching GCC's ~1 s update period) it
runs the two classic GCC controllers and takes the more conservative move — the
**loss-based** rule (`loss > 10%` → `est *= (1 − 0.5·loss)`; `loss < 2%` →
eligible to increase) and a **delay-based** rule (queuing delay = RTT over the
tracked min-RTT past 50 ms → back off to `0.85 × received rate`) — otherwise
increasing by 8% (`est *= 1.08`). Because it carries its own estimate and only
reads goodput on the delay-back-off branch, it **probes** for capacity and never
spirals to the bitrate floor the way the reactive goodput rule does (§4.3). It is
stateful, so each policy gets its own instance and `_rollout` re-arms it per
episode via a `reset()` hook.

The learned policy is reconstructed by `_learned_policies`, which reads the
Path checkpoint's `arch` tag to rebuild the flat or scoring agent and feed it
the matching (flat vs. structured) observation.

---

## 3. The training pipeline

### 3.1 Reward functionals (`src/ns3env/qoe.py`)

**Perceptual quality (VMAF, 0–100).** Default is a log rate-quality curve fit
through Netflix-style anchors `(300 kbps, 25)` and `(4300 kbps, 92)`:
```
VMAF(R) = a·ln(R) + b,   clipped [0,100]   (a,b solve the two anchors)
```
The VMAF term is a **pluggable hook** (`compute_qoe_reward(..., vmaf_fn=…)`). With
`--learned-vmaf` (or `reward.use_learned_vmaf: true`) it is replaced by a
**WebRTC-grounded learned surrogate** (`qos_vmaf_reward.py` + `reward_model.npz`,
adapted by `learned_vmaf.py`): a multilinear interpolant over real WebRTC VMAF
measurements mapping `(bitrate, loss%, delay_ms, jitter_ms) → VMAF`.

**App QoE reward** (`compute_qoe_reward`), credited per window, clipped `[−2, 1]`:
```
default (log curve):  QoE = a·VMAF(bitrate)/100 − b·lat − c·jit − d·loss
learned surrogate:    QoE = a·VMAF(bitrate,loss,delay,jitter)/100 − b·lat − c·jit
                            + e·(1−loss)·util
```
where `lat = clip(latency/latency_norm, 0, 2)`, `jit = clip(jitter/jitter_norm,
0, 2)`, `loss = clip(loss, 0, 1)`. Default weights `a=1.0, b=0.5, c=0.5, d=1.0`;
normalizers `latency_norm = 200 ms`, `jitter_norm = 50 ms`. Two learned-only
adjustments:
- the explicit `− d·loss` is **dropped** (the learned VMAF already folds in loss —
  avoid double-counting);
- a **utilization term** `+ e·(1−loss)·util` (`util = clip(bitrate/util_norm, 0,
  1)`, `util_norm ≈ 3300 kbps`, `e_util = 0.25` in the four-path/dynamic configs)
  restores the delivered-bits gradient the ~bitrate-flat surrogate erases, gated by
  `(1−loss)` so it only pays for on-time bits. `e_util` defaults to `0.0` (off) and
  is **only applied with a learned scorer**.

**Path reward** (`compute_path_reward`), per frame, clipped `[−2, 1]`:
```
R_p = (1 − loss) − b·clip(latency/latency_norm,0,2) − c·clip(jitter/jitter_norm,0,2)
```
The quality term is **intentionally excluded** — the Path agent doesn't set
bitrate, so rewarding it for quality would be miscredited. It is rewarded purely
for delivering the given frame quickly and intact. This clean decomposition is what
gives each agent a well-attributed gradient.

### 3.2 Two-timescale credit assignment (`realtime_env.py`)

The env maintains a **reward window** — running lists of per-frame `latency`,
`jitter`, `loss` for the frames governed by the *current* bitrate. Every `step`
appends to it. At an app boundary the training loop calls `pop_app_window_reward()`,
which aggregates the window (mean latency/jitter/loss), scores it with the App QoE
functional, **clears the window**, and returns the scalar reward + unweighted
components. That reward is credited to the **previous** App action — the bitrate
that actually governed those frames (a correct temporally-extended reward for the
slow agent). The Path agent is rewarded immediately each frame.

### 3.3 The dual-cadence loop (`src/train/hierarchical_train.py`)

`run_training` builds the data plane, env, and both agents (sizing the Path agent
from `env.num_paths` after the first `reset`), then runs episodes. Per episode
(`_run_episode`):

```
obs ← env.reset(seed)
target_kbps ← obs.current_bitrate
prev_app_obs, prev_app_raw ← None, None
while not done:
    if obs.app_decision_due:                       # ── slow (App) cadence ──
        app_obs ← build_app_obs(obs)
        if prev_app_obs is not None:               # credit the window just closed
            r_app, comps ← env.pop_app_window_reward()
            app.store(prev_app_obs, prev_app_raw, r_app, app_obs, done=False)
            app.update()
        target_kbps, prev_app_raw ← app.select(app_obs)
        prev_app_obs ← app_obs

    p_obs  ← path_obs(obs, target_kbps, arch)       # ── fast (Path) cadence ──
    split, p_raw ← path_agent.select(p_obs)
    next_obs, r_p, done, info ← env.step(target_kbps, split)
    p_next ← path_obs(next_obs, target_kbps, arch)
    path_agent.store(p_obs, p_raw, r_p, p_next, done=False)
    path_agent.update()
    obs ← next_obs
# episode end: credit the final (partial) App window, also done=False
```

Correctness points:
- The App transition's *next state* is the app-obs at the **next** boundary — a
  genuine temporally-extended tuple over the window.
- `path_obs(...)` builds the flat vector (flat arch) or the `PathState`
  triple (scoring arch) to match the agent.
- The App agent's warm-up thresholds are scaled down by `frames_per_app`
  (`_app_sac_config`: `start_steps`/`update_after` ÷ 30, floored at 50) since it
  steps 30× less often.

### 3.4 Time-limit handling (truncation, not terminal)

The episode horizon is a **truncation** — the call could continue past it — so
**every transition (including the final partial App window) is stored with
`done = False`**, bootstrapping the value off the next state rather than cutting the
return. This is the standard finite-horizon correction (Pardo et al. 2018) and
keeps both policies horizon-agnostic (consistent with the App agent not observing
episode progress).

### 3.5 Checkpointing, logging, lifecycle

- Checkpoints (`app.pth`, `path.pth`) are written **after every episode**, so
  an interrupted run leaves a usable latest model. `--resume` reloads the latest and
  skips the uniform-random warm-up (the buffer restarts empty; updates resume once
  it refills) — enabling long trains split across interruptible chunks.
- Per-episode logging: mean QoE, mean path reward, mean bitrate/latency/loss/
  VMAF; a `stats.json` history is written to the run dir.
- A `finally` block guarantees `dp.close()` (→ `ACT_TERMINATE`, shared-memory
  release) even on exceptions.

Default hyperparameters (`SACConfig`, overridable per YAML `sac:` block): hidden
256, γ 0.99, τ 0.005, lr 3e-4, batch 256, buffer 200k, start_steps 1000,
update_after 1000, updates_per_step 1, auto_entropy true.

---

## 4. The evaluation experiments

`src/train/evaluate.py::run_evaluation` rolls out policies **without learning**
(agents act deterministically) for `episodes` episodes on the chosen backend, and
dumps everything to `<out_dir>/evaluation_results.json` (turned into figures by
`evaluation/generate_figures.py`). Default eval seed `1000`; each policy's episode
`e` uses `seed + e`. The reward machinery reused is identical to training
(`pop_app_window_reward`), so reported QoE matches the training objective.

### 4.1 What each rollout records

Per policy, across all episodes (`_rollout`):
- **Aggregate stats** (mean/std/p50/p95/min/max) of QoE, VMAF, latency, jitter,
  loss, bitrate, throughput.
- **Decision (inference) time** of each agent call, measured *around the policy
  callable only* — so the learned policy captures observation-build + forward pass,
  and the baselines capture their cheap heuristic compute. Reported as
  `app_decision_ms` and `path_decision_ms` (the printed table shows the
  Path p50).
- **Dynamics diagnostics per frame**: `active_paths` (live path count) and
  `split_entropy` (nats; `0` = one path, `ln k` = even over `k` paths).
- **`deadline_miss_rate`**: fraction of frames with `loss ≥ 0.999`.
- One **representative per-frame trace** (episode 0) for time-series/split plots.

Policies are warmed up (10 dummy calls) before timing to stabilize torch's
first-call latency.

### 4.2 The baseline comparison (always run)

`even`, `single`, `proportional`, `random` (reactive bitrate) and `webrtc` (GCC
bitrate), plus `learned` **iff** both `--app` and `--path` checkpoints are supplied
(otherwise baselines-only). The reactive-bitrate baselines isolate **scheduling
quality** because they share the same bitrate rule; `webrtc` is the realistic
end-to-end reference (GCC + proportional split). Expected reading: on a
no-dominant-path topology `single` is quality-capped (no one path saturates VMAF),
`proportional` eats the latency-trap path's delay, and `learned` aggregates
capacity while down-weighting high-RTT/queue paths.

### 4.3 The ablation study (`--ablation`, requires both checkpoints)

Adds single-agent variants that disable exactly one learned agent by swapping in a
heuristic counterpart, to isolate each agent's marginal contribution:

| Variant | Bitrate source | Split source | Isolates | Compare against |
|---|---|---|---|---|
| `app_only` | learned bitrate | `even` split (Path **off**) | value of the **learned bitrate** | `even` (same split, heuristic vs learned bitrate) |
| `path_only_gcc` | **GCC** (App **off**) | learned split | value of the **learned scheduler** | `webrtc` (same GCC bitrate + proportional split) |
| `learned` | learned | learned | full system | all of the above |

**Why the App-off ablation uses GCC, not the reactive goodput rule.** The reactive
bitrate rule (`0.9 × recent goodput`) is a *feedback loop*: a split that
under-utilizes the network delivers few bytes → low goodput → low bitrate → even
fewer bytes offered, a self-reinforcing spiral down to `min_bitrate_kbps` (the VMAF
floor). Paired with the learned split it traps the whole variant at the floor and
**masks** the scheduler's contribution (an earlier `path_only` variant sat at the
even-split floor regardless of scheduling skill). Swapping in the GCC driver
(§2.6) — which probes for capacity and holds a realistic load — lets the learned
split run at a sensible bitrate, so the QoE gap vs. the matched `webrtc` reference
(same GCC bitrate, proportional split) reflects scheduling quality alone.

**The headline result the ablation is designed to expose.** On the *static*
four-path topology, `app_only` reaches ~98% of the full system's QoE — the optimal
split barely moves, so the scheduler wins little. On the **non-stationary**
`configs/dynamic.yaml` scenario (with the scoring Path agent), `app_only`
**collapses** (an even split shoves the encoder's bitrate onto congested/dead
paths), while the full learned pair stays far ahead. This contrast is the argument
for (a) making the network non-stationary and (b) giving the Path agent a model
that ingests the changing path set.

### 4.4 The two headline scenarios

| Scenario | Config | Paths | Dynamics | Path arch | Purpose |
|---|---|---|---|---|---|
| Static no-dominant | `configs/four_path.yaml` | 4 | off | `flat` | Show aggregation beats single-best / proportional; scheduler earns a modest QoE gain. Ablation → `app_only` ≈ full system. |
| Non-stationary | `configs/dynamic.yaml` | 6 (min 3 active) | churn + regime + burst + corr `[[4,5]]` | `scoring` | Make the split decision matter; ablation → `app_only` collapses. |

Both run on **either** backend (mock for iteration/CI, NS-3 for final results). The
`dynamic.yaml` NS-3 run exercises the mirrored C++ dynamics (§1.6); `parity_check.py`
guards that the two backends agree on loss regime.

### 4.5 CLI entry points

```bash
# train (mock, static four-path):
uv run python train.py --config configs/four_path.yaml --backend mock --episodes 50
# train (mock, dynamic + scoring):
uv run python train.py --config configs/dynamic.yaml   --backend mock --episodes 200
# evaluate with ablation:
uv run python evaluate.py --config configs/dynamic.yaml --backend mock \
    --app runs/<run>/app.pth --path runs/<run>/path.pth --ablation
# NS-3 backend + UDP subflows:
uv run python train.py --config configs/dynamic.yaml --backend ns3 --ns3-transport udp
```

`--backend {mock,ns3}`, `--ns3-transport {tcp,udp}`, `--learned-vmaf`, `--resume`,
`--figures` are the main knobs. NS-3 requires a rebuild first
(`scripts/install_ns3_example.sh`, then `./ns3 run "ns3ai_realtime_mpquic
--selftest"` to validate C++ alone).

---

## 5. Appendix: exact dimensions, constants, file map

**Dimensions (N = candidate path count):**

| Quantity | Value |
|---|---|
| App observation dim | 5 (horizon-agnostic) |
| App action dim | 1 → bitrate ∈ [300, 6000] kbps |
| Flat Path obs dim | `4 + 5N` (24 @ N=4, 34 @ N=6) |
| Scoring Path state | glob `(4,)`, paths `(N, 6)`, mask `(N,)` |
| Path action dim | N → softmax split (masked over active paths in scoring) |
| Bridge exchanges/sec | `fps` = 30 |
| App decisions/sec | 1 (every 30 frames) |

**Key constants:**

| Constant | Default | Where |
|---|---|---|
| fps / app period / deadline | 30 / 1 s / 180 ms | config, C++ CLI |
| bitrate range / init | 300–6000 / 1500 kbps | `VideoSourceConfig` |
| keyframe interval / size jitter | 30 / 0.25 | `VideoSourceConfig` |
| reward `a,b,c,d` | 1.0, 0.5, 0.5, 1.0 | `QoEWeights` |
| `e_util` / `util_norm_kbps` | 0.0 (off) / 3300; 0.25 in four-path & dynamic (learned-VMAF only) | `QoEWeights` |
| latency / jitter normalizers | 200 / 50 ms | `QoEWeights` |
| VMAF anchors (log curve) | (300, 25), (4300, 92) | `qoe.py` |
| SAC hidden / γ / τ / lr | 256 / 0.99 / 0.005 / 3e-4 | `SACConfig` |
| batch / buffer / start_steps / update_after | 256 / 200k / 1000 / 1000 | `SACConfig` |
| cwnd / buffer obs normalizers | 200 000 B | `realtime_env.py` |
| `path_arch` | `flat` (default) / `scoring` | `run:` |
| `transport` (NS-3 protocol) | `tcp` / `udp` | `run:`, `--ns3-transport` |
| `dynamics` | off by default (both backends) | `DynamicsConfig` |
| kMaxPaths | 8 | `realtime_mpquic.h` |
| EWMA gains (both backends) | jitter .3, loss .1, thr .3, rtt .2, path-thr .4; socket/UDP sRTT .15 | `dataplane.py`, `realtime_mpquic.cc` |

**File map:**

| Path | Role |
|---|---|
| `ns3/realtime_mpquic.h` | `EnvStruct`/`ActStruct` wire contract, `kMaxPaths`, `ActCommand` |
| `ns3/realtime_mpquic.cc` | NS-3 scenario: topology, TCP/UDP subflows, frame gen/striping, watermark completion, decision loop, dynamics |
| `ns3/realtime_mpquic_py.cc` | pybind11 binding of structs + interface |
| `ns3/CMakeLists.txt`, `scripts/install_ns3_example.sh` | build + install into ns-3-dev |
| `scripts/parity_check.py` | mock⇄NS-3 loss-regime parity guard |
| `scripts/wedge_repro.py` | minimal repro for churn/rescale failure modes |
| `src/ns3env/dataplane.py` | `DataPlane` ABC, `MockRealtimeDataPlane`, `Ns3DataPlane`, `FrameObs`/`FrameResult`, `DynamicsConfig` |
| `src/ns3env/video_source.py` | frame-size model (mirrors C++) |
| `src/ns3env/qoe.py` | VMAF curve, App QoE reward (pluggable `vmaf_fn`, optional util term), path reward |
| `src/ns3env/qos_vmaf_reward.py` + `reward_model.npz` | vendored learned QoS→VMAF surrogate |
| `src/ns3env/learned_vmaf.py` | adapter: learned surrogate → `vmaf_fn` |
| `src/ns3env/realtime_env.py` | observation builders, reward window, `PathState` |
| `src/rl/sac_agent.py` | generic flat SAC (actor, twin critics, targets, entropy tuning) |
| `src/rl/scoring_sac_agent.py` | permutation-equivariant SAC (per-path actor + DeepSets critic) |
| `src/rl/replay_buffer.py` | `ReplayBuffer` + `StructuredReplayBuffer` |
| `src/rl/app_agent.py`, `src/rl/path_agent.py` | action-space wrappers (Path agent dispatches flat vs scoring) |
| `src/train/config.py` | YAML → typed config; backend factories |
| `src/train/hierarchical_train.py` | dual-cadence training loop |
| `src/train/evaluate.py` | rollout + baselines + ablation |
| `train.py`, `evaluate.py` | thin CLIs (`--backend`, `--ns3-transport`, `--learned-vmaf`, `--resume`) |
| `evaluation/generate_figures.py` | figures from `evaluation_results.json` |
| `configs/four_path.yaml`, `configs/dynamic.yaml` | the two headline scenarios |
| `docs/TUNING_DYNAMICS.md` | practical guide to tuning the dynamics parameters |
| `tests/` | mock-only unit + smoke tests (`test_dynamics.py`, `test_scoring_agent.py`, `test_qoe.py`, …) |

---

*End of implementation reference.*
