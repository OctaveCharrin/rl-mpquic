# Hierarchical Reinforcement Learning for Real-Time Video over Multipath QUIC: A System Architecture Report

**Project:** `rl-mpquic`
**Scope:** A complete description of the system — the packet-level network
simulator (the "body"), the shared-memory bridge, the observation/reward layer,
and the dual-agent Soft Actor-Critic controller (the "brain") — at a level of
detail sufficient to reproduce, extend, or critique the design.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Design philosophy and system overview](#2-design-philosophy-and-system-overview)
3. [The simulated network (NS-3 "body")](#3-the-simulated-network-ns-3-body)
4. [The shared-memory bridge (ns3-ai)](#4-the-shared-memory-bridge-ns3-ai)
5. [The Python data-plane abstraction](#5-the-python-data-plane-abstraction)
6. [The hierarchical environment: observations and rewards](#6-the-hierarchical-environment-observations-and-rewards)
7. [The reinforcement-learning model](#7-the-reinforcement-learning-model)
8. [The training loop](#8-the-training-loop)
9. [Evaluation and baselines](#9-evaluation-and-baselines)
10. [End-to-end worked example: the life of one frame](#10-end-to-end-worked-example-the-life-of-one-frame)
11. [Design rationale, assumptions, and limitations](#11-design-rationale-assumptions-and-limitations)
12. [Appendix: dimensions, constants, and file map](#12-appendix-dimensions-constants-and-file-map)

---

## 1. Abstract

This system studies *joint application-and-transport adaptation* for real-time
interactive video (a WebRTC-like conferencing flow) delivered over a multipath
transport. A sender must continuously answer two coupled questions: **how much
video to produce** (the encoder target bitrate, which trades perceptual quality
against the load it imposes) and **how to spread that video across multiple
network paths** (the per-path scheduling split, which trades each path's capacity
and delay against the others). These decisions operate on different natural
timescales — bitrate adaptation is comparatively slow and global, packet
scheduling is fast and local — and they interact: the right split depends on how
much load the encoder is generating, and the right bitrate depends on what the
paths can collectively absorb.

We cast this as a **two-agent hierarchical control problem** solved with **Soft
Actor-Critic (SAC)**. An *App agent* sets the bitrate once per second; a
*Transport agent* sets the path split once per video frame (~33 ms) and is
explicitly conditioned on the App agent's current bitrate, making the pair
hierarchical rather than independent. The controller is trained against a
**packet-level NS-3 simulation** in which "Multipath QUIC" is abstracted as a set
of independent congestion-controlled subflows carrying application-striped video,
each subflow crossing its own bottleneck shared with bursty background traffic.
A second, pure-Python **trace-driven backend** mirrors the simulator's interface
exactly, enabling fast iteration and continuous testing without compiling NS-3.
The reward is a **real-time Quality-of-Experience (QoE)** functional combining a
VMAF perceptual-quality term with penalties on latency, jitter, and deadline-miss
loss.

---

## 2. Design philosophy and system overview

### 2.1 The brain/body split

The codebase is organized around a strict separation between *network mechanics*
and *decision logic*:

- The **body** (C++, NS-3) owns physics only: topology, sockets, congestion
  control, queues, packet loss, frame generation, and byte striping. It contains
  **no reinforcement-learning logic** and no notion of "reward." Its sole job is
  to advance the simulation by one video frame, report what it measured, and
  apply whatever action it is told.
- The **brain** (Python, PyTorch) owns everything cognitive: observation
  construction, normalization, reward computation, the replay buffer, the neural
  networks, and the training loop.

The two communicate over a narrow, explicitly versioned contract: two C structs
(`EnvStruct`, `ActStruct`) exchanged through shared memory. This boundary is the
single most important architectural decision in the system, because it lets the
RL research proceed independently of the C++/NS-3 integration risk, and it lets a
*mock* implementation of the body stand in for the real one with byte-for-byte
interface compatibility.

```
            ┌─────────────────────────── PYTHON (the "brain") ───────────────────────────┐
            │  train.py / evaluate.py        thin CLIs                                     │
            │        │                                                                     │
            │  src/train/   config loader · dual-cadence loop · evaluation+baselines       │
            │        │                                                                     │
            │  src/ns3env/realtime_env.py    observation builders + QoE reward windows     │
            │        │                                                                     │
            │  src/rl/   SACAgent (generic)  ·  AppAgent / TransportAgent wrappers         │
            │        │                                                                     │
            │  src/ns3env/dataplane.py   DataPlane ABC                                     │
            │            ├── MockRealtimeDataPlane  (pure Python, trace-driven)            │
            │            └── Ns3DataPlane ──────────────┐                                  │
            └────────────────────────────────────────────┼──────────────────────────────┘
                                                          │ ns3-ai shared memory
                                                          │ (EnvStruct ⇄ ActStruct, per frame)
            ┌─────────────────────────────────────────────┼──────────── C++ (the "body") ─┐
            │  ns3/realtime_mpquic.cc   RealtimeController │                               │
            │        ├── N × RealtimeSource (TCP subflow senders)                          │
            │        ├── N × RealtimeSink   (per-path receivers)                           │
            │        ├── N × bottleneck links + FqCoDel AQM + OnOff UDP cross-traffic       │
            │        └── FlowMonitor (per-path loss)                                       │
            └──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The two-timescale contract

The bridge runs in **synchronous lockstep**: exactly one `EnvStruct → ActStruct`
exchange per **video frame**. At 30 fps this is one decision every ~33 ms of
simulated time. The Transport agent therefore acts every frame. The App agent
acts only when the body signals an **app-decision boundary** (`appDecisionDue`,
asserted once per `app_period_s`, i.e. every 30 frames at the defaults); between
those boundaries its previously chosen bitrate persists. This single-bridge,
frame-cadence design is what allows two agents on two timescales to share one
synchronous channel without a second IPC mechanism. The mechanics are detailed in
§6.2 and §8.

---

## 3. The simulated network (NS-3 "body")

All C++ lives in `ns3/realtime_mpquic.{h,cc,_py.cc}`. The file is vendored in the
repository and symlinked into the NS-3 source tree
(`$NS3_DIR/contrib/ai/examples/rl-mpquic`) by `scripts/install_ns3_example.sh`,
which also registers it with CMake and builds both the simulation binary
(`ns3ai_realtime_mpquic`) and the pybind11 module
(`ns3ai_realtime_mpquic_py`).

### 3.1 Topology and the abstraction of Multipath QUIC

NS-3 has no native QUIC implementation, and a faithful kernel-level MPQUIC would
be a multi-month integration. We therefore use **application-layer multipath over
independent transport subflows**, which preserves the properties that matter for
this control problem — *per-path congestion control, per-path queueing/delay,
per-path loss, and an application that decides how to stripe data across paths* —
while remaining tractable and honest about what is abstracted.

Concretely, `RealtimeController::Build()` constructs a dumbbell of `N` parallel
paths between a single **client** node and a single **server** node (the
canonical NS-3 scenario uses `N = 4`). For each path `i`:

- A `PointToPointHelper` link with a configured **bottleneck data rate** and
  **one-way propagation delay**. The canonical 4-path topology is a deliberately
  **no-dominant** mix in which *no single path saturates perceptual quality*, so
  aggregation across subflows is mandatory:

  | Path | Rate | Delay | `cross_frac` | Role |
  |---|---|---|---|---|
  | 0 | 3 Mbps | 10 ms | 0.40 | clean, low RTT (~1.8 Mbps usable) |
  | 1 | 3 Mbps | 15 ms | 0.45 | clean (~1.65 Mbps usable) |
  | 2 | 2.5 Mbps | 40 ms | 0.30 | **latency trap** — good rate, 40 ms delay (~1.75 Mbps usable) |
  | 3 | 2 Mbps | 20 ms | 0.55 | congested (~0.9 Mbps usable) |

  The aggregate usable capacity (~6 Mbps) sits below the VMAF quality knee
  (~3.3 Mbps target buys most of the perceptual gain) only if the paths are
  *combined*: each path alone tops out well under the knee, so a single-best-path
  policy is quality-capped. Path 2 is a deliberate **latency trap** — its
  throughput is attractive but its 40 ms propagation delay punishes any scheduler
  that splits purely in proportion to throughput, rewarding instead a policy that
  weighs per-path RTT/queue. (The earlier 3-path access topology —
  8/4/2 Mbps · 10/17/30 ms — was replaced precisely because its clean 8 Mbps path
  already saturated VMAF, making single-best near-optimal and leaving the learned
  scheduler nothing to win; see §11.)
- A **FqCoDel** active-queue-management discipline (`TrafficControlHelper`) on the
  link, so that queueing delay and loss *emerge endogenously* under load rather
  than being scripted.
- A dedicated IPv4 subnet (`10.1.(i+1).0/24`), a server-side **`RealtimeSource`**
  (a persistent TCP sender) and a client-side **`RealtimeSink`** (a receiver),
  forming one long-lived TCP connection — the "subflow."

The C++ `ScenarioConfig` carries a **built-in default** 4-path topology, but
`Ns3DataPlane` now forwards the YAML `topology:` block to the body (serialized as
a `paths` CLI argument), so the config drives the path list on *both* backends; an
empty argument keeps the built-in default. `configs/four_path.yaml` encodes
exactly the built-in default (`cap_mbps = 12`, raw total ~10.5 Mbps), so it
reproduces the historical NS-3 scenario byte-for-byte, while `configs/dynamic.yaml`
drives a 6-path set (§5.3.1). The repo's `configs/default.yaml` retains the legacy
3-path topology for quick mock-only experiments; it now also runs as 3 paths on
NS-3 (previously the body ignored the YAML and always built its 4-path default).

A subflow is thus a stock single-path NS-3 TCP connection (`TcpSocketFactory`,
the build's default congestion control). The "multipath QUIC" property lives one
layer up, in the controller, which splits each video frame's bytes across the `N`
subflows. The mapping to true MPQUIC is: *subflow ↔ QUIC path; the controller's
per-frame byte split ↔ the QUIC packet scheduler's path choice; TCP's
loss-recovery ↔ QUIC's per-path reliability.* See §11 for what this faithfully
captures and what it does not.

### 3.2 Background (cross) traffic

Each bottleneck is shared with a competing, time-varying UDP flow generated by an
NS-3 `OnOffApplication` (`BuildCrossTraffic`). The mean rate is a configured
fraction of the link rate (`cross_frac`, 0.40/0.45/0.30/0.55 across the four
paths), the packet size is 1200 B, and the on/off sojourn times are
**exponentially distributed**
with per-path means that are deliberately *phase-shifted* (on-mean `0.6 + 0.2·i`
s, off-mean `0.8 + 0.3·i` s). The effect is that each path's *available* capacity
fluctuates burstily and the paths' good/bad periods are decorrelated, so the
Transport agent faces a genuinely non-stationary, path-heterogeneous scheduling
problem rather than a static capacity assignment.

### 3.3 The real-time video source model

Real-time conferencing differs fundamentally from adaptive streaming (DASH): the
encoder emits a frame on a **fixed wall-clock cadence** regardless of whether the
previous frame finished delivering. There is no playback buffer to hide
under-delivery; instead, late data manifests directly as latency and, past a
deadline, as a discarded frame. This is the dynamic the controller must manage,
and it is the source of bufferbloat-style failure modes (if the sender overdrives
a path, its socket backlog grows and *every subsequent frame on that path
inherits the accumulated delay*).

`RealtimeController` generates one frame every `1/fps` seconds of simulated time.
The frame size in bytes is

```
base   = (bitrate_kbps · 1000 / 8) / fps          # bytes per frame at the target rate
kf     = 2.5  if (frame_index mod keyframe_interval == 0) else 1.0   # I-frame burst
jitter = 1 + frame_size_jitter · U(-1, 1)          # P-frame content variability
frame_bytes = max(1, round(base · kf · jitter))
```

with defaults `fps = 30`, `keyframe_interval = 30` (one I-frame per second),
`frame_size_jitter = 0.25`. The I-frame multiplier and per-frame jitter reproduce
the bursty, variable-bitrate envelope of a real encoder, which matters because it
is precisely the bursts (large I-frames) that stress the scheduler. This exact
model is mirrored in Python (`src/ns3env/video_source.py::frame_bytes`) so the
mock and NS-3 backends generate statistically comparable frames.

### 3.4 Per-frame striping and the byte-watermark delivery tracker

When the controller applies an action (`GenerateFrame`), it (a) clamps and stores
the target bitrate; (b) normalizes the requested split (clamping negatives,
renormalizing to sum 1, falling back to uniform if degenerate); (c) computes
`frame_bytes`; and (d) splits those bytes across paths by the split ratio, with a
rounding-reconciliation step that pushes any residual onto the largest share so
the per-path shares sum *exactly* to `frame_bytes`.

The hard part is measuring **per-frame completion latency** over reliable,
in-order TCP byte streams that carry many pipelined frames. The controller solves
this with a **cumulative byte-watermark** scheme that needs no per-packet tagging:

- For each path `i` it maintains a monotonically increasing count
  `m_pathEnq[i]` of bytes ever handed to that subflow.
- When a frame contributes `b` bytes to path `i`, the controller advances
  `m_pathEnq[i] += b` and enqueues a **share record** `{watermark = m_pathEnq[i],
  frameId, bytes = b}` into a per-path FIFO. The share is "delivered" exactly when
  the sink's *cumulative received bytes* on path `i` reach `watermark` — because
  TCP delivers reliably and in order, cumulative-received monotonically chases
  cumulative-enqueued.
- The `RealtimeSink` reports its cumulative received byte count (and the current
  time) on every read. `OnPathBytes` pops all share records whose watermark has
  been reached, stamping each with its completion time.

A frame is **complete** when all of its non-empty shares (possibly across several
paths) have been delivered. The controller tracks, per in-flight frame, the
number of outstanding shares and the running max of share-completion times; when
the count hits zero, `CompleteFrame` fires.

### 3.5 Frame outcome: latency, jitter, and deadline loss

On completion of frame `f`:

```
latency_ms = (max share-completion time − frame generation time) · 1000
late       = latency_ms > deadline_ms                     # default deadline 180 ms
jitter_ms  = |latency_ms − previous frame's latency_ms|    # inter-frame delay variation
```

The controller maintains episode-scoped exponential moving averages (EWMAs) that
feed the *aggregate* observation, with the following gains:

| Quantity | Update | Gain |
|---|---|---|
| jitter EWMA | `0.7·old + 0.3·jitter` | 0.30 |
| app-loss EWMA | `0.9·old + 0.1·[late]` | 0.10 |
| aggregate throughput EWMA | `0.7·old + 0.3·goodput` | 0.30 |
| aggregate RTT EWMA | `0.8·old + 0.2·latency` | 0.20 |
| per-path goodput EWMA | `0.6·old + 0.4·share_goodput` | 0.40 |

where `goodput = frame_bytes·8 / (latency·1e-3) / 1e6` Mbps. Frames that exceed
the deadline before fully arriving are *expired* as losses by `ExpireLateFrames`
(swept once per frame): they bump the app-loss EWMA, are reported with
`lastLoss = 1` and `lastBytes = 0`, and their straggler shares are skipped when
they later arrive (the frame record is gone, so the share is a no-op). Thus, in
this system, **"loss" at the application layer means a deadline miss** — the
real-time-conferencing notion — distinct from network packet loss (§3.6), which
TCP would otherwise hide behind retransmission as added latency.

### 3.6 Transport-state measurement

Per path, the body exposes the transport-layer signals a real multipath sender
could read from its sockets:

- **Congestion window (`cwnd`, bytes):** read from each subflow's
  `TcpSocketBase` `"CongestionWindow"` trace.
- **Smoothed RTT (`srtt`, ms):** an EWMA (`0.85·old + 0.15·sample`) of the
  socket's `"RTT"` trace.
- **Send-buffer backlog (`bufferOcc`, bytes):** the application-level bytes the
  sender has queued but not yet handed to TCP — the most direct early-warning
  signal of overdrive on a path.
- **Per-path delivered goodput (Mbps):** the EWMA above.
- **Per-path network loss in [0, 1]:** computed from NS-3 `FlowMonitor` as
  `lost / (tx + lost)` over the path's TCP flow. Because a full `FlowMonitor`
  sweep is comparatively expensive and the bridge runs every 33 ms, this sweep is
  **throttled to the 1 s app cadence** and cached between sweeps — a deliberate
  cost/fidelity trade documented in the code.

### 3.7 The decision loop and episode lifecycle

The control flow per frame (`RealtimeController::Decide`) is:

1. `ExpireLateFrames(now)` — retire any frames past their deadline.
2. If this is an app boundary, refresh the cached `FlowMonitor` per-path loss.
3. `FillObservation(env)` — populate `EnvStruct` (aggregate + per-path + last
   frame), and set `done` if the episode's frame budget is exhausted.
4. **Send** `EnvStruct` to Python; **receive** `ActStruct` (or, in `--selftest`,
   synthesize an even-split fixed-bitrate action locally with no bridge).
5. Dispatch on the action command:
   - `ACT_TERMINATE` → stop the simulation (process exit).
   - `ACT_RESET` *or* `done` → reset episode-scoped counters and EWMAs (keeping
     the network warm — sockets, queues, and congestion state persist across
     episodes) and re-enter `Decide`.
   - `ACT_STEP` → `GenerateFrame(act)`, increment the frame counter, and schedule
     the next `Decide` at `now + 1/fps`.

Crucially, the simulator **advances during frame delivery, not during the bridge
exchange**: the `CppRecv` is a blocking semaphore wait in which sim time is
frozen; sim time only progresses between `Decide` calls, while the striped bytes
traverse the paths. One long-lived NS-3 process serves an entire training run;
episode boundaries are signalled **in-band** (`ACT_RESET`) rather than by
relaunching the process, because ns3-ai permits only one shared-memory creator
per Python process (§4).

Episode length is `round(fps · episode_seconds)` frames (e.g. 900 frames for a
30 s episode at 30 fps). A warm-up delay (default 1 s) precedes the first
decision so the TCP connections establish and RTT estimates become live.

### 3.8 The contract: `EnvStruct` and `ActStruct`

These two POD structs (`ns3/realtime_mpquic.h`) are the entire wire format.
Fixed-size arrays (`kMaxPaths = 8`) keep them trivially copyable for the
struct-based shared-memory interface.

**`EnvStruct` (C++ → Python), what the application "sees":**

| Field | Meaning |
|---|---|
| `numPaths`, `clockS`, `done`, `appDecisionDue` | path count, sim time, terminal flag, app-boundary flag |
| `currentBitrateKbps` | bitrate currently in effect |
| `rttMs`, `jitterMs`, `loss`, `throughputMbps` | aggregate app-level state (split-weighted RTT; jitter, deadline-loss and goodput EWMAs) |
| `cwnd[]`, `srttMs[]`, `bufferOcc[]`, `pathThroughputMbps[]`, `pathLoss[]` | per-path transport state (length `numPaths`) |
| `pathActive[]` | per-path liveness mask (length `numPaths`): `1` = live, `0` = churned out. All-ones when dynamics are off (§3.9) |
| `lastLatencyMs`, `lastJitterMs`, `lastLoss`, `lastBytes` | realized result of the most-recently-completed frame |

**`ActStruct` (Python → C++), what the agents control:**

| Field | Meaning |
|---|---|
| `command` | `ACT_STEP` / `ACT_RESET` / `ACT_TERMINATE` |
| `targetBitrateKbps` | App action — encoder target, **persists** across frames |
| `splitRatio[]` | Transport action — per-path fractions for the next frame |

### 3.9 Non-stationary dynamics in the body (optional)

The body now mirrors the mock's four non-stationary mechanisms (§5.3.1), so
`--backend ns3` can exercise the same variable-active-path scenario the mock does
and a policy sees one observation contract on either backend. Everything is **off
by default** (`dynamicsEnabled = 0`): when disabled the dynamics RNG is never
drawn and per-path link rates stay at their nominal values, so the static NS-3
scenario is byte-identical to before and `pathActive[]` is reported all-ones.

The parameters are the same `DynamicsConfig` the mock consumes, forwarded from
Python into the C++ CLI (§5.2) and parsed into `ScenarioConfig`. `RealtimeController`
runs the four state machines once per frame (event probability `1 − e^{−rate·Δt}`,
same draw order as the mock — regime, burst, correlation, churn), from a
**dedicated NS-3 RNG stream** seeded by the scenario `seed` so the frame-size
jitter stream is unperturbed and runs stay deterministic per seed:

- **Churn.** A per-path on/off Markov chain (floored at `minActive`). A churned-out
  path collapses its NS-3 link (`DataRate → ~0`) *and* has any bytes the scheduler
  routes onto it dropped at the app layer and counted as loss — the same penalty
  the mock applies (`CompleteFrame` folds the dropped fraction into the frame's
  loss; a whole-frame-on-dead-paths is an immediate deadline miss). It reports a
  dead per-path row and `pathActive = 0`.
- **Regime / burst / correlated failures.** These scale the per-path bottleneck
  `DataRate` by the same multiplier logic as the mock's `_cap_mult`
  (`regime_mult × burst_mult × corr_mult`), reapplied to the live NS-3 devices each
  frame, so per-path throughput/RTT reflect the shift/collapse.

Episodes reset the dynamic state in-band (all paths live again, regime multipliers
resampled) alongside the existing counter/EWMA reset (§3.7).

**Where the body only approximates the mock.** (i) The two backends draw from
independent RNG streams, so they are *behaviorally* equivalent but **not**
frame-identical. (ii) Per-path throughput/RTT evolve through NS-3's real transport
(TCP over the modulated bottleneck) rather than the mock's analytical standing
queue, so aggregate magnitudes differ even when the dynamics match. (iii)
Correlated-failure group indices `≥` the topology's path count are dropped — a
non-issue now that the topology is forwarded (§3.1), so `configs/dynamic.yaml`'s
`corr_groups: [[4, 5]]` is in-range on the 6-path NS-3 set.

---

## 4. The shared-memory bridge (ns3-ai)

### 4.1 Synchronous lockstep protocol

The bridge is the `ns3-ai` **struct-based message interface**: two named
shared-memory regions (C++→Python and Python→C++) guarded by interprocess
semaphores. The protocol is strict lockstep and **C++ leads with a send** every
decision:

```
   C++ (body)                          Python (brain)
   ──────────                          ──────────────
   FillObservation(env)
   CppSendBegin/End(env)  ───────────► PyRecvBegin/End()  → snapshot EnvStruct
   CppRecvBegin()  ◄─────────────────  PySendBegin/End(act)  ← write ActStruct
   apply action; run 1/fps of sim
   (repeat)
```

Because the channel is semaphore-ordered, the two sides must alternate exactly: if
one side begins a receive, the other must begin the matching send. The Python data
plane tracks a single bit of protocol state (`_owe_send`, "C++ is currently
blocked waiting for our action") so it can always issue a clean `ACT_TERMINATE`
on shutdown.

### 4.2 Struct marshalling (pybind11)

`ns3/realtime_mpquic_py.cc` is a pybind11 module exposing the two structs and the
interface object. `EnvStruct` scalars are exposed read-only as attributes; its
per-path C arrays cannot be `def_readwrite` directly, so they are exposed through
index getters (`e.cwnd(i)`, `e.srtt(i)`, …). `ActStruct` exposes its scalars plus
`setSplit(i, v)` / `getSplit(i)`. The Python side reads/writes the shared struct
*in place* between the `Begin`/`End` calls — no serialization — and immediately
snapshots the fields it needs into a plain Python dataclass (`FrameObs`) because
the shared region is only valid inside the critical section.

### 4.3 Process lifecycle and in-band episode control

`Ns3DataPlane` launches the NS-3 binary through `ns3ai_utils.Experiment`, which
creates the shared memory (Python is the creator), spawns `./ns3 run …` with the
episode parameters as CLI arguments, and returns the interface handle. Because
only one shared-memory creator may exist per process, the design uses **one
long-lived process** and resets episodes in-band. Teardown sends `ACT_TERMINATE`
(if owed), kills the subprocess tree, and garbage-collects the interface to
release the shared region.

---

## 5. The Python data-plane abstraction

`src/ns3env/dataplane.py` defines the contract that makes the two backends
interchangeable.

### 5.1 The `DataPlane` ABC and the shared types

```python
class DataPlane(ABC):
    num_paths: int
    cap_mbps: float                                  # normalization cap (aggregate)
    def reset(seed) -> FrameObs
    def current_obs() -> FrameObs
    def step_frame(target_bitrate_kbps, split_ratio) -> FrameResult
    def is_done() -> bool
    @property clock_s -> float
    def app_decision_due() -> bool
    def close()
```

`FrameObs` is a one-to-one Python mirror of `EnvStruct`; `FrameResult` carries the
realized `(latency_ms, jitter_ms, loss, bytes_delivered)` of a single frame. The
protocol is identical to the bridge's: `reset` returns the initial observation,
and each `step_frame` *sends the action and returns the next observation's
realized result* (its `last_*` fields), with the next observation available via
`current_obs()`.

### 5.2 `Ns3DataPlane`

A thin marshaller over §4: it imports the pybind `.so`, launches the process,
forwards episode parameters (fps, episode length, app period, deadline, bitrate
bounds, seed), the `topology:` path list, and — when enabled — the `dynamics:`
parameters (§3.9) as CLI flags, and translates between `FrameObs`/`FrameResult`
and the shared structs (including reading the per-path liveness mask from
`EnvStruct.pathActive[]` into `FrameObs.path_active`). `reset` either launches the
process (first call) or issues an in-band `ACT_RESET`. `step_frame` normalizes the
split, writes `ACT_STEP + targetBitrate + split`, receives the next `EnvStruct`,
and returns its `last_*` fields as a `FrameResult`.

### 5.3 `MockRealtimeDataPlane`

A pure-Python, deterministic-per-seed reimplementation of the body's *observable
behavior*, so the entire brain (env, reward, agents, training loop) can run and be
unit-tested with no NS-3 dependency. It is not a packet simulator; it is a
**fluid/queueing surrogate** calibrated to the same topology.

Each path has a time-varying capacity (`_PathTrace.capacity_mbps`):

```
season = 1 + amp · sin(2π t / period + phase)
cross  = cross_frac · (0.5 + 0.5 · sin(2π t / cross_period + phase))
cap    = max(0.05, base_mbps · season · max(0.05, 1 − cross) · (1 + N(0, noise_std)))
```

i.e. a sinusoidal capacity envelope, a phase-shifted bursty cross-traffic
reduction, and multiplicative noise — the fluid analogue of §3.1–3.2.

#### 5.3.1 Non-stationary dynamics and the liveness mask (optional)

The sinusoidal envelope above is smooth and stationary: the *best* path barely
moves, so the optimal split is nearly constant and a learned scheduler can win
almost nothing over a fixed/heuristic split (see §11 — on the static topology the
App-only ablation reaches ~98 % of the full system's QoE). To make *which path to
send on* a genuinely informative, time-varying decision, the mock supports an
optional `DynamicsConfig` (`configs/dynamic.yaml`); the NS-3 body now mirrors the
same four mechanisms (§3.9). It is **off by default**: when
disabled the dynamic RNG is never drawn, so the static behavior — and its exact
per-seed draw sequence — is unchanged. Four mechanisms, all deterministic per
seed and advanced one step per frame (event probability `1 − e^{−rate·Δt}` over a
frame interval):

- **Path churn (appear/disappear).** Each path is an on/off Markov chain
  (`churn_up_rate` / `churn_down_rate`, floored at `min_active` live paths). The
  candidate count `num_paths` is now an *upper bound*; the **active** count varies
  per frame. A churned-out path reports a dead row and `path_active = 0`; bytes a
  scheduler routes onto it **never arrive and count as loss** — the penalty that
  teaches the policy to respect the mask.
- **Regime shifts (best-path swaps).** A per-path capacity multiplier is resampled
  `~ Uniform(regime_lo, regime_hi)` at Poisson change-points (`regime_rate`),
  multiplying the envelope. Because paths shift independently, the *ranking* of
  paths jumps abruptly rather than drifting.
- **Congestion bursts.** Poisson per-path events (`burst_rate`) collapse a path's
  capacity to `burst_intensity` for `burst_duration_s`, which the standing-queue
  model turns into a latency spike the scheduler must route around within a few
  frames.
- **Correlated failures.** `corr_groups` of path indices degrade together during
  group events (`corr_rate`, `corr_intensity`, `corr_duration_s`), so naive
  diversification across a shared bottleneck does not help.

The effective per-path capacity is `envelope × regime_mult × burst_mult ×
corr_mult` (all 1.0 when static). The `path_active` field on `FrameObs` mirrors
`EnvStruct.pathActive[]`, which the NS-3 body now emits directly (§3.9; all-ones
when dynamics are off). Delivery folds the fraction of bytes routed onto dead
paths into the frame's loss, so misrouting is graded, not silent.

Delivery uses a **standing-queue model** that reproduces bufferbloat. Each path
keeps a `busy_until` serialization clock; delivering `b` bytes on path `i`:

```
ser_s   = (b · 8) / (cap · 1e6)                    # serialization time at current capacity
ready   = max(now, busy_until[i])                  # queue ahead of us if path is busy
finish  = ready + ser_s ;  busy_until[i] = finish
arrival = finish + (base_rtt_ms/1000)/2            # + one-way propagation
```

The frame's latency is `max` over its shares of `(arrival − now)`, and
`late = latency > deadline`. The clock then advances by exactly `1/fps`,
**independent of how long delivery took** — the real-time cadence. Overdriving a
path makes `busy_until` run ahead of `now`, so each subsequent frame on that path
inherits a growing standing queue: latency diverges exactly as it does in the
packet simulator. The per-path observation fields are synthesized consistently:
`srtt = base_rtt + standing-queue delay`, `bufferOcc = queue · cap / 8`,
`cwnd ≈ bandwidth-delay product`, plus the same EWMA gains as the C++ body. A
test (`test_overdriving_one_path_builds_queue`) asserts this divergence, and
another (`test_determinism_same_seed`) asserts reproducibility.

---

## 6. The hierarchical environment: observations and rewards

`src/ns3env/realtime_env.py` (`HierarchicalRealtimeEnv`) is a *pure* function of
the data plane — no torch, no learning. It (a) builds the two agents' observation
vectors, (b) computes the per-frame transport reward, and (c) accumulates the
per-window App reward. The path count is fixed within a run, so observations are
fixed-length numpy vectors normalized to roughly `[0, 1]`.

### 6.1 Observation construction

**App-agent observation** — `build_app_obs`, dimension **5**:

| Index | Feature | Normalization |
|---|---|---|
| 0 | current bitrate | `(kbps − min)/(max − min)` |
| 1 | aggregate RTT | `rtt_ms / latency_norm_ms` (200) |
| 2 | aggregate jitter | `jitter_ms / jitter_norm_ms` (50) |
| 3 | aggregate loss | `loss` (already in [0,1]) |
| 4 | aggregate throughput | `throughput_mbps / cap_mbps` |

The App agent thus sees a compact, *aggregate* summary: how good its current
quality choice is and what it is costing in delay/jitter/loss. It deliberately
does **not** see per-path detail — that is the Transport agent's concern — and,
just as deliberately, it does **not** see episode progress / time-to-go. A real
video call has unknown, unbounded duration, so a deployable policy must be
**horizon-agnostic**; conditioning on "how far through the episode we are" would
both leak information the sender cannot have at deployment and invite
end-of-episode-specific behavior (e.g. spending latency budget at the very end
where there is no discounted future to pay for it). The fixed training horizon is
instead handled as a *time-limit truncation* rather than a terminal state (§8).

**Transport-agent observation** — `build_transport_obs`, dimension **4 + 5·N**
(= 24 for the canonical N = 4 NS-3 topology; 19 for the legacy 3-path config):

*Global block (4):* `[ bitrate_norm, rtt_ms/latency_norm, loss, throughput_mbps/cap ]`.
The first element is the **App agent's current target bitrate** — this is the
hierarchical coupling: the Transport agent is conditioned on how much load it must
schedule.

*Per-path block (5 features × N paths):* for each path `i`,
`[ cwnd/200000, srtt/latency_norm, bufferOcc/200000, path_throughput/cap, path_loss ]`.

All features are clipped to `[0, 1]` (RTT/jitter clipping saturates at the
normalizer; nothing is unbounded). The Transport agent therefore sees the full
per-path transport picture (window, smoothed delay, backlog, recent goodput, loss)
plus the global demand it must satisfy.

**Structured (scoring) transport state** — `build_transport_state`. The flat
`4 + 5N` vector hard-codes the path count and order, so it cannot represent a
*changing* set of paths. For the scoring Transport agent (§7.3) the same features
are instead delivered as a `(glob, paths, mask)` triple — a 4-D global-context
vector, a `(N, 6)` per-path matrix (the five flat per-path features **plus the
liveness flag**), and an `(N,)` mask (`1.0` = live). The path count `N` is fixed
at the candidate cap for tensor shapes; the *mask*, not the array width, says
which rows are live, so the policy and critic (which are permutation-equivariant)
handle any active subset. The flat builder is retained unchanged for the legacy
`transport_arch: "flat"` agent.

### 6.2 Two-timescale credit assignment

The environment maintains a **reward window** — running lists of the per-frame
`latency`, `jitter`, and `loss` for the frames governed by the *current* bitrate,
plus that bitrate. Every `step_frame` appends to the window. When the training
loop hits an app boundary, it calls `pop_app_window_reward()`, which aggregates
the window (mean latency, mean jitter, mean loss), scores it with the QoE
functional, **clears the window**, and returns the scalar reward plus unweighted
components for logging. That reward is credited to the *previous* App action — the
bitrate that actually governed those frames — implementing a correct
delayed/temporally-extended reward for the slow agent. The Transport agent, by
contrast, is rewarded immediately each frame.

### 6.3 Reward functionals

Both rewards live in `src/ns3env/qoe.py`.

**Perceptual quality (VMAF).** Quality is measured in VMAF (0–100), not raw
bitrate, because the rate→quality map is concave (equal bitrate steps are not
equal perceptual steps). We fit a logarithmic curve through two Netflix-style
anchors `(300 kbps, 25)` and `(4300 kbps, 92)`:

```
VMAF(R) = a · ln(R) + b ,   clipped to [0, 100],
   with a, b solving the two anchors.
```

*Pluggable quality model (learned VMAF).* The VMAF term is a swappable hook
(`compute_qoe_reward(..., vmaf_fn=…)`). Passing `--learned-vmaf` (or
`reward.use_learned_vmaf: true` in the config) replaces the bitrate-only log curve
with a **WebRTC-grounded learned surrogate** vendored from the
`WebRTC-QoE-Data-Generator` sibling project (`src/ns3env/qos_vmaf_reward.py` +
`reward_model.npz`, wrapped by `src/ns3env/learned_vmaf.py`). That model is a
multilinear interpolant over *real* WebRTC VMAF measurements mapping
`(bitrate_kbps, loss_pct, delay_ms, jitter_ms) → VMAF`, clamped to its measured
grid box.

Because the learned VMAF **already folds loss into the quality score**, the
explicit `− d·loss` penalty is **dropped** under the learned reward to avoid
double-counting; the latency/jitter penalties are **kept** (the current model does
not separate those — see the grid caveat below). Concretely, the App reward is
selected by which quality model is active:

```
default (log curve):  QoE = a·VMAF(bitrate)/100 − b·lat − c·jit − d·loss
learned surrogate:    QoE = a·VMAF(bitrate, loss, delay, jitter)/100 − b·lat − c·jit
                            (no explicit − d·loss term; loss enters via VMAF)
```

with `lat`, `jit` the same normalized, soft-capped penalties in both cases, and
the result clipped to `[−2, 1]`. The unit translation at the boundary
(`learned_vmaf.py`) is: loss fraction → percent (`×100`), one-way frame
`latency_ms` → the model's one-way `delay_ms` (no `/2`), jitter passes through.

Two caveats follow from how the shipped grid was fitted: (i) its bitrate axis
saturates near ~2500 kbps, so the App agent gets no quality incentive to push
above that under the learned reward (a reasonable conferencing ceiling); and
(ii) the model's delay/jitter axes are currently **degenerate** (a single measured
point), so the shipped surrogate varies over bitrate and loss only — which is
exactly why the latency/jitter penalties must stay explicit. A richer refit
activates the delay/jitter axes with no code change (the adapter already passes
them). The default (log curve) path is byte-for-byte unchanged.

**App QoE reward** (`compute_qoe_reward`, default log-curve form),
credited per window:

```
QoE = a · VMAF(bitrate)/100
        − b · clip(latency/latency_norm, 0, 2)
        − c · clip(jitter/jitter_norm, 0, 2)
        − d · clip(loss, 0, 1) ,            clipped to [−2, 1]
```

with default weights `a = 1.0, b = 0.5, c = 0.5, d = 1.0` and normalizers
`latency_norm = 200 ms`, `jitter_norm = 50 ms`. This is the literal realization of
the project's target functional `R = a·Bitrate − b·Latency − c·Jitter − d·Loss`,
with "Bitrate" replaced by its perceptual proxy VMAF.

**Transport reward** (`compute_transport_reward`), per frame:

```
R_t = (1 − loss) − b · clip(latency/latency_norm, 0, 2)
                 − c · clip(jitter/jitter_norm, 0, 2) ,   clipped to [−2, 1]
```

The quality term is intentionally **excluded** — the Transport agent does not
choose the bitrate, so rewarding it for quality would be miscredited. It is
rewarded purely for *delivering* whatever frame it is given quickly and intact.
This reward decomposition is what gives each agent a clean, well-attributed
gradient.

---

## 7. The reinforcement-learning model

### 7.1 Why SAC

Both agents have **continuous** action spaces (a scalar bitrate; an `N`-vector
split), and environment steps are **expensive** (a packet-level NS-3 frame).
Soft Actor-Critic is the natural fit: it is **off-policy** (so a replay buffer
amortizes each costly transition over many gradient updates) and it maximizes a
**maximum-entropy** objective (reward plus policy entropy), which yields robust
exploration and stable learning in continuous control. The generic algorithm is
implemented once in `src/rl/sac_agent.py` and reused by both agents through thin
wrappers; it operates entirely in a **normalized action space** `[−1, 1]^d`, and
each wrapper maps that to its physical action.

### 7.2 Network architectures

All networks are multilayer perceptrons with ReLU activations and a default hidden
width of 256.

**Actor — `GaussianPolicy` (tanh-squashed diagonal Gaussian):**

```
obs (d_obs) ─► Linear(d_obs, 256) ─ ReLU ─ Linear(256, 256) ─ ReLU ─┬─ Linear(256, d_act)  → mean
                                                                     └─ Linear(256, d_act)  → log_std (clamped [−20, 2])
sample:  x ~ Normal(mean, exp(log_std));   a = tanh(x)            # a ∈ (−1, 1)^d_act
log π(a|s) = Σ_j [ log Normal(x_j) − log(1 − a_j² + ε) ]          # tanh change-of-variables
```

The `tanh` squashing bounds actions to `[−1, 1]` and the log-det correction keeps
the entropy/log-probability exact. In deterministic mode (evaluation) the actor
returns `tanh(mean)`.

**Critics — `QNetwork` (twin Q, clipped-double-Q):** two independent MLPs
`Q1, Q2 : (obs ⊕ action) → ℝ`, each `Linear(d_obs+d_act, 256) → ReLU →
Linear(256,256) → ReLU → Linear(256,1)`. A **target** copy of the twin critics is
held for bootstrapping and tracked by Polyak averaging.

### 7.3 Action spaces and the agent wrappers

**App agent** (`src/rl/app_agent.py`): `d_act = 1`. The normalized action
`a ∈ [−1,1]` maps affinely to a bitrate,
`kbps = min_kbps + (a+1)/2 · (max_kbps − min_kbps)` (defaults 300–6000 kbps). The
agent exposes `select(obs) → (kbps, raw_action)`; the *raw* `[−1,1]` action is
what is stored in the replay buffer and what the critic scores.

**Transport agent** (`src/rl/transport_agent.py`): `d_act = N`. The normalized
action vector is turned into a valid split by a temperature-scaled **softmax**,
`split = softmax(a / τ)` (τ = 1 by default), guaranteeing non-negativity and
sum-to-one. The agent's observation includes the App bitrate (§6.1), realizing the
hierarchy. As with the App agent, the stored action is the pre-softmax `[−1,1]`
vector, so the critic operates in the same normalized space the actor samples in.

This "store normalized, map to physical at the boundary" convention is important:
it keeps the SAC math (entropy, target entropy, log-probabilities) entirely inside
the well-conditioned `[−1,1]^d` cube while the environment receives physically
meaningful actions.

#### 7.3.1 Dynamic-input (scoring) Transport agent

The flat agent above is an MLP whose input width is `4 + 5N`, hard-wiring the
path count and order — it cannot ingest a *changing* set of paths (churn) and must
relearn per ordering. A second, permutation-equivariant Transport agent
(`src/rl/scoring_sac_agent.py`, selected by `transport_arch: "scoring"`) removes
that limitation, adapting the SCION sibling's path-scoring DQN from discrete
argmax to continuous SAC. It consumes the structured `(glob, paths, mask)` state
(§6.1):

- **Actor — `ScoringGaussianPolicy`.** A *shared* per-path encoder maps
  `glob ⊕ path_i` to a tanh-squashed Gaussian latent per path. The latent is the
  SAC action in `[−1, 1]^N` (exactly the flat agent's raw action); the env split is
  a **masked softmax** of it over the *active* paths only. Log-probability and
  entropy are summed over active paths, so dead paths contribute neither density
  nor gradient.
- **Critic — `ScoringQNetwork` (twin).** A DeepSets-style encoder consumes
  `glob ⊕ path_i ⊕ latent_i` per path, **masked-mean-pools** across paths, and maps
  the pooled embedding to a scalar Q — permutation-invariant and variable-N.
- **Entropy target** scales with the per-sample active-path count
  (`−active_count`), not a fixed `−N`, since the effective action dimension shrinks
  as paths churn out.

Transitions are stored in a `StructuredReplayBuffer` (rectangular fixed-`N`
arrays + mask, simpler than ragged padding because `N` is the candidate cap). The
critic, actor, twin-target, and automatic-temperature update are otherwise
identical to §7.4. Inactive paths are excluded from the log-prob, the pooled
critic embedding, and the split, so they receive no gradient. Checkpoints are
tagged `arch: "scoring"`, which evaluation reads back to rebuild the right agent
(legacy flat checkpoints have no such tag and load as `"flat"`).

### 7.4 The learning update

One gradient step (`SACAgent._update_once`) on a minibatch (default 256) sampled
from the replay buffer:

**Critic.** With the entropy-regularized bootstrap target

```
a', log π(a'|s') ← actor(s')
y = r + γ (1 − done) [ min(Q1_targ(s',a'), Q2_targ(s',a')) − α · log π(a'|s') ]
L_critic = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
```

(γ = 0.99). The `min` of the twin targets is the clipped-double-Q trick that
counteracts value overestimation; the `−α log π` term is the entropy bonus folded
into the value.

**Actor.** Reparameterized policy-gradient that maximizes entropy-augmented Q:

```
ã, log π(ã|s) ← actor(s)            # reparameterized sample
L_actor = E[ α · log π(ã|s) − min(Q1(s,ã), Q2(s,ã)) ]
```

**Temperature α (automatic entropy tuning).** With target entropy `H̄ = −d_act`,

```
L_α = − E[ log α · ( log π(ã|s) + H̄ ) ]
```

so α is driven to hold the policy's entropy near `H̄`; the App agent (`d_act = 1`)
targets `−1` and the Transport agent (`d_act = N`) targets `−N`. α is parameterized
as `exp(log_alpha)` for positivity.

**Targets.** Polyak update `θ_targ ← (1−τ)θ_targ + τθ` with τ = 0.005 after every
critic step. Default optimizer is Adam at lr `3e-4` for all three losses.

### 7.5 Replay, warm-up, and exploration

The `ReplayBuffer` is a pre-allocated circular buffer of
`(obs, action, reward, next_obs, done)` (capacity 200 000 by default), storing
actions in the normalized space. To seed exploration, `select_action` returns
**uniform random** actions in `[−1, 1]^d` until `start_steps` transitions have
been stored, after which it samples the policy. Gradient updates begin only once
the buffer holds at least `max(batch_size, update_after)` transitions
(`ready()`), and `updates_per_step` steps are taken per environment frame.
Because the App agent steps 30× less often than the Transport agent, its warm-up
thresholds are automatically scaled down by the frames-per-app factor
(`_app_sac_config`) so it still begins learning within a run.

### 7.6 The hierarchy in practice

The two agents never share weights, buffers, or gradients. They are coupled only
through (i) the **observation channel** — the Transport agent's input contains the
App agent's current bitrate — and (ii) the **environment dynamics** — the App
agent's bitrate choice changes the load the Transport agent must schedule, and the
Transport agent's scheduling quality changes the latency/loss the App agent is
rewarded/penalized for. This is a *cooperative, communication-through-observation*
hierarchy: the slow agent sets the operating point, the fast agent realizes it,
and each sees enough of the other's effect to adapt. Empirically, on the 4-path
NS-3 topology (40-episode train, eval seed 1000) the learned pair reaches
**QoE 0.633** (VMAF 84, ~3.15 Mbps *aggregated* across all four subflows) versus
**0.514** for single-best-path and **0.430** for throughput-proportional
splitting — a +23 % QoE gain it earns specifically by aggregating capacity the
heuristics cannot, while down-weighting the latency-trap path. The App agent
learns to push bitrate up for perceptual quality while the Transport agent keeps
latency under the deadline by spreading load — the two-sided behavior the reward
decomposition was designed to induce.

On the **non-stationary** scenario (`configs/dynamic.yaml`, §5.3.1) with the
scoring Transport agent (§7.3.1), the hierarchy's value becomes stark. A 30-episode
mock train, 20-episode eval (seed 1000): the learned pair reaches **QoE 0.606**
(VMAF 85 at ~3.3 Mbps, latency 60 ms, loss 0.05), versus **0.402**
single-best-active-path, **0.209** proportional, and **0.109** even. Critically,
the `app_only` ablation — the learned bitrate with a (mask-aware) *even* split —
**collapses to −0.212** (latency 453 ms, loss 0.42): once paths churn, shift, and
burst, an even split shoves the encoder's ~2.8 Mbps onto congested or dead paths.
This is the intended contrast with the *static* topology, where `app_only` reached
**98 %** of the full system's QoE (0.665 vs 0.676) because the optimal split
barely moved. Making the network non-stationary — and giving the Transport agent a
model that can ingest the changing path set — is what turns the split back into a
decision that matters.

---

## 8. The training loop

`src/train/hierarchical_train.py::run_training` orchestrates everything. After
building the data plane, the env, and the two agents (sizing the Transport agent
from `env.num_paths`), each episode (`_run_episode`) runs the dual cadence:

```
obs ← env.reset(seed)
target_kbps ← obs.current_bitrate
prev_app_obs, prev_app_raw ← None, None
while not done:
    if obs.app_decision_due:                         # ── slow (App) cadence ──
        app_obs ← build_app_obs(obs)
        if prev_app_obs is not None:                 # credit the window that just closed
            r_app, comps ← env.pop_app_window_reward()
            app.store(prev_app_obs, prev_app_raw, r_app, app_obs, done=False)
            app.update()
        target_kbps, prev_app_raw ← app.select(app_obs)
        prev_app_obs ← app_obs

    t_obs ← build_transport_obs(obs, target_kbps)    # ── fast (Transport) cadence ──
    split, t_raw ← transport.select(t_obs)
    next_obs, r_t, done, info ← env.step(target_kbps, split)
    t_next ← build_transport_obs(next_obs, target_kbps)
    transport.store(t_obs, t_raw, r_t, t_next, done=False)   # truncation, not terminal
    transport.update()
    obs ← next_obs

# episode end: credit the final (partial) App window, also with done=False
```

Key correctness points: the App transition's *next state* is the app-observation
at the **next** boundary (a genuine temporally-extended SARSA-style tuple over the
window), and its reward is the QoE of the window the *previous* action governed.
The Transport transition is an ordinary per-frame tuple. **Time-limit handling:**
the episode horizon is a *truncation*, not a real terminal — the call could
continue past it — so every transition (including the final partial App window) is
stored with `done = False`, bootstrapping the value off the next state rather than
cutting the return. This keeps both policies horizon-agnostic (consistent with the
App agent not observing episode progress, §6.1) and is the standard finite-horizon
correction (Pardo et al., *Time Limits in RL*, 2018). The loop logs per-episode
mean QoE,
mean transport reward, mean bitrate/latency/loss/VMAF, and at the end writes
`app.pth`, `transport.pth`, and a `stats.json` history to the run directory. A
`finally` block guarantees `dp.close()` (hence `ACT_TERMINATE` and shared-memory
release) even on exceptions.

---

## 9. Evaluation and baselines

`src/train/evaluate.py` rolls out policies *without learning* and reports mean QoE
and its components. It compares the trained agents (acting deterministically)
against three scheduling heuristics, all using a common reactive bitrate rule
(target 90 % of recently measured aggregate goodput) so the comparison isolates
*scheduling* quality:

- **even** — uniform split every frame.
- **single** — the whole frame on the highest-recent-throughput path.
- **proportional** — split in proportion to recent per-path throughput.
- **learned** — the trained App + Transport agents.

The rollout reuses the env's `pop_app_window_reward` machinery so the reported QoE
is computed identically to training. A formatted table prints QoE, VMAF, latency,
loss, and bitrate per policy.

When the network is dynamic (§5.3.1), the heuristic baselines are **mask-aware**:
`even`/`single`/`proportional`/`random` all operate over the *active* paths only
(a dead path gets zero weight), so they are not handicapped by churn they cannot
see — the comparison still isolates scheduling quality. `_learned_policies` reads
the Transport checkpoint's `arch` tag and rebuilds either the flat or the scoring
agent, feeding it the matching (flat vs structured) observation. Each rollout also
records two dynamics diagnostics per frame — the **active-path count** and the
**split entropy** (nats; 0 = one path, `ln k` = even over `k` paths) — summarized
alongside the QoE components.

---

## 10. End-to-end worked example: the life of one frame

To make the data/control flow concrete, here is a single Transport-cadence frame
on the NS-3 backend (an app boundary additionally runs the App agent first):

1. **Body → bridge.** `RealtimeController` finishes the previous frame interval,
   fills `EnvStruct` (aggregate EWMAs, per-path cwnd/srtt/buffer/goodput/loss, last
   frame's latency/jitter/loss), and `CppSend`s it.
2. **Bridge → data plane.** `Ns3DataPlane._recv_obs` snapshots the struct into a
   `FrameObs`.
3. **Observation.** The training loop calls `build_transport_obs(obs,
   target_kbps)` → a 24-vector: `[bitrate_norm, rtt_norm, loss, thr_norm, (per
   path: cwnd_norm, srtt_norm, buf_norm, pthr_norm, ploss)×4]`.
4. **Policy.** `TransportAgent.select` runs the actor → raw `a ∈ [−1,1]⁴`, then
   `split = softmax(a)` (e.g. `[0.35, 0.32, 0.13, 0.20]` — note the latency-trap
   path 2 is down-weighted relative to its raw throughput).
5. **Action → body.** `step_frame` writes `ACT_STEP, targetBitrateKbps, split` to
   shared memory; `PySend` releases the body.
6. **Simulate.** The body computes `frame_bytes` (with I-frame/jitter), stripes it
   `[0.35, 0.32, 0.13, 0.20]` across the four subflows, enqueues byte-watermark records,
   and runs `1/30` s of packet-level simulation. As bytes arrive, sinks report
   cumulative counts; the controller resolves share completions, and when the
   frame's last share lands it computes `latency`, `jitter`, `late?`.
7. **Result.** The *next* `EnvStruct` carries this frame's `last_*` fields;
   `step_frame` returns them as a `FrameResult`.
8. **Reward + learn.** The env computes the transport reward
   `(1−loss) − b·lat − c·jit`, appends `(latency, jitter, loss)` to the App window,
   and the loop stores `(t_obs, a, r_t, t_next, done)` and runs one SAC update.

Across 30 such frames the App window fills; at the next app boundary its mean
latency/jitter/loss + the governing bitrate are scored by the QoE functional and
credited to the App agent's previous action.

---

## 11. Design rationale, assumptions, and limitations

**What the abstraction faithfully captures.** Per-path congestion control and its
cwnd/RTT dynamics; endogenous queueing delay and loss under bursty, heterogeneous
cross-traffic; application-controlled striping of a real-time, variable-bitrate,
deadline-bound source across paths; bufferbloat from overdrive; and the
quality/latency/loss tensions a real conferencing sender faces. The
observation/reward contract and the QoE functional are exactly the quantities the
problem statement specifies.

**What it abstracts away.** (i) *Transport identity:* subflows are TCP, not QUIC;
this preserves congestion/queue/loss behavior but not QUIC specifics (0-RTT,
stream multiplexing, connection migration, packet-number spaces). (ii)
*Reliability semantics:* TCP recovers loss via retransmission, so network loss
surfaces as latency, and *application* loss is modeled as deadline misses — a
deliberate, and arguably correct, real-time framing, but not identical to an
unreliable-datagram transport. (iii) *Reordering across paths:* the byte-watermark
tracker measures per-path completion and aggregates to a frame; it does not model
application-layer resequencing cost beyond the max-arrival latency. (iv)
*Single client/server.* The topology is a single client↔server dumbbell of `N`
parallel paths, not a multi-hop mesh. Both backends now vary the *active* path
count within a run via churn (§5.3.1, §3.9), and the scoring Transport agent
(§7.3.1) consumes that variable set; two residual gaps remain between the
backends' dynamics — they draw from independent RNG streams (behaviorally
equivalent, not frame-identical), and correlated-failure group indices `≥` the
topology's path count are dropped. (v) *Quality model:* the default VMAF
is a documented log-anchor stand-in, not measured per-content quality; the
optional `--learned-vmaf` surrogate (§6.3) is grounded in real WebRTC VMAF
measurements but is still a fixed, content-agnostic grid (bitrate axis saturating
~2500 kbps; delay/jitter axes degenerate until refit).

**Notable engineering choices.** The byte-watermark scheme yields per-frame
latency over pipelined reliable streams without per-packet tagging. `FlowMonitor`
loss is throttled to the app cadence to keep the per-frame path lean (the
bridge's hot loop). Episodes reset in-band to respect ns3-ai's single-creator
constraint while keeping congestion state warm across episodes — which makes the
non-stationarity more realistic but means episodes are not i.i.d. The mock backend
is a queueing surrogate, not a packet simulator; it is calibrated to match
*observable* behavior and is used for development/CI, with NS-3 reserved for
final results.

**Natural extensions.** Swap TCP for the `signetlabdei/quic` NS-3 module behind
the same `RealtimeSource`/`RealtimeStruct` interface; add FEC/retransmission and a
true unreliable-datagram loss model; condition the App agent on per-path summaries
or recurrent state; refit the learned VMAF surrogate over a denser
bitrate/loss/delay/jitter grid (or per-content rate-quality tables) to activate
its currently-degenerate delay/jitter axes; and parameterize `N`, topology, and
mobility per episode for domain randomization.

---

## 12. Appendix: dimensions, constants, and file map

**Observation/action dimensions (canonical NS-3 topology, N = 4 paths):**

| Quantity | Value |
|---|---|
| App observation dim | 5 (horizon-agnostic; no episode-progress feature) |
| App action dim | 1 (→ bitrate ∈ [300, 6000] kbps) |
| Transport observation dim (flat) | 4 + 5N = 24 (19 for the legacy 3-path config) |
| Transport state dims (scoring) | global 4, per-path 6 (5 + liveness), mask N |
| Transport action dim | N = 4 (→ softmax split; masked over active paths in scoring) |
| Bridge exchanges per second | fps = 30 |
| App decisions per second | 1 (every 30 frames) |

**Key constants:**

| Constant | Default | Where |
|---|---|---|
| fps / app period / deadline | 30 / 1 s / 180 ms | `configs/default.yaml`, C++ CLI |
| bitrate range / init | 300–6000 / 1500 kbps | `video` |
| reward weights `a,b,c,d` | 1.0, 0.5, 0.5, 1.0 | `reward` |
| latency / jitter normalizers | 200 / 50 ms | `QoEWeights` |
| VMAF anchors (default log curve) | (300, 25), (4300, 92) | `qoe.py` |
| learned VMAF | off by default; `--learned-vmaf` / `reward.use_learned_vmaf` | `learned_vmaf.py`, `reward_model.npz` |
| learned VMAF grid box | bitrate ≤ ~2500 kbps, loss 0–10 %, delay/jitter degenerate | `reward_model.npz` |
| SAC hidden / γ / τ / lr | 256 / 0.99 / 0.005 / 3e-4 | `SACConfig` |
| batch / buffer / start_steps / update_after | 256 / 200k / 1000 / 1000 | `SACConfig` |
| cwnd / buffer obs normalizers | 200 000 B | `realtime_env.py` |
| `transport_arch` | `flat` (default) / `scoring` | `configs/*.yaml` `run:` |
| `dynamics` (churn/regime/burst/corr) | off by default (both backends); on in `configs/dynamic.yaml` | `DynamicsConfig`, `dataplane.py`, `realtime_mpquic.cc` |
| paths (rate/delay/cross), NS-3 | built-in default 3/3/2.5/2 Mbps · 10/15/40/20 ms · 0.40/0.45/0.30/0.55; overridable via forwarded `topology:` | `ScenarioConfig`, `configs/*.yaml` |
| `cap_mbps` (throughput normalizer) | 12 (4-path) / 10 (legacy 3-path) | `configs/*.yaml` |
| paths (rate/delay/cross), legacy mock | 8/4/2 Mbps · 10/17/30 ms · 0.45/0.65/0.35 | `configs/default.yaml` |
| kMaxPaths | 8 | `realtime_mpquic.h` |

**File map:**

| Path | Role |
|---|---|
| `ns3/realtime_mpquic.h` | `EnvStruct`/`ActStruct` wire contract |
| `ns3/realtime_mpquic.cc` | NS-3 scenario: topology, subflows, frame generation, striping, completion tracking, decision loop, non-stationary dynamics (§3.9) |
| `ns3/realtime_mpquic_py.cc` | pybind11 binding of the structs + interface |
| `ns3/CMakeLists.txt`, `scripts/install_ns3_example.sh` | build + install into ns-3-dev |
| `src/ns3env/dataplane.py` | `DataPlane` ABC, `MockRealtimeDataPlane`, `Ns3DataPlane`, `FrameObs`/`FrameResult` |
| `src/ns3env/video_source.py` | frame-size model (mirrors C++) |
| `src/ns3env/qoe.py` | VMAF curve, App QoE reward (pluggable `vmaf_fn`), transport reward |
| `src/ns3env/qos_vmaf_reward.py` + `reward_model.npz` | vendored learned QoS→VMAF surrogate (from `WebRTC-QoE-Data-Generator`) |
| `src/ns3env/learned_vmaf.py` | adapter: learned surrogate → `vmaf_fn` (unit translation) |
| `src/ns3env/realtime_env.py` | observation builders, reward windows |
| `src/rl/sac_agent.py` | generic flat SAC (actor, twin critics, targets, entropy tuning) |
| `src/rl/scoring_sac_agent.py` | permutation-equivariant SAC (per-path scoring actor + DeepSets critic) for variable path counts |
| `src/rl/replay_buffer.py` | circular replay buffer + `StructuredReplayBuffer` (set-shaped transitions) |
| `src/rl/app_agent.py`, `src/rl/transport_agent.py` | action-space wrappers (the Transport wrapper dispatches flat vs scoring) |
| `src/train/config.py` | YAML → typed config; backend factories |
| `src/train/hierarchical_train.py` | dual-cadence training loop |
| `src/train/evaluate.py` | rollout + baselines |
| `train.py`, `evaluate.py` | thin CLIs |
| `configs/dynamic.yaml` | non-stationary, variable-path-count scenario (`transport_arch: scoring`) |
| `tests/` | mock-only unit + smoke tests (incl. `test_dynamics.py`, `test_scoring_agent.py`) |

---

*End of report.*
