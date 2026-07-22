# Reward Coefficient Justification & Tuning Protocol

Scientific grounding for the QoE reward weights in `src/ns3env/qoe.py`
(`QoEWeights`), and a reproducible protocol for calibrating or replacing them.
Read alongside the reward code (`compute_qoe_reward`, `compute_path_reward`) and
the deadline-miss loss semantics in `MockRealtimeDataPlane.step_frame`.

## 1. What is being justified

The App-agent reward (`qoe.py:149`) and Path-agent reward (`qoe.py:208`) are:

```
R_app  = a·VMAF/100 − b·lat_n − c·jit_n − d·loss   (+ e·util, learned-VMAF only)
R_path = (1 − loss) − b·lat_n − c·jit_n
```

where `lat_n = min(2, latency_ms / latency_norm_ms)`,
`jit_n = min(2, jitter_ms / jitter_norm_ms)`, `loss ∈ [0,1]`, and the result is
clipped to `[−2, 1]`.

Current values (identical across `configs/default.yaml`, `dynamic.yaml`,
`four_path.yaml`; `smoke.yaml` omits the norms):

| Coeff | Symbol | Value | Role |
|-------|--------|-------|------|
| `a_quality` | a | 1.0 | perceptual quality (VMAF/100) — the numeraire |
| `b_latency` | b | 0.5 | one-way completion-latency penalty |
| `c_jitter` | c | 0.5 | inter-frame delay-variation penalty |
| `d_loss` | d | 1.0 | deadline-miss / drop penalty (dominant) |
| `latency_norm_ms` | — | 200 | latency normalizer |
| `jitter_norm_ms` | — | 50 | jitter normalizer |
| `e_util` | e | 0.0 / 0.25 | delivered-bitrate reward (learned-VMAF only) |
| `util_norm_kbps` | — | 3300 | utilization normalizer (VMAF knee) |

The claim of this document: the **functional form** and the **normalizers** are
directly grounded in telecom QoE standards and the RTC-RL literature; the three
free **tradeoff weights** (b, c, d relative to a) are principled defaults that
should be *calibrated* against a QoE oracle using the protocol in §5.

## 2. Why an additive weighted-sum of quality minus impairments (the form)

This is not an ad-hoc formula. It is the standard parametric-QoE structure:

- **ITU-T G.1070** ("Opinion model for video-telephony applications") and its
  underlying **E-model (ITU-T G.107)** build overall conversational quality from
  *separable, additively combined impairment factors* — coding/quality, one-way
  delay, and packet loss are each priced and summed. Treating VMAF-quality,
  latency, jitter, and loss as separate additive terms is exactly this model,
  specialized to real-time video.
- The RL-for-video literature uses the same linear structure, with published
  coefficient values:
  - **MPC** (Yin et al., SIGCOMM 2015):
    `QoE = Σ q(Rₖ) − λ·Σ|q(Rₖ₊₁)−q(Rₖ)| − μ·Σ(rebuffer) − μ_s·(startup)`.
    Default weights: `λ = 1` (smoothness), `μ = μ_s = 3000`, chosen so "1 s of
    rebuffering receives the same penalty as reducing a chunk's bitrate by
    3000 kbps."
  - **Pensieve** (Mao et al., SIGCOMM 2017), `QoE_lin`:
    `QoE = Σ q(Rₙ) − μ·Σ Tₙ − Σ|q(Rₙ₊₁)−q(Rₙ)|` with `q(R)=R` (Mbps),
    rebuffer penalty `μ = 4.3`, smoothness coefficient `1`. The value `4.3`
    equals the **maximum bitrate level (4.3 Mbps)** in their ladder — i.e. 1 s of
    stall is priced to cancel exactly one chunk of top-quality video.
  - **QARC** (Zhang et al., ACM MM 2018):
    `QoE = Σ(Vₙ − αBₙ − βDₙ) − γΣ|Vₙ−Vₙ₋₁|` (video quality V, bitrate B, delay
    gradient D), with reported `α = 0.2`, `β = 10.0`, `γ = 1.0` — while noting
    "there is no optimal [coefficient] pair that can fit any network conditions."

Our reward is the direct real-time-conferencing analog, with **rebuffering
replaced by the deadline-miss "loss"** (a frame past its playout deadline is
unusable — see `step_frame`: `late → loss = 1.0`). So the *shape* of the reward
is standards- and literature-backed; §3 pins the constants.

## 3. Per-coefficient justification

**`a_quality = 1.0` (the numeraire).** VMAF (Netflix; Li et al., 2018) is the
perceptual-quality scale, normalized to `[0,1]`. Fixing `a = 1` makes a
full-quality frame the unit of account: every penalty below is expressed in
"fractions of a lost frame of quality." This is the Pensieve convention (bitrate
utility is the reference unit), and it removes one degree of freedom by
construction.

**`latency_norm_ms = 200` (+ 2× soft cap at 400 ms).** Pinned to **ITU-T G.114**
one-way transmission-time thresholds: ≤150 ms *preferred* (transparent
interactivity), 150–400 ms *acceptable but degrading*, >400 ms *unacceptable*.
Normalizing by 200 ms places `lat_n ≈ 1` (a full quality-frame's worth of
penalty) right inside the degrading band, and the hard 2× cap lands the maximum
penalty at **400 ms = the G.114 unacceptability boundary**. The default
`deadline_ms = 180` sits inside this window. The mock's `latency_ms` is one-way
frame-completion latency, the same axis G.114 governs.

**`jitter_norm_ms = 50`.** Inter-frame delay variation (IPDV). Anchored to
**ITU-T Y.1541** network-QoS classes 0/1, whose IPDV upper bound is 50 ms, and to
typical RTC de-jitter buffers (tens of ms): beyond ~50 ms of variation a
fixed-size real-time jitter buffer begins to under-run and drop late frames.
`c` reaches a full quality-frame penalty at that operating point.

**`b_latency = c_jitter = 0.5` (the tradeoff — provisional).** Two claims:
- *Ratio b : c = 1 : 1.* In G.1070, conversational quality is sensitive to both
  mean delay and its variation at comparable magnitude once each is normalized to
  its own perceptual threshold — hence equal weight after normalization.
- *Absolute 0.5.* Chosen so that a frame simultaneously stressed on latency and
  jitter (each near its normalizer) costs `≈ 0.5 + 0.5 = 1.0` — one whole
  quality-frame — i.e. "a fully congested frame roughly cancels top quality"
  (the `QoEWeights` docstring). This magnitude is a **design choice**, calibrated
  in §5, not read off a standard.

**`d_loss = 1.0` (dominant penalty).** In RTC a late-or-lost frame is unusable,
so loss is the harshest impairment. Setting `d = a` follows a precise and
standard convention: **the stall/loss penalty is priced equal to one full unit
of top quality.** Pensieve sets `μ = 4.3` = its maximum bitrate (4.3 Mbps), so
1 s of rebuffering cancels exactly one chunk of highest-quality video; MPC sets
`μ = 3000` so 1 s of rebuffering equals a 3000 kbps quality drop. Our `d = a = 1`
is the direct RTC analog: a fully-lost/deadline-missed frame (`loss = 1`) wipes
out a full-quality frame (`VMAF/100 = 1`), dominating `b` and `c`. Here the
deadline-miss plays rebuffering's role.

**`e_util = 0.25`, `util_norm_kbps = 3300` (learned-VMAF only).** The learned
WebRTC QoS→VMAF surrogate is nearly bitrate-flat, erasing the rate-control
gradient; `e·(1−loss)·min(1, bitrate/util_norm)` restores it, gated so only
on-time bits pay. `util_norm = 3300` is the VMAF "knee" (just below the high
anchor `4300 kbps → VMAF 92` in `vmaf_for_kbps`). `e = 0.25` keeps the reward
below the `+1` clip. Off (`e = 0`) under the default bitrate-sensitive log curve
to avoid double-counting rate.

**Clip `[−2, 1]` and 2× soft caps.** `+1` = a perfect frame; `−2` = roughly two
full impairments stacked. Bounding the per-step reward is standard practice for
SAC target stability (bounded critic regression targets).

## 4. The honest caveat: fixed weights are provably regime-dependent

A single static `(b, c, d)` is a defensible *default* but cannot be optimal
across the non-stationary scenarios (`configs/dynamic.yaml`):

- **QARC** states plainly that "there is no optimal [coefficient] pair that can
  fit any network conditions."
- **Mortise** (Shen et al., USENIX NSDI 2026) is built on exactly this premise:
  the optimal QoS tradeoff depends on the *real-time network state*. Its central
  device is a **QoS tradeoff proxy** — it infers the application's *preferred*
  QoS tradeoff online from **real-time QoE gradients**, then derives the matching
  operating parameters via control-theoretic analysis, rather than fixing weights
  a priori.

The scientifically defensible answer to "why these numbers" is therefore not a
single citation but a *measurement procedure*. §5 gives one.

## 5. Protocol to find / validate the coefficients

### 5A. Fix the standards-pinned anchors (removes 4 of 8 free parameters)

Do **not** tune these — they come from standards:
`a_quality = 1` (numeraire), `latency_norm = 200` with 400 ms cap (G.114),
`jitter_norm = 50` (Y.1541), `util_norm = 3300` (VMAF knee). This reduces the
search to the tradeoff ratios `(b, c, d)`.

### 5B. Offline calibration against a QoE oracle (recommended first pass)

Standard QoE-model fitting — the same way G.1070's own coefficients were derived
(regression to subjective MOS):

1. **Choose an oracle** `Q(quality, latency, jitter, loss)` treated as ground
   truth: either an ITU-T MOS predictor (**P.1203 / P.1204 / G.1070**), or the
   repo's learned WebRTC QoS→VMAF surrogate (`learned_vmaf`).
2. **Assemble a trace corpus**: run `evaluate.py` with fixed seed sets across
   `default`, `dynamic`, and `four_path`. `qoe_components` (`qoe.py:230`) already
   logs per-frame `vmaf / latency_ms / jitter_ms / loss` — dump these.
3. **Grid-search `(b, c, d)`** (e.g. `b, c ∈ {0.25…1.0}`, `d ∈ {0.5…2.0}`). For
   each, compute the linear `R` on the logged components and its **rank/Pearson +
   Spearman correlation** against the oracle `Q` over the corpus.
4. **Select** the `(b, c, d)` maximizing correlation, subject to a monotonicity
   sanity check: each partial derivative must have the correct sign
   (`∂R/∂latency < 0`, etc.) and `d ≥ b, c` (loss stays dominant).

### 5C. Robustness / reward-hacking sweep (validate, don't just fit)

For each surviving candidate, train (or evaluate a fixed policy) and confirm:
the baseline/ablation ranking is stable (use `evaluate.py --ablation`), and no
degenerate policy wins — e.g. min-bitrate-to-dodge-loss, or single-path
collapse. A weight set that changes *which policy is optimal* pathologically is
rejected regardless of correlation.

### 5D. Mortise-style online tradeoff inference (the principled upgrade)

To go beyond one static vector, estimate the *local* marginal rate of
substitution between quality, delay, and loss from the QoE gradient — this is
literally Mortise's "infer the preferred QoS tradeoff from QoE gradients":

1. Define the oracle `Q(·)` (as in 5B; autodiff-able if it's the learned model).
2. Over a sliding window, estimate `∂Q/∂latency`, `∂Q/∂jitter`, `∂Q/∂loss`
   (finite differences from logged perturbations, or autodiff on the surrogate).
3. Set the effective weights to the normalized gradient:
   `b_eff ∝ |∂Q/∂latency|·latency_norm`, etc. (units-matched to the normalized
   penalties).
4. Because these are state-dependent, either recompute per detected regime and
   store a small lookup keyed on network state, **or** feed the current tradeoff
   vector into the agent observation so the policy conditions on it (fits the
   existing `build_path_state` global-context slot).

This converts the weights from hand-set constants into *measured* quantities —
the strongest scientific justification available.

## 6. Recommended default to ship now

Keep the standards-pinned anchors (§5A). Keep `a = 1`, `d = 1` (loss dominant,
Pensieve/RTC rationale). Adopt `b = c = 0.5` as the working default, documented
as **provisional pending the §5B calibration**. Treat §5D as the roadmap item
that makes the weights defensible rather than merely reasonable.

## 7. Role / use-case coefficient profiles

The coefficients above describe *one* operating point. But a video-conferencing
endpoint plays different roles, and the right QoS tradeoff **depends on the role
of the flow** — this is not a heuristic, it is the model of **ITU-T G.1010**
("End-user multimedia QoS categories"), which defines application categories
distinguished precisely by their tolerance to delay and loss, and explicitly
separates *conversational videophone* from *one-way video* ("more tolerance for
delay since there is no direct conversation involved"). The governing variable is
**interactivity**.

**Framing.** This pipeline controls the *sender* side (App sets the encoder
bitrate, Path sets the split, reward is measured at delivery). We never optimize
a receiver directly, so a profile is keyed on the **use-case of the flow**
(interactive / presenting / passive), not a literal sender-vs-receiver bit — a
receiver's needs enter as the requirements of the stream we send it.

**A profile is a bundle**, not just weights: the most role-sensitive knob is
`deadline_ms` (currently a single global in `ExperimentConfig` /
`MockRealtimeConfig` / `Ns3Config`, not in `QoEWeights`). A profile therefore
bundles the `QoEWeights` fields **and** `deadline_ms`. `a_quality` stays the
numeraire (`= 1`) in every profile so the weights remain comparable; the role
difference is carried by `b`, `c`, `d`, the normalizers, and the deadline.

### Starter profiles

Provisional values, to be calibrated per profile with the §5 protocol. Rationale
below the table.

| Profile | G.1010 class | a | b (lat) | c (jit) | d (loss) | latency_norm_ms (2× cap) | jitter_norm_ms | deadline_ms |
|---------|--------------|---|---------|---------|----------|--------------------------|----------------|-------------|
| **interactive** (two-way talking heads) | conversational videophone | 1.0 | 0.5 | 0.5 | 1.0 | 200 (→400) | 50 | 180 |
| **presenter** (screen-share / slides → attendees) | one-way video | 1.0 | 0.2 | 0.2 | 1.0 | 400 (→800) | 100 | 400 |
| **passive** (low-priority secondary camera / view-only) | background / streaming | 1.0 | 0.1 | 0.3 | 0.7 | 800 (→1600) | 150 | 800 |

- **interactive** — the current default. Conversational two-way video is the
  most delay-sensitive class: `latency_norm = 200` pins the penalty to G.114's
  150/400 ms band, `deadline = 180` sits inside it, and `d = 1` (loss dominant)
  because a late frame cannot be retransmitted in a live conversation.
- **presenter** — an *active presenter* sends mostly one-way content to
  attendees who can buffer, so the delay budget widens (looser `latency_norm`,
  `deadline = 400`) and delay/jitter are downweighted (`b = c = 0.2`). But
  screen-share/slide content is text-heavy and G.1010 puts one-way/streaming
  video at *near-zero loss tolerance*, so `d` stays high (`= 1`) — a dropped
  slide region is unreadable. Effectively a quality-and-integrity-first,
  delay-relaxed point.
- **passive** — the endpoint's own low-value outbound stream (e.g. an attendee's
  camera thumbnail), or a view-only feed. Delay is nearly irrelevant (`b = 0.1`,
  `deadline = 800`); continuity/smoothness matters more than instantaneous
  latency, so jitter keeps modest weight (`c = 0.3`); and because the stream
  itself is low-priority, even loss is relaxed (`d = 0.7`) so it yields path/rate
  budget to higher-value flows.

### Implementation levels

Three options, increasing power and cost. `QoEWeights` is already a per-run
dataclass loaded from the `reward:` YAML block, so this extends the existing
structure rather than rewriting it.

1. **Static named profiles (recommended first).** Ship
   `configs/profiles/{interactive,presenter,passive}.yaml` (or a `profile:` key
   selecting a preset), each carrying its `reward:` weights **and** `deadline_ms`.
   Calibrate each once via §5B. Low risk, standards-backed, immediately useful.
2. **Role-conditioned single policy (the upgrade).** Feed the profile into the
   observation — a one-hot role, or the weight vector itself — so a *single* agent
   learns role-appropriate behavior (contextual / multi-task RL). Fits the
   existing `build_path_state` global-context slot and the App observation, and it
   natively handles roles that **change mid-call** (an attendee starts presenting
   → the context input flips, no policy swap). Harder to train: pair with domain
   randomization over roles and return scaling (see `docs/RL_AGENT_DESIGN.md`
   §3 Tier-2 #7, Tier-3 #9), since reward magnitudes differ across profiles.
3. **Separate policies per role.** Simplest to reason about, most memory, no
   transfer between roles.

**Mid-call role changes** are an *event*, not a static session property
(presenter hand-off). Level 1 needs an explicit profile switch at that moment (a
natural App-cadence decision); level 2 absorbs it through the context input. If
live-meeting dynamics matter, that is the argument for level 2 — but ship level 1
as the calibrated, G.1010-grounded default first.

## References

- ITU-T G.1010, *End-user multimedia QoS categories* (application classes by
  delay/loss tolerance; conversational vs one-way video).
  <https://www.itu.int/rec/t-rec-g.1010-200111-i>
- Y. Shen et al., "Mortise: Auto-tuning Congestion Control to Optimize QoE via
  Network-Aware Parameter Optimization," USENIX NSDI 2026.
  <https://www.usenix.org/conference/nsdi26/presentation/shen-yixin>
- ITU-T G.1070, *Opinion model for video-telephony applications*.
- ITU-T G.107, *The E-model: a computational model for use in transmission
  planning*.
- ITU-T G.114, *One-way transmission time* (150 ms preferred / 400 ms limit).
- ITU-T Y.1541, *Network performance objectives for IP-based services* (IPDV).
- ITU-T P.1203 / P.1204, *parametric bitstream-based QoE / MOS models*.
- X. Yin, A. Jindal, V. Sekar, B. Sinopoli, "A Control-Theoretic Approach for
  Dynamic Adaptive Video Streaming over HTTP" (MPC), SIGCOMM 2015. Default QoE
  weights `λ = 1`, `μ = μ_s = 3000` (1 s rebuffer ≡ 3000 kbps quality drop).
  <https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p325.pdf>
- H. Mao, R. Netravali, M. Alizadeh, "Neural Adaptive Video Streaming with
  Pensieve," SIGCOMM 2017. `QoE_lin`: `q(R)=R`, rebuffer penalty `μ = 4.3`
  (= max bitrate 4.3 Mbps), smoothness coefficient `1`.
  <https://people.csail.mit.edu/hongzi/content/publications/Pensieve-Sigcomm17.pdf>
- T. Zhang, F. Ren, W. Cheng, X. Luo, R. Shu, X. Liu, "QARC: Video Quality Aware
  Rate Control for Real-Time Video Streaming via Deep Reinforcement Learning,"
  ACM MM 2018. `QoE = Σ(V − αB − βD) − γΣ|ΔV|`, reported `α = 0.2`, `β = 10.0`,
  `γ = 1.0`. <https://arxiv.org/abs/1805.02482>
- Z. Li et al., "VMAF: The Journey Continues," Netflix Tech Blog, 2018.
