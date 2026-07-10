# Tuning the network dynamics — when does the hierarchy matter?

How to move the simulated network (both backends) along the axis from
"App agent alone is sufficient" to "the Path (split) agent is essential",
so the boundary can be mapped systematically.

## The quantity that decides it

The single-agent `app_only` ablation uses the learned bitrate with an **even
split over active paths** (the liveness mask is free — dead-path avoidance never
differentiates the path agent). What differentiates it is **per-path
headroom**:

```
even-split share per path      ≈ bitrate / n_active
residual capacity of path i    ≈ rate_i × envelope_i(t) × regime_i × burst_i × corr_i × (1 − cross_i)
```

- **share ≪ weakest residual, always** → even split works, `app_only ≈ learned`
  (this was the pre-envelope NS-3 behavior: `runs/dyn-ns3-*-fixed`).
- **weak paths dip below the share for sustained periods, but aggregate residual
  still fits the bitrate** → a scheduler that routes around dips wins; the
  hierarchy separates (current tuning: `runs/dyn-ns3-*-tight`).
- **aggregate residual < any useful bitrate** → everything collapses, separation
  is muddy. Avoid this regime; check with the neutral-policy probes below.

So: to make the hierarchy matter, create *transient, per-path* capacity
shortfalls; to make the App agent sufficient, keep every path's residual
comfortably above `bitrate / n_active`.

## Knob layers

### 1. Topology (`topology.paths` in the YAML — both backends, no rebuild)

```yaml
- {rate: "3Mbps", delay: "10ms", cross_frac: 0.40}
```

| knob | mock effect | NS-3 effect | direction |
|---|---|---|---|
| `rate` | `base_mbps` of the analytic path | bottleneck link `DataRate` | ↓ rate ⇒ less headroom ⇒ more path-relevant |
| `delay` | base RTT (×2) | link propagation delay | ↑ delay eats deadline budget; ≥ `deadline/2` makes a path a latency trap |
| `cross_frac` | always-on analytic sinusoid, mean `0.5×cross_frac` of capacity | on/off UDP flood at `cross_frac × rate` while ON (duty ≈ 40–45%) | ↑ ⇒ deeper, path-specific dips |

Heterogeneity matters as much as level: identical paths make even-split near
optimal by symmetry. Skew `rate`/`cross_frac` so the *identity* of the weak path
changes over time.

### 2. `dynamics:` block (YAML — mirrored on both backends, no rebuild)

All rates are per-second hazards (`1 − exp(−rate·dt)` per frame). Off by
default; see `configs/dynamic.yaml` for the current headline values.

| mechanism | knobs | what it stresses |
|---|---|---|
| churn | `churn_up_rate`, `churn_down_rate`, `min_active` | mask-following only — does **not** separate `app_only` (the ablation gets the mask). ↓`min_active` / ↑`churn_down_rate` concentrates load on fewer survivors, which *does* tighten headroom |
| regime | `regime_rate`, `regime_lo`, `regime_hi` | **sustained** best-path swaps (persist ~`1/regime_rate` s). Deepening `regime_lo` (e.g. 0.55 → 0.35) is the most direct way to create long weak-path windows |
| burst | `burst_rate`, `burst_intensity`, `burst_duration_s` | transient dips the scheduler must dodge within frames. Expected fraction of time bursting ≈ `rate × duration` |
| corr | `corr_groups`, `corr_rate`, `corr_intensity`, `corr_duration_s` | defeats naive diversification across a shared bottleneck |

Semantics on both backends are multiplicative on capacity; on NS-3 the
cross-traffic rate scales proportionally so cross stays a *fraction* of current
capacity.

### 3. Capacity envelope + noise (code constants — **edit both sides + rebuild**)

The largest single lever, and the one that closed the mock/NS-3 ablation gap:

- **mock**: seasonal sinusoid `1 + amp·sin(2πt/period + phase)`, with
  `amp = 0.45 + 0.1·(i % 3)`, `period ∈ {12, 7, 20}` s — built in
  `ExperimentConfig.mock_dataplane` (`src/train/config.py`); Gaussian noise
  `noise_std = 0.05` (`MockRealtimeConfig`).
- **NS-3**: the same formulas duplicated in `EnvelopeMult()`
  (`ns3/realtime_mpquic.cc`), folded into `ApplyPathRate` — dynamics-gated so
  the static scenario stays byte-identical. Noise sigma set in `InitDynamics`.

↑ `amp` ⇒ deeper multi-second troughs ⇒ more path-relevant. Setting the
envelope amplitude near 0 on both sides largely restores "App-only sufficient"
on NS-3 even with churn/regime/burst on. **Keep the formulas identical in both
files**, then rebuild: `scripts/install_ns3_example.sh`.

### 4. Structural asymmetries (know them; change only deliberately)

These make the *same* parameters harsher on the mock than on NS-3:

- Mock sender queue (`_busy_until`) is unbounded — overload compounds into
  multi-second full-loss streaks. NS-3 drops a share at generation once a path
  holds > one deadline of backlog (`m_maxBufferBytes`), and UDP mode
  deadline-drops stale shares ⇒ bounded, partial-loss failures (the realistic
  behavior).
- Mock-only `network_loss` term: up to 10% frame loss whenever *any used path*
  is below 50% of its baseline (max over paths) ⇒ mock shows nonzero loss on
  ~90%+ of frames; NS-3 loss is only deadline-miss or dead-path bytes.

Expect the mock ablation gap to always be more extreme than NS-3's at equal
parameters.

## Workflow for a tuning experiment

1. Edit the YAML (and, for envelope changes, both code sites + rebuild).
2. **Cheap probes before any training** (minutes, no checkpoints):
   - `uv run python scripts/parity_check.py` — both backends survivable and in
     the same loss regime under neutral even split.
   - Fixed-bitrate sweep of `even` vs a reactive split (weight by
     `path_throughput` and de-weight backlog/srtt-inflated paths via
     `obs.buffer_occ` / `obs.srtt_ms`), driving the `DataPlane` directly. If
     even-split loss ≫ reactive loss at bitrates the reactive policy sustains
     cleanly, the scenario will separate; if *both* drown at every useful
     bitrate, it is over-tightened. (Pattern: `probe_oracle.py` from the
     2026-07-02 session — even 25–55% vs adaptive 2–5% loss at 2.5–3 Mbps.)
3. Train (~15 min per NS-3 run on this machine):
   `uv run python train.py --config configs/dynamic.yaml --backend ns3 --ns3-transport udp --episodes 100 --seed 1 --out-dir runs/<name>`
4. Evaluate with ablations:
   `uv run python evaluate.py --config configs/dynamic.yaml --backend ns3 --ns3-transport udp --app runs/<name>/app.pth --path runs/<name>/path.pth --ablation --figures --out runs/<name>/eval`
5. Read `app_only / learned` QoE and the deadline-miss gap
   (`eval/figures/figure12_ablation.png`).

Checkpoints do **not** transfer across environment changes — retrain per
configuration. Everything is deterministic per seed; use ≥2 seeds before
trusting a boundary point.

## Suggested sweep to map the boundary

Starting from `configs/dynamic.yaml` (which separates on both backends as of
2026-07-02: NS-3 `app_only/learned` ≈ 0.49–0.57; mock ≈ 0.07):

| direction | change (one at a time) |
|---|---|
| → App-only sufficient | envelope `amp` → 0.2 / 0.0 (both code sites); or `rate` +50%; or `regime_lo` 0.55 → 0.8; or `cross_frac` −0.15 |
| → Path essential | `regime_lo` 0.55 → 0.40; or `burst_rate` 0.10 → 0.20 with `burst_duration_s` 0.5 → 1.0; or `cross_frac` +0.10; or `min_active` 3 → 2 |
| sanity floor | if even-split loss at the *minimum* bitrate exceeds ~10% on NS-3, back off — the scenario is beyond the useful regime |

Reference points measured so far (UDP, seed 1000, 5 eval episodes):

| scenario | app_only/learned QoE | app_only miss |
|---|---|---|
| static (`four_path.yaml`) | ~0.98 | — |
| dynamic, pre-envelope NS-3 | 0.89 | 5.6% |
| dynamic, tightened NS-3 (current) | 0.49 | 20.2% |
| dynamic mock (unchanged) | 0.07 | 23.6% |
