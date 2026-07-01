# CLAUDE.md

Guidance for working in this repository.

## What this is

Dual-agent hierarchical RL (PyTorch SAC) controlling real-time video over
abstracted multipath QUIC in NS-3, bridged by `ns3-ai` shared memory. The Python
side is the "brain"; the C++ NS-3 scenario is a thin "body".

## Layering (where logic lives)

- `ns3/` — **canonical** C++ scenario (`realtime_mpquic.{h,cc}`, pybind
  `realtime_mpquic_py.cc`, `CMakeLists.txt`). Vendored here, symlinked into
  `$NS3_DIR/contrib/ai/examples/rl-mpquic` by `scripts/install_ns3_example.sh`.
  Keep C++ thin: topology, per-frame push + split, transport-state reporting. No
  RL logic. It runs **one Env/Act bridge exchange per video frame**.
- `src/ns3env/dataplane.py` — `DataPlane` ABC + two interchangeable backends:
  `MockRealtimeDataPlane` (pure Python, trace-driven, for tests/iteration) and
  `Ns3DataPlane` (the bridge). They share the `FrameObs` / `FrameResult` types,
  which mirror the C++ `EnvStruct`. **Edit both backends together** when changing
  the observation/action contract, and keep `EnvStruct`/`ActStruct` (in
  `ns3/realtime_mpquic.h`) and the pybind binding in sync. The mock also carries
  an optional `DynamicsConfig` (churn/regime/burst/correlated failures) and a
  per-path `FrameObs.path_active` liveness mask (see below).
- `src/ns3env/realtime_env.py` — observation builders + reward window. No torch.
  Builds both the flat transport vector (`build_transport_obs`) and the structured
  `TransportState` (`build_transport_state`, `glob`/`paths`/`mask`) for scoring.
- `src/ns3env/qoe.py`, `video_source.py` — reward math and the frame-size model
  (the latter mirrors `RealtimeController::GenerateFrame` in C++).
- `src/rl/` — generic flat `SACAgent` and permutation-equivariant
  `ScoringSACAgent` (both in normalized `[-1,1]` action space), plus the
  `AppAgent` / `TransportAgent` wrappers that map to bitrate / split. The Transport
  wrapper dispatches on `arch` (`"flat"` | `"scoring"`).
- `src/train/` — config loader, the dual-cadence training loop, evaluation+baselines.
- `train.py` / `evaluate.py` — thin CLIs only.

## Two-timescale contract (important)

The bridge is **synchronous lockstep, one exchange per frame**, C++ leads with a
send. The Transport agent acts every frame. The App agent acts only when
`FrameObs.app_decision_due` is set (every `app_period_s`); its target bitrate
persists in `ActStruct` between app decisions, and the Transport agent's
observation includes that bitrate (the hierarchy). The App reward is the QoE
accumulated over the *window* of frames its bitrate governed
(`env.pop_app_window_reward()`), credited to the previous app action.

## Dynamic network state + dynamic-input model

Two coupled, **mock-only** features make *which path to send on* an informative,
time-varying decision (the static topology made an App-only policy reach ~98% of
the full system; on the dynamic scenario the same App-only ablation *collapses*):

- **Non-stationary dynamics** (`DynamicsConfig` in `dataplane.py`, wired via the
  `dynamics:` YAML block, off by default): path **churn** (appear/disappear →
  variable active count; bytes onto a dead path = loss), **regime shifts**
  (abrupt best-path swaps), **congestion bursts**, **correlated failures**.
  `FrameObs.path_active` is the liveness mask. Deterministic per seed; static
  behavior is byte-identical when disabled.
- **Scoring Transport agent** (`transport_arch: scoring`): consumes the structured
  `(glob, paths, mask)` state, shared per-path actor + masked-softmax split,
  DeepSets masked-mean critic — handles a variable/changing path set. `"flat"`
  (legacy fixed-dim MLP) is the default. Checkpoints are arch-tagged; eval/`resume`
  read the tag. `configs/dynamic.yaml` is the headline scenario.

**CONTRACT:** the NS-3 body does *not* yet emit churn/regime/burst dynamics or a
`pathActive[]` field — `Ns3DataPlane` fills `path_active` all-ones. Porting the
dynamics into `ns3/realtime_mpquic.{h,cc}` (add `EnvStruct.pathActive[kMaxPaths]`)
is the pending follow-up; keep the mock and C++ in sync when it lands.

## Dev loop

```bash
uv sync --extra dev
uv run pytest                                  # fast, mock-only, no NS-3
uv run python train.py --backend mock --episodes 50
# dynamic scenario + scoring (dynamic-input) transport agent:
uv run python train.py --config configs/dynamic.yaml --backend mock --episodes 50
uv run python evaluate.py --config configs/dynamic.yaml --backend mock \
    --app runs/<run>/app.pth --transport runs/<run>/transport.pth --ablation
```

Training checkpoints every episode; `train.py --resume` continues from the latest
(useful for interruptible/chunked runs).

NS-3 changes require a rebuild: `scripts/install_ns3_example.sh` then
`./ns3 run "ns3ai_realtime_mpquic --selftest"` to validate C++ alone before
running `--backend ns3`.

## Gotchas

- The venv **must be Python 3.12** — the `ns3-ai` pybind `.so` is `cpython-312`.
- `ns3-ai` allows only one shared-memory creator per process: use a single
  long-lived `Ns3DataPlane`; episodes reset in-band (`ACT_RESET`).
- Always `dp.close()` (or let training's `finally` do it) to send `ACT_TERMINATE`
  and free the shared memory.
