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
  `ns3/realtime_mpquic.h`) and the pybind binding in sync.
- `src/ns3env/realtime_env.py` — observation builders + reward window. No torch.
- `src/ns3env/qoe.py`, `video_source.py` — reward math and the frame-size model
  (the latter mirrors `RealtimeController::GenerateFrame` in C++).
- `src/rl/` — generic `SACAgent` (operates in normalized `[-1,1]` action space)
  and the `AppAgent` / `TransportAgent` wrappers that map to bitrate / split.
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

## Dev loop

```bash
uv sync --extra dev
uv run pytest                                  # fast, mock-only, no NS-3
uv run python train.py --backend mock --episodes 50
```

NS-3 changes require a rebuild: `scripts/install_ns3_example.sh` then
`./ns3 run "ns3ai_realtime_mpquic --selftest"` to validate C++ alone before
running `--backend ns3`.

## Gotchas

- The venv **must be Python 3.12** — the `ns3-ai` pybind `.so` is `cpython-312`.
- `ns3-ai` allows only one shared-memory creator per process: use a single
  long-lived `Ns3DataPlane`; episodes reset in-band (`ACT_RESET`).
- Always `dp.close()` (or let training's `finally` do it) to send `ACT_TERMINATE`
  and free the shared memory.
