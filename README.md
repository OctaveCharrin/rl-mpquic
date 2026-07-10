# rl-mpquic

Dual-agent **hierarchical reinforcement learning** for real-time video
conferencing (WebRTC-like) over **(abstracted) Multipath QUIC**, simulated in
NS-3 and trained through the `ns3-ai` shared-memory bridge with PyTorch SAC.

Two agents cooperate at two timescales:

| Agent | Cadence | Observes | Acts |
|-------|---------|----------|------|
| **App** | every 1 s | aggregate RTT / jitter / loss / throughput / bitrate | target encoder bitrate (kbps) |
| **Path** | every frame (~33 ms) | per-path cwnd / sRTT / send-backlog / goodput / loss **+ the App agent's target bitrate** | per-path traffic split (softmax) |

The reward is a real-time video **QoE**: `a·VMAF(bitrate) − b·latency − c·jitter − d·loss`.

## Architecture

```
configs/default.yaml      one config drives both backends
        │
train.py / evaluate.py    thin CLIs
        │
src/train/                config loader + dual-agent loop + evaluation/baselines
        │
src/ns3env/               HierarchicalRealtimeEnv  (obs builders + QoE reward window)
        │   ├── qoe.py            VMAF curve + QoE / path rewards
        │   ├── video_source.py   frame-size model (mirrors the C++ generator)
        │   └── dataplane.py      DataPlane ABC | MockRealtimeDataPlane | Ns3DataPlane
        │
src/rl/                   SACAgent (generic) + AppAgent / PathAgent wrappers
        │
ns3/                      C++ NS-3 scenario (vendored; symlinked into ns-3-dev)
```

**Multipath QUIC is abstracted** as N independent single-path TCP subflows, each
with its own bottleneck, AQM queue and bursty UDP cross-traffic; the application
layer splits each frame's bytes across the subflows per the agent's ratio. NS-3
has no native QUIC, and this keeps the focus on the scheduling/rate-control RL.

The **mock backend** mirrors the NS-3 bridge interface exactly (trace-driven
per-path capacity + a standing-queue model), so the env, reward, agents and
training loop run and are tested without compiling NS-3.

## Setup

Requires Python **3.12** (to match the `ns3-ai` pybind `.so` ABI). With
[`uv`](https://github.com/astral-sh/uv):

```bash
uv sync --extra dev
```

## Mock workflow (no NS-3 — runs anywhere)

```bash
uv run pytest                                    # unit + smoke tests
uv run python train.py --backend mock --episodes 50
uv run python evaluate.py --backend mock \
    --app runs/<ts>/app.pth --path runs/<ts>/path.pth
```

## NS-3 workflow (Linux / WSL2)

Build the C++ scenario into the existing NS-3 tree (`$NS3_DIR`, default
`~/ns-3-dev`) — this symlinks `ns3/` into `contrib/ai/examples/rl-mpquic`,
registers it, and builds:

```bash
scripts/install_ns3_example.sh
```

Sanity-check the scenario without the bridge (C++-only even-split episode):

```bash
cd ~/ns-3-dev && ./ns3 run "ns3ai_realtime_mpquic --selftest"
```

Then train / evaluate against the real simulator:

```bash
uv run python train.py --backend ns3 --episodes 5 --show-output
uv run python evaluate.py --backend ns3 \
    --app runs/<ts>/app.pth --path runs/<ts>/path.pth
```

## Configuration

Everything is in `configs/default.yaml`: topology (per-path rate / delay /
cross-traffic), video source (fps, bitrate range, frame variability), episode
timing, QoE reward weights, and SAC hyperparameters. The same `topology.paths`
become NS-3 link attributes and the mock's per-path capacity baselines, so the
two backends are directly comparable.
