# Evaluation & Figures

Compare the trained hierarchical controller against the scheduling baselines
(`even`, `single`, `proportional`) and render an exhaustive figure set.

## Two steps

1. **Evaluate** — run the policies, record metrics + per-method decision
   (inference) times + per-frame traces + distributions, dump
   `evaluation_results.json`:

   ```bash
   # learned vs baselines (needs trained checkpoints)
   uv run python evaluate.py --backend mock \
       --app runs/<ts>/app.pth --transport runs/<ts>/transport.pth \
       --episodes 5 --out runs/eval-001

   # baselines only (graceful fallback — no checkpoints needed)
   uv run python evaluate.py --backend mock --episodes 5 --out runs/eval-001
   ```

   The data plane is backend-agnostic (`--backend ns3` works too); the figures
   read only the JSON, so they don't care which backend produced it.

2. **Plot** — turn the JSON into figures (PNG) under `<run>/figures/`:

   ```bash
   uv run --extra viz python evaluation/generate_figures.py runs/eval-001
   ```

   Or do both at once: add `--figures` to the `evaluate.py` call.

   > Needs `matplotlib` — installed via the `viz` extra (`uv sync --extra viz`).

## Figures produced

| File | What it shows |
|------|---------------|
| `figure1_qoe` | App-agent QoE per method (mean ± std) — the headline metric |
| `figure2_metric_panels` | VMAF, latency (p50/p95 vs deadline), loss / deadline-miss |
| `figure3_qoe_distribution` | QoE distribution (box plot, real per-window samples) |
| `figure4_decision_time` | **Decision/inference time per method** (App vs Transport, log scale) + per-frame split-time box plot |
| `figure5_quality_vs_cost` | QoE vs decision time — the quality/compute Pareto view |
| `figure6_timeseries` | Representative episode: bitrate, throughput, latency, loss over time |
| `figure7_split_behavior` | Learned per-path traffic split over time (stacked area) |
| `figure8_latency_cdf` | Latency CDF per method + deadline-miss-rate bar |
| `figure9_radar` | Normalized multi-metric radar (QoE / VMAF / low-latency / low-loss / fast-decision) |
| `summary_table.csv` | Machine-readable per-method summary |

`figure4` and `figure5` are where the hierarchical controller's compute cost
shows up: the SAC forward passes cost ~0.1–0.3 ms per decision vs ~0.002 ms for
the heuristics — still far inside a 33 ms (30 fps) frame budget.
