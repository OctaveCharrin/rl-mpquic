# Implementation Roadmap

Sequenced code plan tying together the reward, agent-architecture, and role-profile
work. **Status: Phases 0–1 done; Phase 2 (G.1070 calibration), Phase 3 (role
profiles), and the executable parts of Phase 4 (4.1 domain randomization, 4.2
PopArt, 4.4 attention pool) are implemented — all off by default / additive. 2.1
(deadline softening) was skipped by decision; 4.3 (role-conditioned policy) and
4.5 (recurrence/HIRO) remain sketched.**

Companion docs (the "what" and "why"; this file is the "in what order"):
- `docs/REWARD_TUNING.md` — reward coefficient justification, calibration
  protocol (§5), role/use-case profiles (§7).
- `docs/RL_AGENT_DESIGN.md` — agent-architecture justification + improvement tiers.
- `docs/PATCH_TIER1_SAMPLE_EFFICIENCY.md` — concrete patch plan for P1/P2/P3.

## Ordering principle

Build the **measurement + speed** foundation first, so every later change is
judgeable and cheap to iterate; then calibrate the reward; then add the role
feature; then the research-grade architecture work.

**Critical path:** buffer persistence + A/B harness → sample-efficiency (P1, P2)
→ reward calibration → static role profiles → domain-randomization / return-scaling
→ role-conditioned policy & other architecture upgrades → online tradeoff inference.

## Phase 0 — Foundations (unblocks everything, low risk)

| # | Task | Why first | Effort | Source |
|---|------|-----------|--------|--------|
| 0.1 | **Replay-buffer persistence across `--resume`** (P3) | Every later phase needs long/chunked training; resume currently discards the buffer | S | `PATCH_TIER1` P3 |
| 0.2 | **Multi-seed A/B eval script** — wrap `evaluate.py --ablation`, dump `qoe_components`, compare across seeds | The ruler for all of Phase 1–3; no reward/algorithm change is judgeable without it | S–M | `REWARD_TUNING` §5B, `RL_AGENT_DESIGN` §4 |

## Phase 1 — Sample efficiency (makes all later experiments faster & better)

| # | Task | Why here | Effort | Source |
|---|------|----------|--------|--------|
| 1.1 | **Critic LayerNorm + higher UTD** (P1) | Biggest single win; accelerates every subsequent run. Must precede 1.2 (shared `_update_once`) | M | `PATCH_TIER1` P1 |
| 1.2 | **Prioritized Experience Replay** (P2) | Focuses learning on rare churn/regime/corr events; depends on P1's loss structure | M–L | `PATCH_TIER1` P2 |

Do these before touching the reward: a faster, more stable learner shortens the
Phase-2 calibration loop.

## Phase 2 — Reward calibration (now that training is fast + measurable)

| # | Task | Why here | Effort | Source |
|---|------|----------|--------|--------|
| 2.1 | *(optional)* **Soften the deadline→loss cliff** (step→ramp near `deadline_ms` in `step_frame`) | It's a reward-*shape* change; decide it **before** fitting numbers, or you calibrate the wrong function | S | reward `loss` semantics |
| 2.2 | **Implement §5A/§5B calibration** — pick oracle (repo learned-VMAF, or ITU P.1203/G.1070), fit `(b, c, d)` by correlation over the trace corpus; replace provisional `b=c=0.5` | Needs the Phase-0.2 harness + Phase-1 speed | M | `REWARD_TUNING` §5 |

## Phase 3 — Role / use-case profiles (level 1)

| # | Task | Why here | Effort | Source |
|---|------|----------|--------|--------|
| 3.1 | **Refactor `deadline_ms` into the profile bundle** (global today; most role-sensitive knob) | Prerequisite for real profiles | S–M | `REWARD_TUNING` §7 |
| 3.2 | **Static named profiles** `configs/profiles/{interactive,presenter,passive}.yaml` (+ `profile:` selector); calibrate each via the Phase-2 harness | Standards-backed (G.1010), low risk, immediately useful; reuses the calibration just built | M | `REWARD_TUNING` §7 |

## Phase 4 — Architecture upgrades enabling level-2 profiles (research-grade)

| # | Task | Why here | Effort | Source |
|---|------|----------|--------|--------|
| 4.1 | **Domain randomization over `DynamicsConfig`** per episode | Needed for robust and role-conditioned policies; machinery already exists | S–M | `RL_AGENT_DESIGN` Tier-2 #7 |
| 4.2 | **Return/reward scaling (PopArt)** | Profiles have different reward magnitudes → needed before mixing them in one policy | M | Tier-3 #9 |
| 4.3 | **Role-conditioned single policy** (feed profile into obs) — profiles **level 2** | Handles mid-call presenter hand-off natively; depends on 4.1 + 4.2 | L | `REWARD_TUNING` §7 level 2 |
| 4.4 | **Set-Transformer attention pool in scoring critic** | Targets `corr_groups` (mean-pool is blind to it); *independent* — slot in any time after Phase 1 | M | Tier-2 #5 |
| 4.5 | Recurrence/history (POMDP); HIRO manager off-policy correction | Diminishing returns; only if 4.1–4.4 leave gaps | L | Tier-2 #4/#6 |

## Phase 5 — Capstone (optional)

| # | Task | Effort | Source |
|---|------|--------|--------|
| 5.1 | **Mortise-style online tradeoff inference** — weights = normalized QoE gradient, per regime/role | L | `REWARD_TUNING` §5D |

## Dependencies & notes

- **Hard ordering:** 0.2 before Phase 1–3 (need the ruler); 1.1 before 1.2 (shared
  loss); 2.2 before 3.2 (calibrate, then profile); 4.1 + 4.2 before 4.3 (can't
  train a role-conditioned policy without them).
- **Parallelizable / independent:** 4.4 (attention pool) and 2.1 (deadline
  softening) can be done almost anytime after Phase 1.
- **Docs step per code phase:** end each phase with the documentation-update step
  from `PATCH_TIER1` (flip the relevant `RL_AGENT_DESIGN` §3 items to "done";
  update `CLAUDE.md` / `IMPLEMENTATION.md` / `ARCHITECTURE.md`).
- **Natural stopping points:** after Phase 1 you have a materially better learner;
  after Phase 3 you have the calibrated, role-aware system that motivated this
  work. Phases 4–5 are upside, not table stakes.

## Effort key

S = hours · M = a day or two · L = multi-day / research-grade.
