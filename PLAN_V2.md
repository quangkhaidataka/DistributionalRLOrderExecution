# PLAN_V2.md — Design v2: Constrained fine-grid study (living doc)

> v3 of this doc, 2026-07-18. Branch: `feature/design-v2`. Old N=5 state frozen at tag
> `locked-main-results`.
> **Structure: B1–B4 are CODE + LOGIC-TEST ONLY — absolutely NO full training/eval runs.**
> Allowed runs in B1–B4: unit tests, smoke tests (≤200 episodes, outputs under `_smoke/`,
> never presented as results), statistical env validation (no-trade simulation), and the
> regression gate. **All real runs happen in B5**, triggered by the user saying
> "run AC" / "run Regime-Switching" / "run real data".
>
> Standing rules (apply to every step):
> - ADDITIVE changes only: every new behaviour behind config with defaults = old behaviour.
>   Legacy runners (`run_simulation.py`, `run_jd_staged.py`, `run_tag.py`, …) must keep
>   reproducing the locked results untouched.
> - v2 runners read/write ONLY under `results/_v2_*/`; v1 result paths are read-only
>   (only the regression gate reads a v1 checkpoint). Runners REFUSE to write into a
>   non-empty output dir unless `--force-resume` is passed.
> - Config JSON dumped for every job; param guard asserted on every network build;
>   `main_paper.tex` untouched; `PLAN.md` is the frozen v1 plan (reference only) — this
>   file governs all Design-v2 work.
> - Network: keep 2 hidden × d=64 (IQN cosine n=32). Do NOT enlarge. Escalation to d=128
>   is allowed ONLY at the Phase-2 pilot, only on documented underfit symptoms (quantile
>   loss plateau, flat action-vs-state heatmap, IQN ≪ DQN on val), one round max.
> - Watch-items at N=20 (do not change without symptom): C=500 now refreshes ~every 25
>   episodes (was ~100) — first knob if curves oscillate is C=1000, not width; ε-decay
>   semantics (per-step vs per-episode) must keep old meaning — document which.

## Design constants (frozen)

| Item | Value |
|---|---|
| Horizon | N=20, T=60 min, dt=3 min |
| Action grid (q0 mode) | a ∈ {0, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250} × q0; cap=0.25 |
| Oversell protection | mask (both selection AND target max) → clip x_t=min(a·q0,q_t) → assert q_t≥0 |
| Terminal | final period force-sells remainder, exempt from cap |
| State | 5-D [t/N, q/q0, Δp, spread, imb]; +σ̂_t (rv window 5, /σ, clip [0,5]) behind flag `use_rv_feature` (default off; decide at regime pilot) |
| Networks | IQN d=64/n=32/N_τ=N'_τ=8/policy τ=32/κ=1; DQN·DDQN width 64; Adam 1e-4, batch 64, C=500, discount 0.99, replay 100k |
| Protocol | train 40k eps (plateau gate at 30k → +10k), ckpt every 2k (20 candidates), val 1,200 eps CRN, select min val-CVaR₉₅ (+plateau report; mean-select secondary), test 10k eps |
| Seeds | sim: 42, 123, 7, 2024, 31 · TAQ: 42 only (by design) |
| Agents | TWAP · AC(λ̃=1e-6, continuous, cap slack) · MaxSpeed (cap until exhausted) · DQN · DDQN · IQN-neutral · IQN-CVaR-α (EVAL-ONLY distortion τ~U[0,α], α ∈ {0.3,0.5,0.7,0.9,0.95}) |

---

## B1 — Global engine refactor (+ logic tests)

### Files to code

| # | File | New/Modify | What to implement |
|---|---|---|---|
| 1 | `envs/base_env.py` | **Modify** | `EnvConfig` gains `action_basis` ('remaining' default \| 'q0'), `action_fracs` (default = old 6-level grid), `use_rv_feature` (False), `rv_window` (5). In 'q0' mode `step()` executes `x_t = min(a·q0, q_t)`. Add ONE function `feasible_action_mask(q_t)` = {a: a·q0 ≤ q_t} ∪ {smallest a with a·q0 > q_t (sell-remainder via clip)} ∪ {0} — the single source of truth for masking. `_build_state()` appends normalized realized-vol feature when flag on (re-enable the commented `_realized_vol`); `STATE_DIM` becomes dynamic. Terminal force-sell unchanged. Defaults must be byte-identical to current behaviour. |
| 2 | `envs/simulated_env.py` | **Modify (minimal)** | `SimConfig` inherits the new fields; NO behaviour change to `AlmgrenChrissEnv`/`JumpDiffusionEnv`/`RegimeSwitchingEnv`. |
| 3 | `networks/iqn_networks.py` | **Modify** | Input/output dims fully driven by (state_dim, n_actions) — remove any residual hard-coding; no architecture change (2×64, cosine 32). |
| 4 | `agents/param_utils.py` | **Modify** | Recompute expected trainable-param counts for (state_dim ∈ {5,6}) × (n_actions ∈ {6,11}); table of expected values keyed by config; assert + print on every build (replaces the 11,462/5,190 constants). |
| 5 | `agents/iqn_agents.py` | **Modify** | Action selection = masked argmax over (distorted) Q; ε-greedy samples only feasible actions; **Bellman target max also masked** (use the env's `feasible_action_mask` of the NEXT state); α-distortion path untouched. |
| 6 | `agents/baselines.py` | **Modify** | Same masking treatment for DQN/DDQN (selection + target max). Add `MaxSpeedAgent` (rule-based: trade at cap until inventory exhausted). Verify TWAP/AC agents are N-agnostic; AC executes continuous sinh-schedule amounts (NOT grid-projected). |
| 7 | `training/trainer.py` | **Modify** | Checkpoint interval from config (2,000); replay capacity from config (100k); no hard-coded 30k anywhere; ensure episode-range resume (`a:b`) saves/restores optimizer + RNG state so split runs are equivalent to single runs (pattern exists in `run_jd_staged.py` — lift it into the trainer if not already there). |
| 8 | `training/scheduler.py` | **Modify (check)** | Document + preserve ε-decay semantics (per-step vs per-episode) so N=20 keeps the old exploration meaning; expose in config. |
| 9 | `evaluation/metrics.py` | **Modify** | Add `cap_frac` metric (fraction of decision steps trading exactly at cap); add generic episode-flag conditional stats helper (used by B3 for stress-hit vs calm). |
| 10 | `experiments/exp_utils.py` | **Modify** | v2 output-dir helper with refuse-if-non-empty guard; reuse existing config-JSON dump. |
| 11 | `experiments/selection_lib.py` | **New (extract)** | Extract the core of `run_selection_appendix.py` into a reusable function: evaluate all checkpoints on 1,200-ep CRN validation (identical price-path seeds per checkpoint), primary min val-CVaR₉₅, plateau report (±1 ckpt within 10%), secondary min mean-IS. v2 runners call this; the old script keeps working. |
| 12 | `scripts/regression_gate.py` | **New** | **T1**: legacy-config eval of `results/_seeds/seed42/jump_diffusion/checkpoints/IQN-neutral_ep7000.pt` on 10k episodes → compare against stored `all_results.json` (Mean 1.8065 / Std 0.7865 / CVaR₉₀ 2.6998 / CVaR₉₅ 3.4605 / Max 12.1433); print old/new side-by-side + PASS/FAIL; exit non-zero on FAIL. |
| 13 | `tests/test_v2_env.py` | **New** | **T2** invariants (1,000 random action sequences: Σx=q0, q≥0, x≤0.25q0 for t<N−1, termination); **T3** mask consistency (masked actions never chosen over 10k random states; hand-built Q-table target-max ignores masked; both sites share the one mask function); **T4** baseline schedules @N=20 (TWAP=0.05q0×20; AC(1e-6)≈TWAP within 1%/period; MaxSpeed=(0.25×4,0…); zero-noise episode IS = hand-computed impact cost); **T6** CRN determinism (same seed twice → identical; two checkpoints see identical paths). |
| 14 | `tests/test_v2_smoke.py` | **New** | **T5**: 200-episode smoke train per learned agent (no NaN), checkpoint save/load round-trip, param guard prints expected counts. |

**Exit gate of B1: T1–T6 all PASS (T1 is the hard gate). Then commit + push.**

### B1 backlog (OPTIONAL — code at the next convenient touch; does NOT block B2–B4)

- **`feature_scale` config flag** (`envs/base_env.py`): a fixed per-feature multiplier
  vector applied in `_build_state`, default = all-ones (byte-identical legacy behaviour,
  T1 gate unaffected). Motivation: input features have unequal magnitudes — Δp* is
  ~1e-4–1e-3 while t*/q* are O(1) and spread* ~0.02 — so the price channel is numerically
  suppressed (hidden LayerNorm only partially compensates). Candidate value: scale Δp* by
  1/(σ√T) — "price move in cumulative-sigma units", O(1) with a clean financial meaning.
  NO running/statistical standardization (breaks reproducibility, moves Bellman targets,
  invalidates the regression gate). Activation is DATA-DRIVEN: only if the Phase-A
  feature-scale diagnostic (Pipeline 2 step A3b) shows the policy is price-blind AND the
  mechanism under study needs the price channel. Any activation is a config change logged
  in the decision log; legacy default keeps T1 passing.

---

## B2 — AC study (code + logic tests only)

### Files to code

| # | File | New/Modify | What to implement |
|---|---|---|---|
| 1 | `experiments/run_v2_ac.py` | **New** | Staged CLI modeled on `run_jd_staged.py`: flags `--only-agent {DQN,DDQN,IQN-neutral}`, `--episodes a:b` (IQN split 2×20k), `--seed S`, `--assemble-eval`, `--smoke`. Builds the v2 config: AlmgrenChrissEnv, q0=100,000, p0=100, η=2.5e-6, γ=2.5e-7, σ=0.00095, a=0.001, N=20, q0-grid/cap 0.25, 40k eps, ckpt/2k, replay 100k. `--assemble-eval` = run `selection_lib` (1,200 CRN) → test 10k eps for all 7 agents (incl. MaxSpeed, AC continuous) → write `results/_v2_ac/seed{S}/` (checkpoints/, logs/, config JSON). |
| 2 | `evaluation/tables_v2.py` | **New** | Table generators (do NOT modify old generators): (i) comparison table with **cap-frac** column → `comparison_table.txt`; (ii) 5-seed aggregate mean±std, txt + LaTeX booktabs (reuse `aggregate_seeds.py` core) → `_aggregate/ac_seed_summary.{txt,tex}`; (iii) α-ladder table (reuse `sweep_cvar_alpha.py` core; eval-only distortion on the SELECTED IQN-neutral checkpoint) → `alpha_ladder.{txt,tex}`. |
| 3 | `tests/test_v2_ac_pipeline.py` | **New** | Smoke through the CLI (200 eps/agent) into `results/_v2_ac/_smoke/`; table generators run on smoke outputs (format check only); **resume-split equivalence**: `--episodes 0:200` + `200:400` == `0:400` single run (same seed → identical checkpoint). |

### Tables produced later in B5
T-AC-1 `results/_v2_ac/seed{S}/logs/comparison_table.txt` (7 agents × Mean/Std/CVaR₉₀/CVaR₉₅/Max/cap-frac) · T-AC-2 `_aggregate/ac_seed_summary.{txt,tex}` (5 seeds) · T-AC-3 `alpha_ladder.{txt,tex}` (expect FLAT — sanity).

---

## B3 — Regime-Switching study (code + logic tests only)

### Files to code

| # | File | New/Modify | What to implement |
|---|---|---|---|
| 1 | `envs/regime_jump_env.py` | **New** | `RegimeJumpEnv` (inherits `AlmgrenChrissEnv`; existing classes untouched): hidden 2-state Markov volatility — σ_low=0.0005 fixed, σ_high & p₀₁ from config, p₁₀=0.4 fixed (mean stress 2.5 periods) — **plus compound-Poisson jumps ONLY in stress** (λ_J=0.1/period in stress, 0 in calm; σ_J=0.16, μ_J=0). Spread ×3 in stress (reuse existing mechanism). Expose per-episode info flag `stress_hit` (≥1 stress period) for conditional eval. Works with `use_rv_feature` on/off. |
| 2 | `experiments/run_v2_regime_scan.py` | **New** | Pilot scan modeled on `scan_jump_calibration.py`: **2×2 grid σ_high ∈ {0.002, 0.004} × p₀₁ ∈ {0.05, 0.10}**, seed 42, agents TWAP + DDQN + IQN-neutral only (pilot budget). Automatic criteria checker, hard-coded: (a) TWAP CVaR₉₅ ∈ [8,20] bps; (b) Std IS > 0.05 bps all learned; (c) cap-saturation < 50% of IQN-neutral steps; (d) soft: neutral−CVaR gap > 0 monotone in α. Output `results/_v2_regime/_scan/scan_summary.{txt,csv}` + cell recommendation. STOPS after the report (user locks the cell; ≤2 calibration rounds total). |
| 3 | `experiments/run_v2_regime.py` | **New** | Full-study staged CLI, same shape as `run_v2_ac.py`, env=`RegimeJumpEnv` with the LOCKED cell params (read from a `locked_cell.json` written at lock time); writes `results/_v2_regime/seed{S}/`. |
| 4 | `evaluation/tables_v2.py` | **Extend** | (iv) regime-conditional breakdown table (stress-hit vs calm episodes × per-agent {Mean, CVaR₉₅}); (v) action-vs-spread heatmap dump (csv + png via `visualizer.py`) — the input to the σ̂ on/off decision. |
| 5 | `tests/test_v2_regime_env.py` | **New** | Statistical env validation WITHOUT trading (10k no-trade episodes): stationary stress fraction ≈ p₀₁/(p₀₁+p₁₀) ±1%; jumps occur ONLY in stress periods; empirical jump rate ≈ 0.1/stress-period; spread multiplier fires in stress; plus 200-ep smoke per agent into `_smoke/`. |

### Tables produced later in B5
T-RG-0 `_scan/scan_summary.{txt,csv}` (4 cells × criteria PASS/FAIL) · T-RG-1 main comparison (seed 42) · **T-RG-2 FULL α-ladder — thesis headline** (expect Mean ↑, CVaR/Max ↓ monotone in α) · T-RG-3 stress-vs-calm breakdown · T-RG-4 5-seed aggregate.

---

## B4 — Real data study, TAQ AAPL 2014 (code + logic tests only)

### Files to code

| # | File | New/Modify | What to implement |
|---|---|---|---|
| 1 | `data/build_taq_3min.py` | **New** | Rebuild bars at **3-minute** resolution from `data/processed/AAPL_2014.parquet`; **split-adjust** (7:1 on 2014-06-09: divide all pre-split prices by 7); write `data/processed/AAPL_2014_3min_adj.parquet` (NEVER overwrite the old parquet); emit a validation report: price continuity across the split date, bars/day count, NaN scan, monotone timestamps, post-adjust summary stats (→ feeds thesis Table 3.1). |
| 2 | `envs/taq_env.py` | **Modify (additive)** | Config-driven bar source + N: v2 mode reads the 3-min parquet, episode = 20 consecutive bars **within a single day** (never straddle days); q0-grid actions + mask (from B1); legacy defaults (12-min bars, N=5) unchanged. |
| 3 | `experiments/run_v2_taq.py` | **New** | Staged CLI modeled on `run_tag.py`: folds train Jan–Jul / val Aug–Sep / test Oct–Dec; q0=5,000, η=1e-5, a=1e-4, 50k train eps, ckpt/2k, val 1,200 CRN, test 10k; seed 42; writes `results/_v2_taq/AAPL/`; refuses to start training unless the η-scale gate report exists and is marked OK. |
| 4 | `scripts/eta_scale_check.py` | **New** | η-scale gate: deterministic TWAP and MaxSpeed costs (bps) computed on a sample of test days at dt=3′; prints report (MaxSpeed must be meaningfully more expensive than TWAP mean; magnitudes sane vs v1). Run in B5 BEFORE training. |
| 5 | `tests/test_v2_taq_data.py` | **New** | Data-pipeline unit tests: adjusted 2014-06-06 close ≈ 2014-06-09 open within a normal daily move; no NaN bars; bars/day sane (~130 for 6.5h); episode windows never straddle days; 200-ep smoke per agent into `_smoke/`. |

### Tables produced later in B5
T-TAQ-0 data-validation report → Table 3.1 · T-TAQ-1 main comparison (7 agents × 6 stats) · T-TAQ-2 α-ladder.

---

## B5 — RUN phase (user-triggered; nothing starts automatically)

Common rules for all three pipelines: every training/eval unit is a separate staged job
< 30 min (harness reap ~34′); every job dumps config JSON; every split job must pass the
resume-equivalence assertion; all reads/writes under `results/_v2_*/` only; after each
pipeline finishes, update PROGRESS.md and commit tables (never checkpoints) on
`feature/design-v2`.

### Pipeline 1 — trigger: "run AC"

1. **Preflight**: confirm B1 exit gate (T1–T6 PASS) and B2 tests passed; confirm
   `results/_v2_ac/` is empty (or user explicitly asked to resume).
2. **Training loop** — for each seed S in {42, 123, 7, 2024, 31}, run 4 staged jobs:
   - Job 1: DQN, 40k episodes (~11′)
   - Job 2: DDQN, 40k episodes (~12′)
   - Job 3: IQN-neutral part A, episodes 0:20000 (~15′)
   - Job 4: IQN-neutral part B, episodes 20000:40000 (~15′, resume-equivalence asserted)
   Checkpoints every 2k → 20 candidates per agent per seed.
3. **Assemble-eval per seed** (cheap, no gradients):
   a. Selection: all 20 checkpoints × 3 learned agents evaluated on the 1,200-episode CRN
      validation set → pick min val-CVaR₉₅ per agent; print plateau report.
   b. Test: 10,000 episodes (separate eval seed) for 12 policy rows = 3 selected learned
      agents + TWAP + AC + MaxSpeed + IQN-CVaR-α for α ∈ {0.3, 0.5, 0.7, 0.9, 0.95}
      (the 5 α rows are distorted readings of the selected IQN-neutral checkpoint).
   c. Write `results/_v2_ac/seed{S}/logs/` + **T-AC-1** for that seed.
4. **Cross-seed stage** (after all 5 seeds): aggregate → **T-AC-2** (mean±std, txt+tex);
   α-ladder table (seed 42 primary; per-seed values in the txt) → **T-AC-3**.
5. **Final report, then STOP**: compare against the pre-registered sanity expectations —
   all agents ≈ equal, α-ladder FLAT, cap-frac ≈ 0. Any violation is reported as a finding,
   not silently absorbed.

Budget: ≈55′ compute/seed → ~5h total; ~20 train jobs + 5 assemble jobs.

### Pipeline 2 — trigger: "run Regime-Switching" (two phases, hard STOP between)

**Phase A — calibration scan (cheap pilot):**
1. Preflight: B3 tests passed (incl. the no-trade statistical env validation).
2. For each of the 4 cells (σ_high ∈ {0.002, 0.004} × p₀₁ ∈ {0.05, 0.10}), seed 42 only:
   - TWAP: eval-only (measures the environment's tail — criterion (a))
   - DDQN: train 40k (~12′) — watches for corner/cap-saturation (criterion (c))
   - IQN-neutral: train 40k (2 staged jobs) — then quick α-ladder eval (criterion (d))
   - 10k-episode eval per agent; criteria checker scores the cell on (a)–(d).
3. Outputs: **T-RG-0** (`_scan/scan_summary.{txt,csv}`, 4 cells × PASS/FAIL + recommended
   cell) and the action-vs-spread heatmap for IQN-neutral in the recommended cell.
3b. **Feature-scale diagnostic** (cheap, part of the scan report): per-feature std/range
   of the state vector over ≥1,000 episodes (no-trade + pilot-policy), printed next to
   the heatmap. Read together at the STOP: policy responds to spread ⇒ current scaling is
   adequate for the mechanism, change nothing; policy price-blind AND the price channel
   matters ⇒ consider enabling the `feature_scale` flag for Δp* (B1 backlog) before
   Phase B. Decision is the user's, recorded in the decision log.
4. **STOP — user decisions (Claude Code proposes, user disposes):**
   - Lock the calibration cell → write `locked_cell.json`.
   - σ̂ on/off: heatmap shows action varying with spread → keep 5-D state; heatmap flat →
     enable `use_rv_feature` (state 6-D; param guard changes accordingly — expected).
   - If NO cell passes: user may specify ONE new 2×2 grid (max 2 rounds total per the
     stopping rule); after 2 failed rounds → fallback to the locked-tables narrative.

**Phase B — full study (only after the user's lock):**
5. Same structure as Pipeline 1 steps 2–4, with env = `RegimeJumpEnv(locked_cell.json)`:
   5 seeds × (DQN, DDQN, IQN 2×20k) → per-seed assemble-eval → cross-seed stage.
   Extra outputs in assemble-eval: regime-conditional breakdown (stress-hit vs calm).
6. Tables: **T-RG-1** (main, seed 42), **T-RG-2** (FULL α-ladder — thesis headline;
   expectation: Mean monotone ↑, CVaR/Max monotone ↓ in α), **T-RG-3** (stress vs calm),
   **T-RG-4** (5-seed aggregate).
7. **Final report, then STOP**: criteria (a)–(d) on the full run; monotonicity verdict on
   T-RG-2 per seed; d=128 escalation only on documented underfit symptoms (one round).

**Phase A first, always. Phase B never auto-starts.**

### Pipeline 3 — trigger: "run real data" (gate before any training)

1. **Preflight**: B4 tests passed. If `data/processed/AAPL_2014_3min_adj.parquet` does not
   exist: run `data/build_taq_3min.py` first and present the data-validation report
   (**T-TAQ-0**: split continuity, bars/day, NaN scan) — any anomaly ⇒ STOP and ask.
2. **η-scale gate** (`scripts/eta_scale_check.py`, deterministic, no training): on ≥10
   sample test days at dt=3′, compute TWAP and MaxSpeed costs in bps.
   **STOP if suspicious**, meaning either direction of insanity:
   - impact too EXPENSIVE (MaxSpeed cost ≫ plausible, e.g. hundreds of bps): training
     would only learn "never trade fast"; IS distribution dominated by impact, not price
     risk — the study measures the wrong thing;
   - impact too CHEAP (MaxSpeed ≈ TWAP): fast selling is free, every agent pins the cap —
     the corner problem returns through the back door.
   Sanity reference: magnitudes within the same order as v1 (TWAP mean ≈ 1.6 bps) and
   MaxSpeed meaningfully above TWAP's mean. Adjusting η (or switching to price-relative
   impact) is a USER calibration decision — never auto-tune.
3. **Training, seed 42 only**: DQN (1 job), DDQN (1 job), IQN-neutral (staged 2–3 jobs;
   50k episodes at N=20 ⇒ IQN ~50′ total), checkpoints every 2k.
4. **Assemble-eval on the walk-forward fold**: selection on 1,200-episode CRN validation
   drawn from Aug–Sep; test 10,000 episodes from Oct–Dec; 12 policy rows as in Pipeline 1.
5. Tables: **T-TAQ-1** (main), **T-TAQ-2** (α-ladder). **Final report, then STOP**:
   headline comparisons vs TWAP, cap-frac per agent, and the answer to the central
   question — with dumping impossible, does IQN-CVaR still dominate the scalar agents on
   tail metrics?

Budget reference (CPU, N=20): 40k eps — DQN ~11′, DDQN ~12′, IQN ~30′; 50k eps (TAQ) —
DQN ~14′, DDQN ~15′, IQN ~50′ (split to stay < 30′/job).

---

## Decision log
- 2026-07-20: **Regime cell LOCKED (Phase-A decision): σ_high=0.002, p₀₁=0.05**
  (`results/_v2_regime/locked_cell.json`). Rationale: all 4 scan cells pass hard
  criteria (a) TWAP CVaR₉₅∈[8,20], (b) Std IS>0.05, (c) cap<0.5; the differentiator
  is the α-ladder, and (0.002,0.05) is the ONLY cell with all-positive α-ladder gaps
  AND the largest at every α (+0.95 → +2.41 bps). Soft criterion (d) monotone fails
  in ALL cells at pilot scale → to be judged on the 5-seed mean in Phase B, not the
  single-seed pilot. **σ̂ (`use_rv_feature`) ENABLED** for the full study (state 6-D;
  param guard IQN 11851 / DQN·DDQN 5579) — the Phase-A action-vs-spread heatmap was
  ~FLAT and Δp* is numerically suppressed, so the 5-D state's spread signal isn't
  driving behaviour. **`feature_scale` stays OFF** (one variable at a time; the fixed
  price-channel rescale remains B1 backlog, decided later on σ̂-on evidence). A single
  σ̂-on confirmation pilot (calibration round 2/2) precedes Phase B; the σ̂-off scan
  outputs are the "σ̂ off" arm of the ablation appendix.
- 2026-07-19: **B4 done** (code + logic tests; data built). `data/build_taq_3min.py`
  BUILT `AAPL_2014_3min_adj.parquet` (32,742 bars/252 days, 7:1 split-adjusted;
  split continuity +0.22%, bars/day median 130, 0 NaN). `taq_env.py` v2 mode additive
  (bar_source/bar_minutes, stride 1, q0-grid+mask, continuous TWAP/AC, σ̂ flag; legacy
  byte-identical). `run_v2_taq.py` staged CLI (folds Jan–Jul/Aug–Sep/Oct–Dec, reuses
  run_v2_ac.train_segment — made RNG-robust for TAQEnv's RandomState; refuses to train
  without an OK eta_scale_report). `scripts/eta_scale_check.py` RAN (η=1e-5): impact-only
  TWAP 0.251 / MaxSpeed 1.255 bps (5×); realized IS 0.96 / 1.73 (drift-dominated per day)
  → GATE OK. `tests/test_v2_taq_data.py` all green. Reading: η=1e-5 sane — impact real but
  secondary to price-drift risk (right for a tail study); formal go/no-go at B5.
- 2026-07-18: **B1–B3 done** (engine refactor + AC study + regime study; all logic-tests
  green, byte-identical T1 gate). See git log on `feature/design-v2` for the per-batch
  commits; run phase (B5) still user-triggered.
- 2026-07-19: **B3b done** — feature-scale diagnostic (Pipeline-2 step A3b) implemented
  in `run_v2_regime_scan.py` (`collect_feature_scale`; per-feature std/min/max over ≥1,000
  no-trade + IQN-neutral-policy episodes → a table in `scan_summary.txt`). Smoke confirms
  the backlog concern: **Δp* std ≈ 3–7e-4 vs q*/t* std ≈ 0.21–0.30 (Δp*/q* ≈ 0.001–0.003)
  → the price channel IS numerically suppressed.** The `feature_scale` FLAG itself stays
  backlog (decided at the Phase-A STOP). Format check added to `test_v2_regime_env.py`.
- 2026-07-19: feature scaling reviewed — env-level hand normalization exists
  (`_build_state`), no statistical standardization anywhere (by design). Added: optional
  fixed `feature_scale` flag to B1 backlog (default identity) + feature-scale diagnostic
  as Pipeline 2 step A3b. Running standardizers explicitly rejected (reproducibility /
  Bellman-target drift / regression-gate).
- 2026-07-18: doc v3 — file-by-file coding tables added per user request (B1: 14 files, B2: 3, B3: 5, B4: 5). Reuse identified: `sweep_cvar_alpha.py` (α-ladder core), `run_selection_appendix.py` (CRN selection core → extract `selection_lib.py`), `scan_jump_calibration.py` (scan pattern), `run_jd_staged.py` (staged CLI + resume pattern).
- 2026-07-18: keep network 2×64 (no capacity increase); d=128 only as symptom-gated pilot escalation; watch-items C=500 semantics and ε-decay semantics at N=20.
- 2026-07-18: v1 checkpoints/results stay at original paths (read-only); no file moves — separation is by the `results/_v2_*/` namespace + non-empty-dir guard.
- Appendices & paper-surgery intentionally out of scope of this doc.
