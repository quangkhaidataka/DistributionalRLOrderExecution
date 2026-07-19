# PROGRESS.md — DisRL project state

> Living status doc for the DisRL thesis work. **Update this whenever meaningful progress is made** (task finished, batch run, blocker hit). Newest state at the top of each section. Dates are absolute. Companion: **DECISION.md** (why), **PLAN.md** (the plan), **RUNBOOK.md** (run commands), **VERIFY.md** (checks), **PAPER_FIXES.md** (paper audit).

**Last updated:** 2026-07-19
**Active branch:** `fix/unify-arch-jd-recalibration` (off `main`) · **Design-v2 work on `feature/design-v2`**
**Repo:** `/Users/user/Desktop/DisRL` · conda env `finrl_env` (py3.11, torch 2.2.2, numpy 2.4.6) · MPS available.

> **Design-v2 (branch `feature/design-v2`, gov. doc `PLAN_V2.md`):** B1–B4 code COMPLETE + logic-tested (see PLAN_V2 decision log). **B5 Pipeline-1 (AC study) RUN & COMPLETE (2026-07-19)** — 5 seeds {42,123,7,2024,31} × (DQN, DDQN, IQN-neutral 2×20k) trained N=20 at CPU, per-seed 1,200-CRN selection → 10k test (11 rows). Tables in `results/_v2_ac/` (T-AC-1 per-seed, T-AC-2 `_aggregate/ac_seed_summary.{txt,tex}`, T-AC-3 α-ladder). **Verdict: all pre-registered AC-sanity expectations HELD** — all agents ≈ equal (learned mean 1.73–1.84, CVaR₉₅ 2.43–2.48 bps, overlapping within seed-std), α-ladder FLAT (Mean/CVaR₉₅ vary < across-seed std over α∈{.3,.5,.7,.9,.95,1}), cap-frac ≈ 0 for all except MaxSpeed (=1.0); TWAP≡AC. AC confirms the Gaussian null (no CVaR benefit) — the intended sanity check. Ops note: harness reaps background jobs on a ~34-min window, so each seed ran as staged sub-jobs (DQN/DDQN, IQN 0:20k, IQN 20k:40k, assemble) via an idempotent driver; the proven `--episodes a:b` resume made reaps free to re-run. **B5 Pipeline-2 PHASE A (regime calibration scan) RUN & COMPLETE (2026-07-19)** — 2×2 grid σ_high∈{.002,.004}×p₀₁∈{.05,.10}, seed 42, 40k eps, per cell {TWAP eval, DDQN, IQN-neutral 2×20k staged} → 10k eval + α-ladder + criteria (a)–(d) + feature-scale. `run_v2_regime_scan.py` got additive reap-safe staging (`--stage {ddqn,iqn_a,iqn_b,assemble}`; results-neutral). **All 4 cells HARD-PASS (a,b,c); RECOMMENDED (0.002,0.05)** — largest α=0.95 gap +0.947 and the only cell with all-positive & largest α-ladder gaps. Outputs in `results/_v2_regime/_scan/` (`scan_summary.{txt,csv}`, per-cell metrics.json, action-vs-spread heatmap for the recommended cell). Feature-scale: Δp* std ~3–7e-4 vs q*/t* ~0.2–0.3 (price channel suppressed); the action-vs-spread heatmap is ~FLAT (IQN trades ≈0.1·q0 across all spread bins) → the policy barely responds to spread. **Phase-A decisions (user, 2026-07-20):** cell LOCKED (0.002,0.05) → `results/_v2_regime/locked_cell.json`; σ̂ (`use_rv_feature`) ENABLED (6-D; IQN 11851/DQN·DDQN 5579); feature_scale stays OFF. **σ̂ confirmation pilot (round 2/2, seed 42, IQN-neutral 40k)** in `results/_v2_regime/seed42/logs/` (`pilot_sigma_ablation.{txt,json}`, 3 heatmaps). **Finding — σ̂-on is a mixed/negative result:** CVaR₉₅ improves 7.90→5.42 bps (−31%) BUT via a faster near-constant liquidation (~0.2·q0/step, front-loads in ~5 steps; mean IS worsens 1.93→2.65) that IGNORES both spread and σ̂ per-step (action-vs-σ̂ heatmap FLAT) and NEUTRALIZES the CVaR distortion (α-ladder gaps → 0.000 vs σ̂-off's +0.5…+1.2). Single seed — could be training variance; the CVaR-benefit story lives in the σ̂-OFF arm. Flagged for the user's Phase-B σ̂ go/no-go. **HARD STOP after the pilot report; Phase B on explicit go only.** B4 TAQ run phase still user-triggered. v1 locked results frozen at tag `locked-main-results`.

---

## Current status: Batches A/B1/C/E COMPLETE, D dropped → final paper-number handoff

**Batch E DONE** (cheap CPU review follow-ups; `results/_batch_e_report.md`):
- **E1 — Immediate-Liquidation baseline** (`ImmediateLiquidationAgent` + `run_il_baseline.py`, AC/JD/TAQ → `results/_il_baseline/`): IL=(2.084,2.084) in AC/JD (= DDQN's JD collapse); **on TAQ IQN-CVaR₀.₉₅ ≡ IL exactly** — the headline TAQ CVaR win is a dump-at-t₀ policy.
- **E2 — DDQN jump-sensitivity rows** (`run_jump_sensitivity.py --ddqn-level`, new subdirs): DDQN dumps at all 3 λ (dump 0.965/0.999/0.999), even at low λ where IQN-neutral does not.
- **E3 — selection-rule appendix** (`run_selection_appendix.py`, eval-only, 1200-ep CRN val, 5 seeds → `results/_selection_appendix/`): **DDQN's dump-collapse is a CVaR-selection artifact — it un-dumps under mean-IS selection at 5/5 JD seeds** (Std ~4.4, dump 0.000). Tail-based checkpoint selection *picks* the dump checkpoint; the agent isn't intrinsically degenerate. Main tables stay cvar-selected; this is appendix material.

Batch D (width ablation) dropped (not blocking). Two findings feed the paper narrative (PAPER_FIXES §4.3): the TAQ headline = IL, and the DDQN collapse = selection artifact.

<details><summary>previous status: Batches A/B1/C complete — paper-number handoff (kept)</summary>

**Locked JD config: λ_J=0.05, σ_J=0.16, μ_J=0, cvar-selection.** Batches **A** (seed-42 headline + TAQ), **B1** (AC+JD × 5 seeds, staged CPU + aggregated), and **C** (jump sensitivity re-centred on the locked cal + impact-misspec 3×3) are all done and green. Consolidated results: `results/_batch_bc_report.md`; aggregates in `results/_aggregate/`, `results/jump_sensitivity_summary.*`, `results/impact_misspec_*/`.

**Headline findings:** (1) **DDQN dump-collapses at 5/5 JD seeds** (dump-signature CVaR₉₀=CVaR₉₅; Std<0.05 at 2/5) — the reclassified scalar-vs-distributional finding, now at full seed coverage. (2) **Differentiation Δ(α=0.95) = +0.098 ± 0.226 bps, positive 3/5 seeds** — the zero-cost CVaR effect is real on average but small and seed-dependent (dominated by the one seed where IQN-neutral isn't already tail-aggressive). (3) **Jump sensitivity:** low λ → IQN under-hedges (high tail); base λ=0.05 → sweet spot (−73% vs TWAP); high λ → IQN itself dumps → locked λ is the non-degenerate regime. (4) **Impact misspec:** all agents robust, CVaR₉₅ scales smoothly ±2×; η=γ=1.0 reproduces the locked table. (5) AC is a clean Gaussian sanity check (CVaR gives no benefit). Next: hand numbers to PAPER_FIXES WILL-CHANGE rows (list below / in the final report).

### ▶ Next step
1. Populate PAPER_FIXES WILL-CHANGE rows from the aggregated tables (JD/AC/TAQ DQN·DDQN rows; jump-param rows; JD prose with the *honest* weak-differentiation framing). Do NOT edit main_paper.tex.

</details>

<details><summary>superseded status: staged JD retrain + R.4 gate FAIL (kept for history)</summary>

### staged full-scale JD retrain DONE → R.4 (original rule) FAILED → resolved by the 2×2 diagnostic + rule amendment

Locked **(λ_J=0.05, σ_J=0.16)** and ran the 30k-ep JD retrain **staged** (one agent per job) via `experiments/run_jd_staged.py` — validated byte-identical to the monolithic path (equivalence smoke) and to split `--eval-agents`. **Ran on CPU, not MPS:** the MPS DQN job was reaped at ep16000/30000; CPU is ~6× faster here (DQN 164s, DDQN 183s, IQN-neutral 484s — all ≪ 34 min). Order-independent per-agent RNG reseed added to `train_agent`+`evaluate_all` makes staged ≡ monolithic and CPU deterministic.

**R.4 gate = FAIL.** Table at `results/_seeds/seed42/jump_diffusion/logs/comparison_table.txt`; diagnostics at `logs/gate_diagnostics.json`.
- ✅ Meaningful tail (TWAP CVaR₉₅ 12.63 ∈ [4,15]); IQN-neutral cuts tail to 3.46 (−73% vs TWAP); IQN-neutral dump 0.000; DQN/IQN/IQN-CVaR Std > 0.05; param guard 11462/5190.
- ❌ **DDQN degenerate** — dump-at-t0 fraction 0.999 → Std IS 0.0092 < 0.05 (**triggers the FALLBACK RULE**: any learned agent Std < 0.05).
- ❌ **No differentiation at headline α** — IQN-CVaR₀.₉₅ CVaR₉₅ (3.483) not < IQN-neutral (3.461); sign flips across eval seeds (within noise). α-sweep shows a reduction only at aggressive α=0.5 (3.35 vs neutral 3.54).

**Resolution (2026-07-17):** the 2×2 diagnostic (D1 σ=0.16 selection test; D2 σ=0.12 fallback retrain; `results/_jd_diagnostics/decision_matrix.md`) showed all four cells fail the *original* rule, driven by DDQN's σ/selection-independent collapse. User locked **σ=0.16-cvar** and **amended the rule** (DDQN collapse → reported finding; IQN-only env-health criteria). See the ★ section in DECISION.md.

</details>

---

## Done (chronological)

### Onboarding / planning
- **CLAUDE.md** written (repo guide). **PLAN.md** + **PAPER_FIXES.md** written and user-approved: verified findings F1–F4, decisions D1–D4, robustness suite R1–R5, run schedule, and a full `main_paper.tex` audit (paper never edited directly).

### Phase 1 — code fixes (branch, one commit per task)
- **P1-T0** unblock `envs` imports (dropped moved `lobster_env`). **On this branch `from envs import ...` works.**
- **P1-T1/T2** unify DQN/DDQN width 128→64; add `agents/param_utils.py` param-count guard. **IQN = 11,462 trainable params, DQN/DDQN = 5,190.** Guard wired into all build sites.
- **P1-T3** fix jump-intensity bug (was `poisson(λ·dt)` with dt=12 → ~18 jumps/ep; now per-period) + recalibrate.
- **P1-T4** config hygiene: YAML marked reference-only, matched to dataclass defaults.
- **P1-T5** fix stale comments/tests: **state is 5-D (not 6-D), action space is 6 (not 5)** — paper was right, code was stale.
- **P1-T6** remove per-env architecture branch in `eval_simulation.py`.
- **P1-T7** add `--device {cpu,mps}` (default cpu) to `run_simulation.py` + `run_tag.py`.
- **N9** delete dead half-spread block in `taq_env.py` (fill = `mid - eta*x_t`, behavior unchanged).

### Phase 2 — robustness infrastructure (branch)
- **P2-T1** `run_seeds.py` (multi-seed) + `aggregate_seeds.py` (mean±std, LaTeX booktabs) + `exp_utils.py` (config dump, non-empty-dir guard) + `tests/test_aggregate.py`.
- **P2-T2** `sweep_cvar_alpha.py` (CVaR-α frontier, eval-only).
- **P2-T3** `run_jump_sensitivity.py`. **P2-T4** `run_impact_misspec.py`. **P2-T5** `run_width_ablation.py`.
- **P2-T6** `--smoke` flags + shared `dump_config_json`.

### Verification fixes (surfaced by VERIFY smoke runs)
- `evaluation/visualizer.py`: `tensor.numpy()` → `.tolist()` (numpy 2.x + torch 2.2.2 ABI break).
- `tests/test_agent.py` (stale module renames + numpy) and `tests/test_baseline.py` (stale 6/5 dims, N=10→5, vacuous DQN/DDQN assertion) repaired. **test_agent 27/27, test_baseline 76/76, run_tests 32/32.**

### Batch A executed (seed 42, MPS, ~2.7 h) — see gate report
- Re-ran AC + JD (`results/_seeds/seed42/…`), TAQ DQN/DDQN retrain (`results/taq/AAPL_seed42/`, kept IQN per D2), 3 CVaR sweeps.
- **TAQ headline intact:** IQN-CVaR₀.₉₅ CVaR₉₅ 6.13 / Max 6.24 → 88.8% / 97.0% reduction vs TWAP.
- **JD found DEGENERATE:** adverse μ_J=-0.30 made immediate liquidation dominant → every agent dump-at-t0, **IQN Std IS = 0.000, IQN-CVaR ≡ IQN-neutral** (no differentiation). Also: JD IQN training unstable (best ep14000, degraded by ep30000); the sweep grabbed the wrong (newest) checkpoint.

### JD recalibration — code only (T1–T5, branch, committed)
- **T1** env → **symmetric μ_J=0**, provisional λ=0.05/σ=0.12 (both SimConfig + DEFAULT_SIM_CONFIG); docstring rewritten with pre-registered acceptance criteria.
- **T2** `scan_jump_calibration.py` — scans λ∈{.05,.10}×σ∈{.08,.12,.16}, scores hard criteria (non-degeneracy, dump<50%, TWAP CVaR₉₅∈[4,15]) + soft (differentiation), recommends a cell.
- **T3** `run_tests.py` Section 7 — computed P-band `1-exp(-λN)±0.03` + symmetric assertion (32/32).
- **T4** `sweep_cvar_alpha.py` — select **best-by-val-CVaR₉₅** checkpoint (not newest) + `--ckpt-dir`.
- **T5** RUNBOOK "JD recalibration" section + `run_seeds.py --jump-intensity/--jump-std` overrides.

### Staged full-scale JD retrain executed (seed 42, CPU) — R.4 gate FAILED
- `run_jd_staged.py` (STEP 1): `--only-agent`/`--assemble-eval`/`--eval-agents`; equivalence smoke = byte-identical to monolithic + split. Order-independent RNG reseed in `train_agent`+`evaluate_all`; `--checkpoint-freq` added.
- Trained DQN (best ep8000)/DDQN (ep13000)/IQN-neutral (ep7000), 30k eps, CPU (MPS DQN was reaped at ep16000 → archived `results/_archive_jd_partial_mps_reaped_seed42/`; CPU ~6× faster).
- **Gate FAIL:** DDQN dump-collapse (Std 0.0092, dump 0.999) → fallback trigger; no headline-α differentiation (noise). Fallback (0.05,0.12) recommended, NOT launched.

### JD calibration scan executed (seed 42, MPS)
- Added `--cell λ σ` / `--summarize` / `--out-root` to `scan_jump_calibration.py` so the ~1 h grid runs **one cell per short job** (the harness reaps long background jobs at ~34 min); whole-grid path unchanged. Committed.
- Ran all 6 cells (λ∈{0.05,0.10}×σ∈{0.08,0.12,0.16}, μ=0), ~4–5 min each, all exit 0 → `results/_jump_scan/summary.{txt,csv}`.
- **Hard-pass cells: (0.05,0.08), (0.05,0.12), (0.05,0.16)**; λ=0.10 all fail (dump ≥70% at σ≤0.12; TWAP CVaR₉₅=17.5>15 at σ=0.16). **RECOMMENDED (0.05, 0.16)** (largest gap +0.068). Differentiation (d, soft) holds for (0.05,0.12) and (0.05,0.16). Awaiting user confirmation before R.3.

---

## Not done / open

- **DECISION: run the pre-declared fallback (0.05, 0.12)?** — R.4 failed at (0.05,0.16) (DDQN dump-collapse + no headline differentiation). Fallback rerun is *recommended, not launched* (per FALLBACK RULE). Awaiting user. If run and it still doesn't differentiate at α=0.90/0.95, discuss alternatives (not auto-tune) — see DECISION.md.
- **Batch B / C / D** (multi-seed, sensitivity/misspec, ablations) — **blocked** until the JD table passes non-degeneracy AND differentiation.
- **`PAPER_FIXES.md` number updates** after retrains (JD table, DQN/DDQN rows, abstract percentages) — WILL-CHANGE rows still placeholders. Note: **unified DQN (hidden=64) is worse than the old 128** on TAQ (CVaR₉₅ 19.7 vs 6.30) — feed to the fair-comparison narrative.
- **`main_paper.tex`** — never edited directly; all changes catalogued in PAPER_FIXES.md.
- **N7 (non-split-adjusted AAPL data)** — flagged, not fixed (R-6); test-window results not corrupted, but note for the final data pipeline.

## Operating notes
- **Long training runs must run in the user's own terminal** (harness reaps background jobs at ~34 min). Use `caffeinate -i`. See RUNBOOK for exact commands.
- **Never overwrite `results/`** — archive only. New runs go to new suffixed dirs; `run_seeds` refuses non-empty dirs.
