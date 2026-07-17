# PROGRESS.md — DisRL project state

> Living status doc for the DisRL thesis work. **Update this whenever meaningful progress is made** (task finished, batch run, blocker hit). Newest state at the top of each section. Dates are absolute. Companion: **DECISION.md** (why), **PLAN.md** (the plan), **RUNBOOK.md** (run commands), **VERIFY.md** (checks), **PAPER_FIXES.md** (paper audit).

**Last updated:** 2026-07-17
**Active branch:** `fix/unify-arch-jd-recalibration` (off `main`)
**Repo:** `/Users/user/Desktop/DisRL` · conda env `finrl_env` (py3.11, torch 2.2.2, numpy 2.4.6) · MPS available.

---

## Current status: staged full-scale JD retrain DONE → R.4 gate FAILED → fallback (0.05, 0.12) recommended, NOT launched

Locked **(λ_J=0.05, σ_J=0.16)** and ran the 30k-ep JD retrain **staged** (one agent per job) via `experiments/run_jd_staged.py` — validated byte-identical to the monolithic path (equivalence smoke) and to split `--eval-agents`. **Ran on CPU, not MPS:** the MPS DQN job was reaped at ep16000/30000; CPU is ~6× faster here (DQN 164s, DDQN 183s, IQN-neutral 484s — all ≪ 34 min). Order-independent per-agent RNG reseed added to `train_agent`+`evaluate_all` makes staged ≡ monolithic and CPU deterministic.

**R.4 gate = FAIL.** Table at `results/_seeds/seed42/jump_diffusion/logs/comparison_table.txt`; diagnostics at `logs/gate_diagnostics.json`.
- ✅ Meaningful tail (TWAP CVaR₉₅ 12.63 ∈ [4,15]); IQN-neutral cuts tail to 3.46 (−73% vs TWAP); IQN-neutral dump 0.000; DQN/IQN/IQN-CVaR Std > 0.05; param guard 11462/5190.
- ❌ **DDQN degenerate** — dump-at-t0 fraction 0.999 → Std IS 0.0092 < 0.05 (**triggers the FALLBACK RULE**: any learned agent Std < 0.05).
- ❌ **No differentiation at headline α** — IQN-CVaR₀.₉₅ CVaR₉₅ (3.483) not < IQN-neutral (3.461); sign flips across eval seeds (within noise). α-sweep shows a reduction only at aggressive α=0.5 (3.35 vs neutral 3.54).

### ▶ Immediate next step (user decision)
1. **Per the FALLBACK RULE: recommend one rerun at the pre-declared fallback (0.05, 0.12); NOT launched.** Rerun via `run_jd_staged.py --only-agent … --jump-std 0.12 --device cpu` into a fresh archived dir. (My read: the fallback may fix DDQN's luck-of-the-draw collapse but likely NOT the weak headline differentiation, which stems from best-by-val-CVaR selecting an already-tail-aggressive IQN-neutral — see DECISION.md caveat.)
2. Do NOT proceed to Batch B until the JD table passes non-degeneracy AND shows differentiation.

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
