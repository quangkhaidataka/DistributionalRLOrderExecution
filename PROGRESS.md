# PROGRESS.md — DisRL project state

> Living status doc for the DisRL thesis work. **Update this whenever meaningful progress is made** (task finished, batch run, blocker hit). Newest state at the top of each section. Dates are absolute. Companion: **DECISION.md** (why), **PLAN.md** (the plan), **RUNBOOK.md** (run commands), **VERIFY.md** (checks), **PAPER_FIXES.md** (paper audit).

**Last updated:** 2026-07-17
**Active branch:** `fix/unify-arch-jd-recalibration` (off `main`)
**Repo:** `/Users/user/Desktop/DisRL` · conda env `finrl_env` (py3.11, torch 2.2.2, numpy 2.4.6) · MPS available.

---

## Current status: paused before the JD calibration scan

Everything up to and including the **JD recalibration code (T1–T5)** is committed and green. The next action is the **user** running the calibration scan in their own terminal (Phase-3 compute is run by the user — the harness kills long background jobs at ~34 min).

### ▶ Immediate next step (user action)
1. Run `experiments/scan_jump_calibration.py` (RUNBOOK "JD recalibration" step R.1) → read `results/_jump_scan/summary.txt` → report the RECOMMENDED (λ_J, σ_J) cell.
2. Then: archive old degenerate JD (R.0), JD retrain seed 42 with the winning calibration (R.3), JD sweep rerun, new JD gate check (R.4).
3. Then Batch B (multi-seed), C (sensitivity/misspec), D (optional).

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

---

## Not done / open

- **Run the JD calibration scan** and pick the final (λ_J, σ_J) — pending (user compute).
- **JD retrain** seed 42 with the final calibration + JD sweep rerun + new gate.
- **Batch B / C / D** (multi-seed, sensitivity/misspec, ablations) — not started.
- **`PAPER_FIXES.md` number updates** after retrains (JD table, DQN/DDQN rows, abstract percentages) — WILL-CHANGE rows still placeholders. Note: **unified DQN (hidden=64) is worse than the old 128** on TAQ (CVaR₉₅ 19.7 vs 6.30) — feed to the fair-comparison narrative.
- **`main_paper.tex`** — never edited directly; all changes catalogued in PAPER_FIXES.md.
- **N7 (non-split-adjusted AAPL data)** — flagged, not fixed (R-6); test-window results not corrupted, but note for the final data pipeline.

## Operating notes
- **Long training runs must run in the user's own terminal** (harness reaps background jobs at ~34 min). Use `caffeinate -i`. See RUNBOOK for exact commands.
- **Never overwrite `results/`** — archive only. New runs go to new suffixed dirs; `run_seeds` refuses non-empty dirs.
