# RUNBOOK.md — Phase 3 execution (training / eval runs)

Run **after** the Phase 1+2 diff is reviewed and `VERIFY.md` §A (and ideally §B smoke) passes. All commands from the repo root, branch `fix/unify-arch-jd-recalibration`, `finrl_env` python.

**Device:** add `--device mps` to every training command to use the Apple-Silicon GPU (default is `cpu`). The benchmarks below assume MPS (DQN 30k eps ≈ 14 min; IQN ≈ 1.7–2× that; eval 10k ≈ 2–15 min/agent). Times are estimates — recalibrate after Batch A.

**Golden rules (R-2 / D2):**
- Never write into an existing non-empty `results/*` dir. All new runs below target **new** dirs (`results/_seeds/…`, `results/taq/AAPL_seed*`, `results/jump_sensitivity_*`, `results/impact_misspec_*`, `results/width_ablation/…`); `run_seeds.py` also hard-refuses non-empty targets.
- **TAQ IQN is never retrained** — only DQN/DDQN are; the kept `results/taq/AAPL` IQN checkpoint is reused.

---

## Step 0 — Archive the old single-run results (recommended, once)

New runs write to new dirs, so this is for clarity, not safety. **Keep `results/taq/` in place** (D2 reuses its IQN checkpoint).

```bash
mkdir -p results/_archive_2026-07-16
mv results/almgren_chriss results/jump_diffusion \
   results/almgren_chriss_old results/jump_diffusion_old \
   results/_archive_2026-07-16/ 2>/dev/null || true
# Leave results/taq/ and results_regime_50000_10000/ untouched.
```

---

## Batch A — MUST (headline, seed 42) · ~4–5 h

Regenerates the three core results tables under the unified architecture + the CVaR frontier figure.

```bash
# A.1 — sim AC + JD, seed 42  (writes results/_seeds/seed42/{almgren_chriss,jump_diffusion}/)
python3 experiments/run_seeds.py --sim --envs ac jump --seeds 42 --device mps

# A.2 — TAQ DQN/DDQN retrain, seed 42 (IQN reused from results/taq/AAPL; writes results/taq/AAPL_seed42/)
python3 experiments/run_seeds.py --taq --seeds 42 --device mps

# A.3 — R2 CVaR-α frontier (eval-only). Pass --ckpt-dir; the sweep auto-selects the
#        best-by-val-CVaR₉₅ checkpoint from the training log (T4 — no fragile `ls -t`,
#        which grabbed a late/degraded epoch in the first Batch A). Prints the ep chosen.
python3 experiments/sweep_cvar_alpha.py --env jump --ckpt-dir results/_seeds/seed42/jump_diffusion/checkpoints
python3 experiments/sweep_cvar_alpha.py --env ac   --ckpt-dir results/_seeds/seed42/almgren_chriss/checkpoints
python3 experiments/sweep_cvar_alpha.py --env taq        # kept results/taq/AAPL IQN (IQN-neutral_best.pt)
```

### ⛔ CHECKPOINT — review before launching Batch B
Inspect `results/_seeds/seed42/jump_diffusion/logs/comparison_table.txt`:
- TWAP/AC mean IS should be modest (no more the old ~2.9 bps drift-inflated values), tails driven by **rare** jumps.
- IQN-CVaR_0.95 should show a lower CVaR₉₅ / Max IS than DDQN & IQN-neutral (the thesis claim).
- Confirm the param-count lines read `IQN … 11462` and `DQN/DDQN … 5190`.
Also confirm `results/taq/AAPL_seed42/logs/all_results.json` looks sane (IQN-CVaR tail ≪ TWAP). If anything looks off, **stop and re-tune** (e.g. jump calibration, R-3) before spending Batch B compute. Then hand the numbers to `PAPER_FIXES.md` (WILL-CHANGE rows).

---

## JD recalibration (between Batch A and Batch B) — required before Batch B

Batch A found the first JD calibration **degenerate**: with adverse jumps (μ_J=−0.30) immediate full liquidation dominated, so every agent collapsed to dump-at-t0 (IQN Std IS = 0.000) and IQN-CVaR ≡ IQN-neutral. The env is now **symmetric (μ_J=0)**; the final (λ_J, σ_J) is chosen by a calibration scan against pre-registered criteria. Do this **before** Batch B so the multi-seed JD is scientifically meaningful.

**R.0 — archive the old degenerate JD run** (frees the seed42 JD slot; keeps it as an "extreme stress" appendix scenario). The seed42 JD slot **must** be free for the re-run (`run_seeds` refuses non-empty dirs):
```bash
cd /Users/user/Desktop/DisRL
mv results/_seeds/seed42/jump_diffusion results/_archive_jd_stress_seed42
# leave results/_seeds/seed42/almgren_chriss and results/taq/AAPL_seed42 in place
```

**R.1 — run the calibration scan** (~1 h on MPS, 6 cells; your terminal, kept awake):
```bash
cd /Users/user/Desktop/DisRL && caffeinate -i \
  /Users/user/miniconda3/envs/finrl_env/bin/python3 experiments/scan_jump_calibration.py --device mps
```
Then read `results/_jump_scan/summary.txt` — PASS/FAIL per criterion per cell and a **RECOMMENDED** cell (all hard criteria a/b/c pass; tie-break = largest IQN-neutral−IQN-CVaR CVaR₉₅ gap). Tell me the recommended (λ_J, σ_J); I'll confirm it against the summary.

**R.2 — LOCKED (2026-07-17):** the scan selected **(λ_J=0.05, σ_J=0.16)**; this is now the permanent default in `envs/simulated_env.py` SimConfig **and** `run_simulation.DEFAULT_SIM_CONFIG` (σ 0.12→0.16, μ_J=0). **Pre-declared fallback: (0.05, 0.12)** — see the FALLBACK RULE in R.4 (invoked only if the full-scale gate fails; not post-hoc tuning).

**R.3 — retrain JD seed 42 STAGED (one agent per sub-34-min job) + rerun the JD sweep** (~1 h total; run each line as its own kept-awake job). The staged path reproduces the monolithic `run_phase` table exactly (validated by `run_jd_staged.py`'s equivalence smoke); weights cross jobs via checkpoints, so the shared dir accumulates all three agents before assembly:
```bash
cd /Users/user/Desktop/DisRL   # PY=/Users/user/miniconda3/envs/finrl_env/bin/python3
# 1) train one agent per job (30k eps each) into results/_seeds/seed42/jump_diffusion/
caffeinate -i $PY experiments/run_jd_staged.py --only-agent DQN         --device mps
caffeinate -i $PY experiments/run_jd_staged.py --only-agent DDQN        --device mps
caffeinate -i $PY experiments/run_jd_staged.py --only-agent IQN-neutral --device mps
# 2) restore best-by-val-CVaR95, share IQN weights, evaluate all 7 (split with --eval-agents if >30 min)
caffeinate -i $PY experiments/run_jd_staged.py --assemble-eval --device mps
# 3) rerun the α-sweep on the best checkpoint (T4)
caffeinate -i $PY experiments/sweep_cvar_alpha.py --env jump --ckpt-dir results/_seeds/seed42/jump_diffusion/checkpoints
echo "=== JD RECALIBRATION DONE ==="
```
Fallback rerun (only if R.4 fails): add `--jump-std 0.12` to each `run_jd_staged.py` line (into a fresh archived dir; μ_J stays 0). Then tell me — I'll produce the updated JD gate report.

**R.4 — new JD gate checklist** (on `results/_seeds/seed42/jump_diffusion/logs/comparison_table.txt`):
- **Non-degeneracy:** Std IS **> 0.05 bps** for DQN, IQN-neutral, IQN-CVaR_0.95 (no Std=0.000 collapse).
- **Dump fraction:** IQN-neutral does **not** dump-at-t0 on ≥50% of episodes (computed at full scale on the retrained IQN-neutral, not the smoke scan).
- **Meaningful tail:** TWAP CVaR₉₅ ∈ [4, 15] bps.
- **Differentiation:** IQN-CVaR_0.95 CVaR₉₅ **<** IQN-neutral CVaR₉₅ — the claim the degenerate run couldn't demonstrate.
- **Param guard:** `IQN … 11462`, `DQN/DDQN … 5190`.
- **Sweep sanity:** α=1.0 point reproduces the IQN-neutral row (the sweep now uses the best checkpoint, T4).

**FALLBACK RULE (pre-registered).** If the full-scale gate fails on any of — **dump fraction > 50%**, OR **any learned agent Std IS < 0.05 bps**, OR **IQN-neutral CVaR₉₅ ≥ TWAP CVaR₉₅** — then **recommend one rerun at the pre-declared fallback (0.05, 0.12)** and STOP (do not launch it; do not re-scan or search for a new cell — that would be post-hoc tuning). Proceed to Batch B only if the JD table passes non-degeneracy **and** shows differentiation.

---

## Batch B — SHOULD (robustness, remaining seeds) · sim only · one night

R1 seed robustness is **simulation-only (AC + JD)**. The TAQ study stays single-seed by design (Decision log 2026-07-17) — there is **no TAQ multi-seed batch**.

### B1 — sim multi-seed · ~11 h
```bash
python3 experiments/run_seeds.py --sim --envs ac jump --seeds 123 7 2024 31 --device mps
python3 experiments/aggregate_seeds.py --env ac      # -> results/_aggregate/ac_seed_summary.{txt,tex}
python3 experiments/aggregate_seeds.py --env jump    # -> results/_aggregate/jump_seed_summary.{txt,tex}
```

> `aggregate_seeds --env {ac,jump}` globs seeds `{42,123,7,2024,31}` (Batch A wrote seed 42, Batch B the rest) and emits mean ± std booktabs tables. **TAQ is intentionally single-seed** — do **not** run `aggregate_seeds --env taq`; report the seed-42 TAQ table as a case study with a limitation statement (PAPER_FIXES §4.1).

---

## Batch C — SHOULD (sensitivity + misspecification) · ~4–6 h

```bash
# C.1 — R3 jump-parameter sensitivity (trains {TWAP,DQN,IQN-neutral,IQN-CVaR_0.95} at 3 intensities)
python3 experiments/run_jump_sensitivity.py --device mps
#        -> results/jump_sensitivity_{low,base,high}/ + results/jump_sensitivity_summary.{txt,csv}

# C.2 — R4 impact misspecification (EVAL-ONLY; uses Batch-A seed-42 checkpoints)
python3 experiments/run_impact_misspec.py --env jump \
    --ckpt-dir results/_seeds/seed42/jump_diffusion/checkpoints
python3 experiments/run_impact_misspec.py --env ac \
    --ckpt-dir results/_seeds/seed42/almgren_chriss/checkpoints
#        -> results/impact_misspec_<env>/matrix.{txt,csv} (per-agent 3x3 eta×gamma)
```

---

## Batch D — OPTIONAL (appendix) · ~2–3 h

```bash
# D.1 — R5 DQN/DDQN width ablation (64 vs 128) on AC + JD
python3 experiments/run_width_ablation.py --device mps
#        -> results/width_ablation/summary.{txt,csv}
```
> The old **D.2 (TAQ IQN 5-seed)** is **removed**: the entire TAQ study is single-seed by design (Decision log 2026-07-17), so there is no asymmetry to symmetrize. Report the seed-42 TAQ table as a case study with a limitation statement (PAPER_FIXES §4.1).

---

## Where outputs land

| Run | Output |
|-----|--------|
| Batch A/B sim | `results/_seeds/seed<seed>/<env_name>/` (checkpoints, logs/all_results.json, comparison_table.txt, figures) |
| Batch A TAQ (seed 42 only) | `results/taq/AAPL_seed42/` (logs/all_results.json, fold1/checkpoints, run_config.json) — single-seed by design |
| Aggregation | `results/_aggregate/<env∈{ac,jump}>_seed_summary.{txt,tex}` (sim only) |
| CVaR sweep | `results/_cvar_sweep/<tag>/` (frontier.png/pdf, sweep.csv, config.json) |
| Jump sensitivity | `results/jump_sensitivity_<level>/` + `results/jump_sensitivity_summary.{txt,csv}` |
| Impact misspec | `results/impact_misspec_<env>/` (matrix.txt/csv, per-cell dirs) |
| Width ablation | `results/width_ablation/` |

Every run also writes a `config.json` / `run_config.json` capturing its full parameters.

---

## After the runs → paper

Feed the regenerated numbers into `PAPER_FIXES.md`: replace the **WILL-CHANGE-AFTER-RETRAIN** placeholders (all `tab:jd_results`; DQN/DDQN rows of `tab:ac_results` / `tab:taq_results`; JD jump-param rows; DQN/DDQN-relative abstract multipliers) with the new values, and re-derive the abstract percentages from the new tables. The **FIX-IN-PAPER** edits (C 100→500, cos n 64→32, per-period jump eq, 11,462-vs-5,190 fair-comparison paragraph, stress-test framing) can be applied independently of the runs.

## Rollback
- Code: `git checkout main` (branch isolates everything).
- Results: originals are untouched under `results/` (+ `results/_archive_2026-07-16/`); delete only the new suffixed dirs (`results/_seeds`, `results/taq/AAPL_seed*`, `results/_cvar_sweep`, `results/jump_sensitivity_*`, `results/impact_misspec_*`, `results/width_ablation`, `results/_smoke`). Paper: never touched.
