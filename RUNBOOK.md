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

## Batch B — SHOULD (robustness, remaining seeds) · split across two nights

### B1 — sim multi-seed · ~11 h
```bash
python3 experiments/run_seeds.py --sim --envs ac jump --seeds 123 7 2024 31 --device mps
python3 experiments/aggregate_seeds.py --env ac      # -> results/_aggregate/ac_seed_summary.{txt,tex}
python3 experiments/aggregate_seeds.py --env jump    # -> results/_aggregate/jump_seed_summary.{txt,tex}
```

### B2 — TAQ multi-seed (DQN/DDQN only) · ~4 h
```bash
python3 experiments/run_seeds.py --taq --seeds 123 7 2024 31 --device mps
python3 experiments/aggregate_seeds.py --env taq     # -> results/_aggregate/taq_seed_summary.{txt,tex}
```

> `aggregate_seeds --env {ac,jump,taq}` globs seeds `{42,123,7,2024,31}` (Batch A wrote seed 42, Batch B the rest) and emits mean ± std booktabs tables. **Asymmetry note (R-4):** TAQ IQN is single-seed (kept), so its ± std is 0 — footnote this in the paper, or run Batch D to symmetrize.

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

## Batch D — OPTIONAL (appendix / symmetry) · ~6–8 h

```bash
# D.1 — R5 DQN/DDQN width ablation (64 vs 128) on AC + JD
python3 experiments/run_width_ablation.py --device mps
#        -> results/width_ablation/summary.{txt,csv}

# D.2 — TAQ IQN 5-seed (removes the R-4 single-seed asymmetry). This retrains IQN too,
#        so write to a SEPARATE dir and do NOT overwrite results/taq/AAPL:
for s in 42 123 7 2024 31; do
  python3 experiments/run_tag.py --episodes 50000 --device mps   # writes results/taq/AAPL (see note)
done
```
> D.2 caveat: `run_tag.py` writes to `results/taq/AAPL` — to keep the kept D2 checkpoints, first move them aside or run each seed with a distinct `STOCK`/`OUT_ROOT`. Simplest: skip D.2 unless the committee demands IQN multi-seed on TAQ, and instead footnote the single-seed IQN.

---

## Where outputs land

| Run | Output |
|-----|--------|
| Batch A/B sim | `results/_seeds/seed<seed>/<env_name>/` (checkpoints, logs/all_results.json, comparison_table.txt, figures) |
| Batch A/B TAQ | `results/taq/AAPL_seed<seed>/` (logs/all_results.json, fold1/checkpoints, run_config.json) |
| Aggregation | `results/_aggregate/<env>_seed_summary.{txt,tex}` |
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
