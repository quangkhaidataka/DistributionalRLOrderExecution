# DECISION.md — DisRL key decisions

> Durable record of the **decisions** that shape this project and **why** — so a fresh session doesn't re-litigate them. **Update whenever a key decision is made or changed** (design, calibration, scope, what to run). Companion: **PROGRESS.md** (status). Dates absolute.

**Last updated:** 2026-07-17

---

## Architecture & fair comparison

- **D1 — Unified architecture for all experiments.** IQN: hidden=64, cos_embedding=32, 2-layer LayerNorm+ReLU → **11,462 trainable params**. DQN/DDQN: hidden=64, same backbone → **5,190 params**. Only code change: `DeepRLConfig.hidden_dim 128→64`. A build-time **param-count guard** (`agents/param_utils.py`) asserts these so config drift self-detects. *Why:* Batch-A-era runs had drifted (JD IQN was 128/64, AC/TAQ 64/32); one honest architecture makes the comparison defensible.
  - **Consequence:** unified DQN (hidden=64) is materially **worse** than the old hidden=128 (TAQ CVaR₉₅ 19.7 vs 6.30). Accepted — it's the honest number; note it in the paper's fair-comparison narrative.
- **Param count is over `.parameters()` (trainable) = 11,462**, not `state_dict()` (= 11,494, which includes the 32-elem cosine `i_vals` buffer).
- **Weight sharing is the mechanism:** train only IQN-neutral; `share_iqn_weights()` copies its weights into IQN-CVaR variants (CVaR = inference-time τ truncation only). Never train CVaR variants separately.

## Data / experiments scope

- **D2 — Keep the TAQ IQN checkpoint; retrain only DQN/DDQN on TAQ.** The headline TAQ result (IQN-CVaR₀.₉₅: 88.8% CVaR / 97.0% Max reduction vs TWAP) is reproduced from the kept `results/taq/AAPL/fold1/checkpoints/IQN-neutral_best.pt`. *Why:* preserve the validated empirical result; only the scalar baselines need re-running at the unified width. (TAQ IQN is therefore single-seed — footnote it, or Batch D symmetrizes.)
- **N7 — AAPL 2014 data is non-split-adjusted (verified: 647.84→93.19 at the Jun-9 split), NOT fixed.** Per-episode arrival-price normalization means the test-window results are not corrupted; residual is a ~7× train-time impact-scale inconsistency. Out of current scope (R-6) — surface, don't silently fix.

## Jump-diffusion calibration (evolving — read carefully)

- **D3 — Jump intensity is a PER-PERIOD Poisson rate** (removed the buggy `·dt` that inflated it ~12×). The Merton compound-Poisson model itself is unchanged.
- **Recalibration #1 (adverse) → REJECTED as degenerate.** μ_J=-0.30, σ_J=0.40 gave a strong negative drift (N·λ·μ_J≈-4.5 bps) on top of the tail, so immediate full liquidation (certain ~2.08 bps) dominated and every agent collapsed to dump-at-t0 (IQN Std IS=0.000, IQN-CVaR ≡ IQN-neutral). Batch A exposed this.
- **Recalibration #2 (CURRENT): symmetric, scan-selected.**
  - **μ_J = 0 (symmetric)** — permanently removes the drift confound so the only reason to trade faster is tail risk (the CVaR story).
  - **Final (λ_J, σ_J) chosen by `scan_jump_calibration.py`** over λ∈{0.05,0.10} × σ_J∈{0.08,0.12,0.16}. Provisional default until the scan decides: **λ=0.05, σ=0.12**.
  - **Pre-registered acceptance criteria** (hard = a,b,c; soft = d): (a) Std IS > 0.05 bps for every learned agent; (b) IQN-neutral first-action dump fraction < 50%; (c) TWAP CVaR₉₅ ∈ [4,15] bps; (d) IQN-CVaR₀.₉₅ CVaR₉₅ ≤ IQN-neutral. Recommended cell = all hard pass, tie-break = largest neutral−CVaR CVaR gap.
- **Old degenerate JD run is retained** at `results/_archive_jd_stress_seed42/` as a potential "extreme stress" appendix.

## Checkpoint selection

- **Best-by-validation-CVaR₉₅** is the model-selection criterion (matches `run_phase`'s restore). `sweep_cvar_alpha.py` (T4) now reads the training log to load that epoch, **not the newest** — the newest was a degraded checkpoint in Batch A (JD ep30000 regressed to val CVaR₉₅ 17.6 vs best ep14000's 2.08).
- Caveat noted: best-by-CVaR structurally favors dump policies when dumping is tail-optimal — mitigated by the symmetric (μ_J=0) calibration.

## Config & process conventions

- **D4 — Dataclasses/script constants are authoritative; `configs/*.yaml` are reference-only** (not loaded by any code). Lower risk than wiring YAML in.
- **`main_paper.tex` is NEVER edited directly.** All required changes are catalogued in `PAPER_FIXES.md` (FIX-IN-PAPER vs WILL-CHANGE-AFTER-RETRAIN).
- **Never modify/delete `results/` except by archiving.** New runs → new suffixed dirs; `run_seeds` refuses non-empty output dirs.
- **All new runs dump full config as JSON** alongside outputs.
- **numpy 2.x + torch 2.2.2 ABI break** (`tensor.numpy()` fails): fixed in-code with `.tolist()` at the 3 call sites rather than downgrading the shared env. Root alternative if preferred: `pip install "numpy<2"`.
- **Long runs execute in the user's own terminal** (harness kills tracked background jobs at ~34 min); Claude does not launch multi-hour training. Use `caffeinate -i`.
- **One commit per task**, message prefixed with the task ID; work on `fix/unify-arch-jd-recalibration`.
