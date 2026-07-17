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
  - **Final (λ_J, σ_J) chosen by `scan_jump_calibration.py`** over λ∈{0.05,0.10} × σ_J∈{0.08,0.12,0.16}. Provisional default in code: λ=0.05, σ=0.12.
  - **Pre-registered acceptance criteria** (hard = a,b,c; soft = d): (a) Std IS > 0.05 bps for every learned agent; (b) IQN-neutral first-action dump fraction < 50%; (c) TWAP CVaR₉₅ ∈ [4,15] bps; (d) IQN-CVaR₀.₉₅ CVaR₉₅ ≤ IQN-neutral. Recommended cell = all hard pass, tie-break = largest neutral−CVaR CVaR gap.
  - **Scan run 2026-07-17 (seed 42, MPS, 5,000-ep smoke scale) → recommends (λ_J=0.05, σ_J=0.16)** (`results/_jump_scan/summary.{txt,csv}`). Hard-pass cells were all at λ=0.05 ((0.05,0.08),(0.05,0.12),(0.05,0.16)); λ=0.10 fails — dump ≥70% at σ≤0.12 and TWAP CVaR₉₅=17.5>15 at σ=0.16. Tie-break picked σ=0.16 (gap +0.068) over σ=0.12 (+0.044); (0.05,0.08) had a negative gap.
  - **LOCKED 2026-07-17 (user-confirmed): (λ_J=0.05, σ_J=0.16) is now the permanent default** in `envs/simulated_env.py` SimConfig **and** `run_simulation.DEFAULT_SIM_CONFIG` (one line each, σ 0.12→0.16). μ_J stays 0.
  - **Pre-declared FALLBACK: (λ_J=0.05, σ_J=0.12).** Declared **before** seeing any full-scale (30k-ep) results — this is **pre-registration, not post-hoc tuning**. It is only invoked if the full-scale retrain **fails the R.4 gate** (dump fraction > 50% OR any learned agent Std IS < 0.05 bps OR IQN-neutral CVaR₉₅ ≥ TWAP CVaR₉₅). Run it via `--jump-std 0.12` (no code edit); do not re-scan or hunt for a "better" cell post-hoc.
  - **Scan runnable one-cell-per-job:** `scan_jump_calibration.py --cell λ σ` runs a single cell (~4–5 min on MPS), `--summarize` assembles the table from per-cell `metrics.json`. *Why:* the 6-cell grid is ~1 h in one process, over the harness's ~34-min background-job reap limit.

## Reproducibility / order-independent RNG (enables staged execution)

- **`train_agent` and `evaluate_all` (run_simulation.py) reseed the GLOBAL torch+np RNG per agent** — training keyed by `seed + {DQN:0,DDQN:1,IQN-neutral:3}`, eval by the eval `seed`. *Why:* agents consume the global RNG (ε-greedy, replay sampling, IQN τ via `torch.rand`); previously each agent inherited RNG state advanced by the *prior* agent trained/evaluated in the same process, so results were execution-order-dependent and a staged (one-agent-per-process) run could not reproduce a monolithic run. Per-agent reseeding makes each agent's train+eval depend only on its own seed → order-independent → **staged ≡ monolithic** (validated by the equivalence smoke). The env RNG was already isolated (`env.seed()` resets its own `default_rng`). This changes JD/AC numbers vs. the pre-fix behavior, but JD is a fresh retrain and AC-seed42 is already archived; all future multi-seed runs use this discipline consistently.
- **Staged JD retrain** (`experiments/run_jd_staged.py`): `--only-agent {DQN,DDQN,IQN-neutral}` trains one agent per job into the shared `results/_seeds/seed42/jump_diffusion/`; `--assemble-eval [--eval-agents …]` restores each agent's best-by-val-CVaR₉₅ checkpoint, shares IQN-neutral weights, evaluates (all 7, splittable), and writes `comparison_table.txt`/`all_results.json` identical in format to `run_phase`. *Why:* the 30k-ep monolithic JD run exceeds the ~34-min background reap; staging splits it into sub-34-min jobs. Weights cross process boundaries only via checkpoints, so `--checkpoint-freq` must divide the episode count (smoke uses 50).
- **JD staged retrain runs on CPU, not MPS (2026-07-17).** The MPS DQN job was reaped at ep16000/30000; empirically CPU is **~6× faster** for these tiny nets (5,190 / 11,462 params — MPS kernel-dispatch overhead dominates): DQN 30k ≈ 164 s, IQN ≈ 484 s vs MPS-projected 12 / 21 min. CPU is also deterministic (the equivalence smoke passed on CPU). All staged jobs for one dir must share the device (manifest-enforced). *Consequence:* JD-seed42 numbers are CPU; if Batch B mixes devices, footnote it or standardize on CPU for sim.

## JD full-scale gate outcome (2026-07-17) — FAILED → fallback recommended

- First 30k-ep retrain at the locked **(0.05, 0.16)** **failed R.4** on two counts: (1) **DDQN collapsed to dump-at-t0** (dump fraction 0.999 → Std IS 0.0092 < 0.05) — triggers the pre-registered FALLBACK RULE; (2) **no differentiation at headline α** — IQN-CVaR₀.₉₅ CVaR₉₅ (3.483) is not below IQN-neutral (3.461), and the sign flips across eval seeds (within noise); the α-sweep shows a CVaR reduction only at aggressive α=0.5 (3.35 vs neutral 3.54).
- The calibration itself is otherwise healthy: TWAP CVaR₉₅ 12.63 ∈ [4,15]; **IQN-neutral cuts the tail 73%** (3.46 vs 12.63) without dumping (dump 0.000); DQN/IQN Std > 0.05.
- **Per the FALLBACK RULE, recommend ONE rerun at the pre-declared fallback (0.05, 0.12) — NOT launched, reported, stopped.** No re-scan / no new-cell search (that would be post-hoc tuning). *Assessment:* the fallback may resolve DDQN's stochastic collapse but likely not the weak differentiation, which is driven by best-by-val-CVaR selecting an already-tail-aggressive IQN-neutral (little room for τ-truncation to improve) — a structural caveat, not a σ-value issue. If the fallback also fails to differentiate at α=0.90/0.95, that's a finding to discuss (the JD stress may need a different lever, e.g. checkpoint-selection by mean-IS, or reporting the α=0.5 frontier point), to be decided with the user — not auto-tuned.
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
