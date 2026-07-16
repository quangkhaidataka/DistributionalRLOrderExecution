# PLAN.md — DisRL: Fixes, Robustness, Run Schedule & Paper Sync

> Scope: code-building tasks (Phases 1–2), a compute-scheduled run plan (Phase 3), and a paper-sync audit (Phase 4 → `PAPER_FIXES.md`).
> Constraint honored throughout: **nothing under `results/` is modified or deleted — only archived.** Every new run dumps its full config as JSON.
> Status of this document: all F/D findings re-verified against code with file:line refs below. Cross-checks against result logs live in `PAPER_FIXES.md`.

---

## 1. Executive summary

1. **All four findings F1–F4 are confirmed** (jump-intensity 12× bug, architecture drift across runs, dead YAML configs, stale comments) — plus a **hard blocker (N1): `envs/__init__.py` imports a moved module, so nothing that touches `envs` currently imports.**
2. The unified architecture (D1) is arithmetically verified: **IQN = 11,462 params** (hidden 64 / cos 32 / state_dim **5** / n_actions **6**); **DQN/DDQN = 5,190 params** at hidden 64. Only one code line changes them: `DeepRLConfig.hidden_dim 128→64`.
3. Phase 1 is pure code (fixes + a param-count assertion so drift self-detects). Phase 2 is pure code (robustness suite R1–R5 + cheap unit tests). Neither trains anything.
4. Phase 3 schedules all training/eval into overnight batches with a dependency graph; JD and AC sims and TAQ DQN/DDQN are re-run, **TAQ IQN checkpoints are kept untouched (D2)**.
5. Phase 4 (`PAPER_FIXES.md`) audits every quantitative paper claim; two genuine table mismatches (C, cos-n) are FIX-IN-PAPER, JD/DQN/DDQN/abstract numbers are WILL-CHANGE-AFTER-RETRAIN. **`main_paper.tex` is not edited.**
6. New findings beyond the brief: state is **5-D not 6-D**, action space is **6 not 5** (paper is right, code comments/tests are stale), and **AAPL TAQ prices are non-split-adjusted** — verified; but per-episode arrival-price normalization means the **test results are not corrupted** (residual: a ~7× train-time impact-scale inconsistency, MEDIUM, not results-invalidating).

---

## Phase 0 — Repo verification & smoke tests

### 0.1 Findings verified against code

| ID | Finding | Verdict | Evidence (file:line) |
|----|---------|---------|----------------------|
| **F1** | Jump intensity 12× too high | **CONFIRMED** | `envs/simulated_env.py:211` `n_jumps = self._rng.poisson(self.cfg.jump_intensity * dt)`; `dt = T/N = 60/5 = 12` (`envs/base_env.py:60-62`); `jump_intensity=0.3` (`simulated_env.py:62`). Effective rate = `0.3×12 = 3.6` jumps/**period**, ×5 periods ≈ **18 jumps/episode**. `_evolve_price` is called once per `step` (`base_env.py:174`). Docstring claim "~26% of episodes have at least one jump" (`simulated_env.py:186`) implies ~0.3/episode; actual P(≥1)=`1−e^(−18)`≈**100%**. Mean jump drift = `18 × −0.002 = −0.036` $ on p0=100 ⇒ **≈ −3.6 bps/episode predictable drift**. ✅ Agree with F1. |
| **F2** | Architecture inconsistent across runs | **CONFIRMED** | JD IQN ckpt = 718 KB (hidden 128/cos 64); AC & TAQ IQN ckpt = 207 KB (hidden 64/cos 32) — confirmed by ckpt byte-sizes on disk. Cause: `agents/iqn_agents.py:112-118` — the `128/64` lines are commented out and replaced by `hidden_dim=64, cos_embedding_dim=32`. DQN/DDQN use `DeepRLConfig.hidden_dim=128` everywhere (`agents/baselines.py:390`). `eval_simulation.py:105-108,133-136` even **branches** IQN arch per-env to load the mismatched checkpoints — direct proof of the drift. ✅ Agree with F2. |
| **F3** | YAML configs are dead | **CONFIRMED** | `grep -rn yaml experiments/ agents/ …` returns **zero** loads. All configs built from dataclass defaults + hardcoded constants: `run_simulation.py:257,277,283,700`; `run_tag.py:94,116,125,130,136`; `eval_simulation.py`, `eval_taq.py` similarly. `configs/agent_config.yaml` (hidden 128, target 500, n_episodes 15000…) and `configs/sim_config.yaml` are never read. ✅ Agree with F3. |
| **F4** | Stale / wrong comments | **CONFIRMED** | `configs/sim_config.yaml:28` says "dt = 6 minutes" — actual `dt = 60/5 = 12` (`base_env.py:60`). `sim_config.yaml:43` annualised-vol comment treats σ as relative, but `_evolve_price` adds `σ·√dt` **in dollars** to a $100 price (`simulated_env.py:111`). `agent_config.yaml` documents hidden 128 / target_update 500 / n_episodes 15000 that don't match code. Also `sim_config.yaml` still documents `regime_switching`, which `run_simulation.py` no longer runs. ✅ Agree with F4. |

### 0.2 Smoke-test protocol (validate any code change before long runs)

> **Prerequisite:** the smoke test cannot run until **P1-T0 (fix `envs/__init__.py`)** is applied — currently every `envs` import raises `ModuleNotFoundError: envs.lobster_env`. Apply P1-T0 first; it is the one fix that must precede Phase 0 validation.

Smoke commands (each ≈ 1–2 min, CPU):

```bash
# Env + agent unit tests (fast, no training)
python3 tests/run_tests.py            # env contract, IS math, jump-rate assertion (new)
python3 tests/test_agent.py           # IQN shapes + param-count assertion (new)
python3 tests/test_baseline.py        # DQN/DDQN/QR-DQN + param-count assertion (new)

# 100-episode training smoke per env (asserts it runs end-to-end, prints param counts)
python3 experiments/run_simulation.py --env ac   --episodes 100 --eval-episodes 200 --results-dir results/_smoke
python3 experiments/run_simulation.py --env jump --episodes 100 --eval-episodes 200 --results-dir results/_smoke
python3 experiments/run_tag.py --smoke            # add a --smoke flag: 100 episodes, 1 fold (see P2-T6)
```

**Acceptance:** all three unit-test scripts exit 0; each smoke run prints `IQN params = 11462` and `DQN/DDQN params = 5190`; no NaNs in loss/IS; the smoke `results/_smoke` dir is disposable (gitignored, never committed).

---

## Phase 1 — Fixes (pure code, NO training)

> Branch strategy: do all of Phase 1–2 on a branch `fix/unify-arch-jd-recalibration` off `main`. `main` currently has only `CLAUDE.md` untracked.

### P1-T0 — Unblock `envs` imports *(new finding N1 — prerequisite for everything)*
- **Files/lines:** `envs/__init__.py:14` (`from envs.lobster_env import …`) and the `LobsterEnv`/`LobsterConfig`/`LOBSTERLoader` names in `__all__` (`:30-32`).
- **Change:** comment out the `lobster_env` import line and remove those three names from `__all__`. (Do **not** restore `unnecessary_files/envs/lobster_env.py` into the active tree — the sim/TAQ pipeline does not use it.)
- **Acceptance:** `python3 -c "from envs import AlmgrenChrissEnv, JumpDiffusionEnv, SimConfig; from envs.taq_env import TAQEnv; print('ok')"` succeeds.

### P1-T1 — Unify DQN/DDQN width to 64 (D1)
- **Files/lines:** `agents/baselines.py:390` `hidden_dim: int = 128` → `64`. (`n_hidden_layers=2` unchanged; `_QMLP` already builds `[Linear→LayerNorm→ReLU]×2 → Linear(hidden, n_actions)`, `baselines.py:405-415`.) `AgentConfig` stays as-is (already 64/32, `iqn_agents.py:116-117`).
- **Acceptance:** built DQN/DDQN report **5,190** params; IQN reports **11,462** (see math note below).
- **Param math (verified against checkpoints):** IQN = StateEncoder `Lin(5,64)+LN+Lin(64,64)+LN = 384+128+4160+128 = 4800` + Cosine `Lin(32,64)=2112` + OutputMLP `Lin(64,64)+Lin(64,6) = 4160+390 = 4550` = **11,462**. DQN = `Lin(5,64)+LN + Lin(64,64)+LN + Lin(64,6)` = `384+128 + 4160+128 + 390` = **5,190**. (state_dim=5, n_actions=6 — see N2/N3.)
- **⚠ Implementation note (verified from checkpoints):** count via `sum(p.numel() for p in net.parameters())` — which is what `IQNAgent.get_num_params()` already does (`iqn_agents.py:486`). **Do NOT sum `state_dict()`:** the IQN network registers a non-trainable 32-element cosine-frequency buffer `quantile_embed.i_vals`, so `state_dict()` totals **11,494**, not 11,462 (a real AC/TAQ checkpoint reports 11,494 over state_dict, 11,462 over parameters). DQN/DDQN have no buffers, so both methods give 5,190.

### P1-T2 — Param-count assertion / log at agent build (D1)
- **Files:** add a small helper `assert_param_count(agent, expected)` used in `run_simulation.build_agents` (`run_simulation.py:250-289`), `run_tag.build_all_agents` (`run_tag.py:103-140`), and the eval scripts. `IQNAgent.get_num_params()` already exists (`iqn_agents.py:486`); add the equivalent to `_DeepRLBase`.
- **Change:** after constructing each learned agent, `print`/`log` its param count and `assert` it equals the expected constant (`IQN_PARAMS=11462`, `DQN_PARAMS=5190`), tolerant of state_dim/n_actions differences between sim (5/6) and TAQ (whatever `env.state_dim`/`n_actions` returns — assert against a value **computed from those**, not a hardcoded literal, so TAQ with the same 5/6 also yields 11462/5190).
- **Acceptance:** running any experiment prints the counts; a deliberate `hidden_dim` edit makes the assertion fail (drift self-detects).

### P1-T3 — Fix jump-diffusion env + recalibrate (F1 / D3)
- **Decision (documented, single source of truth):** intensity is defined **per period; remove the `× dt` multiplication.** Rationale: it is the lower-risk, self-documenting option — `E[jumps/episode] = jump_intensity × N` with no unit ambiguity, and it matches the paper table's "per period" label directly (avoids the per-minute reinterpretation). This overrides the Eq. `Poisson(λ·Δt)` form, which will be corrected in the paper (see `PAPER_FIXES.md`).
- **Files/lines:** `envs/simulated_env.py:211` `poisson(self.cfg.jump_intensity * dt)` → `poisson(self.cfg.jump_intensity)`. Update the class docstring (`:183-198`) — replace "~26% of episodes have at least one jump" and the `Poisson(λ·dt)` line with the stress-test framing.
- **Recalibrated defaults (`SimConfig`, `simulated_env.py:62-64`):** `jump_intensity = 0.03` (per period ⇒ E=0.15/episode, P(≥1)=**13.9%**, in the 10–20% target band), `jump_mean = -0.30` (dollars ⇒ −30 bps on p0=100), `jump_std = 0.40` (dollars). Expose all three via config (already dataclass fields — keep them, just change values). Frame in comments as a **stress-test scenario**, citing **Moazeni, Coleman & Li (2013)** and Merton (1976).
- **Acceptance (unit test, cheap):** with `jump_intensity=λ`, over 20k sampled episodes `E[n_jumps/episode] ≈ λ·N` within ±5%; P(≥1 jump/episode) ∈ [0.10, 0.20] for the base config. JD must be **retrained** after this (Phase 3).

### P1-T4 — Config hygiene (D4)
- **Decision (justified):** **make the dataclasses authoritative; rewrite the YAML to match them exactly (option "update yaml+comments").** Rationale: wiring the run scripts to *load* YAML is a behavioral change that risks silently altering every result (the lower-risk path is to keep the code's proven behavior and make the docs honest). Delete the misleading numbers; keep the YAML as reference only, with a header line stating "reference only — not loaded; `AgentConfig`/`DeepRLConfig`/`SimConfig` dataclasses are authoritative."
- **Files:** `configs/agent_config.yaml`, `configs/sim_config.yaml`. Correct every stale value/comment from F4 (hidden 128→64, cos 64→32, target_update to 500, n_episodes to 30000, `dt = 6`→`dt = 12 minutes`, the σ annualisation comment, remove/mark the dead `regime_switching` block). Add the "not loaded" header.
- **Acceptance:** every numeric value in the YAML equals the corresponding dataclass default; no comment contradicts code. (No code path reads these, so zero runtime risk.)

### P1-T5 — Stale code comments & tests (F4 / N2 / N3)
- **Files:** `envs/base_env.py:86,90-91,256` (state "dim = 6" → 5; "N_ACTIONS = 5" / `{0,0.25,…}` → 6 / `{0,0.2,…,1.0}`); `networks/iqn_networks.py:52-57` (docstring "dim=6", `n_actions=5`); `agents/iqn_agents.py:54,145,148` (docstring `n_actions (5)`, examples `state_dim=6, n_actions=5`).
- **Tests:** `tests/run_tests.py:52-53` asserts `state.shape == (6,)` and `n_actions == 5` — both now **false** (actual 5 and 6). Update the assertions to 5 / 6. Verify `tests/test_agent.py` and `tests/test_baseline.py` don't hardcode 6/5 (`grep` for `(6,)`, `== 5`, `state_dim=6`).
- **Acceptance:** `tests/run_tests.py` passes with corrected assertions; no source comment claims dim 6 or 5 actions.

### P1-T6 — Remove F2 workaround in eval script (N5)
- **Files/lines:** `experiments/eval_simulation.py:105-108,133-136` — after JD is retrained at 64/32 (Phase 3), collapse the `if env_name=='jump_diffusion': …128/64` branch to the single default `AgentConfig(cvar_alpha=…)`.
- **Acceptance:** eval loads unified 64/32 checkpoints for all sim envs without branching. **Order:** land the code change but only run it against JD after JD is retrained (else it can't load the old 128/64 JD ckpt — which is fine, those are archived).

---

## Phase 2 — Robustness infrastructure (pure code, NO training)

All scripts **dump full config as JSON** beside their outputs and write to **new** seed/param-suffixed dirs (never touching existing `results/*`).

### P2-T1 — R1 multi-seed orchestration + aggregation
- **New file `experiments/run_seeds.py`:** loops `{ac, jump} × seeds {42,123,7,2024,31}` calling the existing `run_phase`, writing to `results/<env>_seed<seed>/`. Separately loops TAQ DQN/DDQN × the same seeds into `results/taq/AAPL_seed<seed>/` (IQN reused from the kept `results/taq/AAPL`, see D2).
- **New file `experiments/aggregate_seeds.py`:** reads the per-seed `all_results.json`, emits **mean ± std (bps)** per agent×metric as (a) plain text and (b) LaTeX `booktabs` (`\toprule/\midrule/\bottomrule`) ready to paste. Writes `results/_aggregate/<env>_seed_summary.{txt,tex}`.
- **Unit test (cheap):** feed `aggregate_seeds` two synthetic per-seed dicts, assert mean/std arithmetic and LaTeX row count.
- **Acceptance:** dry-run on 2 fake seeds produces a valid booktabs table; each per-seed dir contains a `config.json`.

### P2-T2 — R2 CVaR-α sweep (eval-only, no retraining)
- **New file `experiments/sweep_cvar_alpha.py`:** loads an existing IQN-neutral checkpoint, and for `α ∈ {0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0}` evaluates by setting the eval-time truncation (mutate `agent.cfg.cvar_alpha`, which flows to `tau_high` in `select_action`, `iqn_agents.py:251`) — **weights never change**. Outputs a mean-IS vs CVaR₀.₉₅ frontier plot (publication-quality matplotlib) + a table.
- **Acceptance:** produces one PNG/PDF + CSV per env; α=1.0 point reproduces the IQN-neutral row from the main table (sanity check).

### P2-T3 — R3 jump-parameter sensitivity grid
- **New file `experiments/run_jump_sensitivity.py`:** grid over 3 intensity levels **low `λ=0.01` (P≈5%) / base `λ=0.03` (≈14%) / high `λ=0.06` (≈26%)** × 1 seed, agents = {TWAP, DQN, IQN-neutral, IQN-CVaR_0.95}. Requires P1-T3. Writes `results/jump_sensitivity_<level>/`.
- **Acceptance:** three result dirs, each with config JSON recording the exact λ; a summary comparing CVaR₀.₉₅ across levels.

### P2-T4 — R4 impact misspecification (eval-only)
- **New file `experiments/run_impact_misspec.py`:** load already-trained agents, evaluate in envs where `eta` and `gamma` are each scaled by `{0.5, 1.0, 2.0}` (9-cell matrix). No retraining.
- **Acceptance:** a 3×3 matrix table (train-impact fixed, eval-impact scaled) per agent; config JSON per cell.

### P2-T5 — R5 (OPTIONAL, appendix) DQN/DDQN width ablation
- **New file `experiments/run_width_ablation.py`:** DQN/DDQN at hidden ∈ {64, 128} on AC and JD only, 1 seed. Reuses the param-count assertion.
- **Acceptance:** a small table (width × {DQN,DDQN} × {AC,JD}) of mean IS / CVaR₀.₉₅.

### P2-T6 — Smoke/`--smoke` flags + config-dump helper
- Add a `--smoke` flag (100 eps, tiny eval, `results/_smoke`) to `run_simulation.py` and `run_tag.py`; add a shared `dump_config_json(cfg_dict, path)` used by every runner (satisfies the "all runs dump config" constraint uniformly).
- **Acceptance:** `--smoke` finishes < 3 min/env; every runner writes `config.json`.

---

## Phase 3 — Run schedule (NO code; execute after Phases 1–2 land + smoke passes)

**Timing basis (given):** DQN 30k eps ≈ 14 min (MPS); full TAQ fold ≈ 35 min. Scaled: DDQN ≈ DQN; IQN ≈ 1.7–2× DQN (8 train-τ + 32 policy-τ + cosine embed). Eval of 10k episodes ≈ 2–15 min/agent (IQN slowest via 32 policy-τ). **These are estimates — the Phase 0 smoke run calibrates them; treat batch sizes as provisional.** Device is currently hardcoded to CPU (`build_agents`, `run_simulation.py:241`); enabling MPS (uncomment) roughly matches the benchmark — decide before Batch A.

**Per-unit estimates (30k-eps sim seed = DQN+DDQN+IQN-neutral; CVaR variants share weights, no training):**
- One AC/JD sim seed: train ≈ 14+14+28 = **~56 min** + final eval (7 agents ×10k) ≈ 20–40 min ⇒ **~1.3–1.6 h/seed**.
- One TAQ seed (DQN+DDQN only, 50k eps): ≈ 23+23 = **~46 min** + eval ⇒ **~1 h/seed**.

### Dependency graph
```
P1 fixes + smoke pass ─┬─► [Batch A: MUST] seed-42 headline re-runs
                       │        ├─ AC(42, unified)         ─┐
                       │        ├─ JD(42, fixed+unified)    ─┤─► R2 CVaR sweep (eval-only) ─► frontier fig
                       │        └─ TAQ DQN/DDQN(42, unified)─┘   (IQN-neutral from kept ckpts)
                       │
                       ├─► [Batch B: SHOULD] R1 multi-seed {AC,JD}×{123,7,2024,31} + TAQ DQN/DDQN×4 seeds
                       │        └─► aggregate_seeds → mean±std booktabs tables
                       │
                       ├─► [Batch C: SHOULD] R3 jump sensitivity (needs P1-T3) + R4 impact misspec (eval-only)
                       │
                       └─► [Batch D: OPTIONAL] R5 width ablation ; TAQ IQN multi-seed
```

### Ordered batches (each sized to ~overnight, ~8–10 h)
| Batch | Priority | Contents | Est. wall-clock | Depends on |
|-------|----------|----------|-----------------|-----------|
| **A** | **MUST** | AC(42), JD(42) full re-run + TAQ DQN/DDQN(42) retrain, then R2 CVaR sweep on all three | ~4–5 h | P1 all + smoke |
| **B** | **SHOULD** | R1: AC & JD × seeds {123,7,2024,31} (8 sim runs) + TAQ DQN/DDQN × 4 seeds; then `aggregate_seeds` | ~13–16 h → **split into B1 (sim, ~11 h) + B2 (TAQ, ~4 h)** across two nights | Batch A code paths proven |
| **C** | **SHOULD** | R3 jump sensitivity (3 levels ×1 seed, 4 agents) + R4 impact misspec (eval-only 3×3) | ~4–6 h | P1-T3; trained agents from A/B |
| **D** | **OPTIONAL** | R5 width ablation (AC,JD ×{64,128}) + TAQ IQN 5-seed (removes the IQN single-seed asymmetry, see Risk R-4) | ~6–8 h | A |

### Thesis-critical path (8-week submission)
- **MUST (Batch A):** regenerates the three core results tables under one honest architecture + the CVaR frontier figure. Sufficient to make the paper's central claims defensible.
- **SHOULD (Batches B, C):** mean±std robustness tables and sensitivity/misspecification appendices — expected by a thesis committee.
- **OPTIONAL (Batch D):** ablation appendix and TAQ-IQN multi-seed symmetry — nice-to-have.

---

## Phase 4 — Paper/thesis sync

Full audit is in **`PAPER_FIXES.md`** (every quantitative claim: *paper says / code-or-logs say / verdict*). `main_paper.tex` is **not edited** — `PAPER_FIXES.md` gives precise section/table/line locations for each change. Summary of what it contains:
- **FIX-IN-PAPER (results unchanged):** target-update `C` 100→500 (`tab:sim_params`, L564); cosine embedding `n` 64→32 (L570); the `Poisson(λ·Δt)` jump equation → per-period form; the "fair comparison" paragraph rewritten to state **11,462 vs 5,190** params and cite Dabney et al. (2018) for the minimal quantile machinery; jump env reframed as a stress test; notation-collision cleanup (λ, η, γ reused).
- **WILL-CHANGE-AFTER-RETRAIN (placeholders, no new numbers yet):** all of `tab:jd_results`; every DQN/DDQN cell in `tab:ac_results`/`tab:taq_results`; JD jump-param rows; abstract/intro percentages that depend on DQN/DDQN (5.2×, 7.3×). Note: the TWAP-relative TAQ headlines (88.8% / 97.0%) depend only on kept TWAP & IQN-CVaR rows and are **provisionally stable**.
- **FLAGGED (verify, don't assume):** TAQ param table already matches code (the D-list's "fix TAQ episodes/q0/η/a" items appear **already satisfied**); internal arithmetic slips (L669 10.7/15.1 vs recomputed 10.8/15.3; 49.5 vs 49.6); a **non-split-adjusted AAPL price** smell in `tab:data_summary` (see New findings).

---

## New findings (beyond F1–F4 / D1–D4)

- **N1 (blocker, fixed in P1-T0):** `envs/__init__.py:14` imports `envs.lobster_env`, moved to `unnecessary_files/` — every `envs` import currently fails.
- **N2 (state is 5-D, not 6-D):** `base_env._build_state` returns 5 features (`base_env.py:268-270`); the σ̂ realized-vol feature is commented out (`:239-252,266`). Docstrings (`base_env.py:86,256`; `iqn_networks.py:52`) and `tests/run_tests.py:52` say 6. **The paper is correct (R⁵, tex L274-278); the code comments/tests are stale.** Fixed in P1-T5.
- **N3 (action space is 6, not 5):** `ACTION_FRACS=[0,0.2,0.4,0.6,0.8,1.0]` ⇒ `N_ACTIONS=6` (`base_env.py:72-75`); docstrings say 5 / `{0,0.25,…}`. **Paper is correct (6 actions, tex L137-139).** `tests/run_tests.py:53` asserts `n_actions==5` → currently failing. Fixed in P1-T5.
- **N4 (tests are stale/failing):** beyond N2/N3, `tests/run_tests.py` imports via `from envs import` (blocked by N1) and references `RegimeSwitchingEnv`. All three test scripts need the import fix + assertion updates before they pass.
- **N5 (eval workaround):** `eval_simulation.py` per-env architecture branch — remove after JD retrain (P1-T6).
- **N6 (paper internal inconsistencies):** a commented-out second abstract (tex L76-80) carries conflicting headline numbers; L669 percentages disagree with their own table by rounding. Detailed in `PAPER_FIXES.md`.
- **N7 (data integrity — verified, MEDIUM):** `data/processed/AAPL_2014.parquet` **is confirmed non-split-adjusted**: `mid_price` drops **647.84 (Jun 6) → 93.19 (Jun 9)** at AAPL's 7:1 split. **However, this does *not* corrupt the reported test results:** `TAQEnv.reset` sets the arrival price per-episode from the first bar (`taq_env.py:163`), and IS/state are normalized relative to it (`:200,215,243`); episodes are `N` consecutive intraday bars (`:182`) so none straddle the between-day split, and the test window (Oct–Dec) is entirely post-split (~$93–111). The **real residual issue** is that the temporary-impact term is in **absolute dollars** (`impact = eta*x_t`, `taq_env.py:207`), so its bps cost scales inversely with price level — training on Jan–Jul (mixed ~$93–$650) sees a **~7× inconsistent impact scale**, which can bias the learned policy; the test table itself is internally consistent. **Recommendation:** for the final thesis, either use split-adjusted prices or make impact price-relative (`eta*x_t/mid`); not results-invalidating, so out of current scope — surface to the user, don't silently fix.
- **N9 (dead code in TAQ step):** `taq_env.py:188-202` computes `p_exec` with a half-spread ("Ning et al. approach") but is **immediately overwritten** by `:204-217` (`p_exec = mid - eta*x_t`, no half-spread). The half-spread block is dead; the active fill model omits the spread despite the comment. Cleanup candidate (same stale-code family as F3/F4); note it does **not** change existing results (the second block always wins).
- **N8 (penalty differs by design):** sim `a=1e-3`, TAQ `a=1e-4` — both match their paper tables; noting so it isn't "fixed" by mistake.

---

## Risks & rollbacks

| ID | Risk | Mitigation / rollback |
|----|------|----------------------|
| **R-1** | A Phase-1 fix silently changes sim behavior (e.g., env/reward). | All Phase-1 changes are surgical + guarded by the smoke test (§0.2) and the param assertion (P1-T2). Work on branch `fix/unify-arch-jd-recalibration`; `git diff main` reviewed before any run. |
| **R-2** | Overwriting existing results. | **Never write into an existing `results/*` dir.** Before Batch A, `mv` current dirs to `results/_archive_2026-07-16/` (or copy). New runs use seed/param-suffixed dirs. Enforced by convention + a guard in `run_seeds.py` that refuses to write to a non-empty existing path. |
| **R-3** | JD recalibration numbers land outside the 10–20% target. | P1-T3 unit test gates it; tune `jump_intensity` (0.02–0.05) until P(≥1)∈[0.10,0.20] **before** the expensive JD retrain. |
| **R-4** | TAQ IQN kept single-seed while DQN/DDQN go multi-seed (asymmetric table). | Report IQN-neutral/CVaR as single-seed with an explicit footnote; offer Batch D (OPTIONAL) to retrain TAQ IQN across seeds for symmetry. |
| **R-5** | Timing estimates wrong → batches overrun the night. | Phase 0 smoke calibrates per-agent wall-clock; Batch B is pre-split into B1/B2. Re-estimate after Batch A before committing to B. |
| **R-6** | N7 (non-split-adjusted data) invalidates TAQ results. | Verify data before Batch A eval; if confirmed, this becomes a new Phase-1 data task (out of current scope) — surface to the user, do not auto-fix. |
| **Rollback** | Any run/fix goes wrong. | Code: `git checkout main` (branch isolates everything). Results: originals live untouched in `results/` (+ archive copy); delete only the new suffixed dirs. Paper: never touched. |
