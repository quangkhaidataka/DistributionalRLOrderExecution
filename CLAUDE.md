# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the paper **"Risk-Averse Optimal Execution via Distributional Reinforcement Learning"** (`main_paper.tex`). It trains RL agents to liquidate a large block of shares over a fixed horizon of `N` decision periods while minimising **implementation shortfall (IS)**. The core contribution: using **IQN** (Implicit Quantile Networks, a distributional RL algorithm) to control **tail risk via CVaR** — something scalar methods (DQN/DDQN) cannot do — at *zero extra training cost*.

This is a scripts-and-modules research repo, **not an installable package**. There is no `requirements.txt`, `setup.py`, or `pyproject.toml`. Dependencies live in the conda env `finrl_env` (Python 3.11, `torch==2.2.2`, numpy, pandas, pyyaml, matplotlib). Scripts prepend the project root to `sys.path` themselves, so **run everything from the repo root** with `python3`.

## Project state & living docs (READ FIRST)

Active branch: **`fix/unify-arch-jd-recalibration`** (off `main`). Substantial work has landed here — the unified architecture, the jump-diffusion fixes, the robustness suite, and Batch A results. On this branch **`from envs import ...` works** (the old `lobster_env` import breakage is fixed). To pick up context in a fresh terminal, read these repo-root docs in order:

- **`PROGRESS.md`** — what's done, in flight, and next (the current state of the work).
- **`DECISION.md`** — the key decisions and their rationale (don't re-litigate these).
- **`PLAN.md`** — the full fix/robustness/run plan (Phases 1–3).
- **`PAPER_FIXES.md`** — the `main_paper.tex` audit. **`main_paper.tex` is NEVER edited directly.**
- **`RUNBOOK.md`** — exact terminal commands for the Phase-3 training runs (run by the user, not the agent — see below).
- **`VERIFY.md`** — fast checks + smoke runs to validate any change.

> **MAINTENANCE RULE — keep the living docs current every session.** Whenever you make meaningful progress (finish a task, run a batch, hit a blocker), **update `PROGRESS.md`**. Whenever you or the user make or change a key decision (design, calibration, scope, what to run), **update `DECISION.md`**. Keep both concise, newest-first, with absolute dates. This is a convention you (Claude) follow — not a deterministic hook — so do it proactively; these two files plus the git log are the source of truth a fresh terminal relies on.

> **Execution constraint:** the harness reaps long-running background jobs (~34 min), so **multi-hour training is run by the user in their own terminal** (`caffeinate -i`, commands in RUNBOOK.md), not launched by the agent. **Never modify/delete anything under `results/` except by archiving.**

## Commands

Run from the repo root. Tests are standalone scripts (no pytest) that print a PASS/FAIL summary and exit non-zero on failure.

```bash
# Tests (standalone; run_tests 32/32, test_agent 27/27, test_baseline 76/76)
python3 tests/run_tests.py       # environment contract, IS math, jump-rate, regimes
python3 tests/test_agent.py      # IQN network + agent (shapes, CVaR≠neutral, save/load, loss decreases)
python3 tests/test_baseline.py   # TWAP, AC, DQN, DDQN, QR-DQN

# Phase 1 — simulated markets (add --device mps to use the Apple-Silicon GPU)
python3 experiments/run_simulation.py --env jump --episodes 30000   # ac | ou | jump | both
python3 experiments/eval_simulation.py --env jump_diffusion         # eval saved checkpoints, no training

# Phase 2 — real NYSE TAQ data (walk-forward)
python3 experiments/run_tag.py --stock AAPL     # note: file is run_tag.py; docstring calls it run_taq
python3 experiments/eval_taq.py --stock AAPL

# Paper figures from saved IS arrays
python3 plots/plot_is_subplots.py --env jump_diffusion
```

Device defaults to `cpu`; pass `--device mps` to use the Apple-Silicon GPU (`run_simulation.py`, `run_tag.py`, and the robustness scripts). A harmless `Failed to initialize NumPy: _ARRAY_API not found` banner appears because numpy 2.x is paired with torch 2.2.2; the one place it actually broke (`tensor.numpy()`) is fixed with `.tolist()`. Root alternative: `pip install "numpy<2"`.

## Architecture

Pipeline: **Environment** (the trading MDP) → **Agents** (learn to liquidate) → **Training loop** → **Evaluation/metrics** → **Visualisation** (paper figures).

### `envs/` — the execution MDP
- `base_env.py`: `BaseExecutionEnv` (abstract), `EnvConfig`, and `ACTION_FRACS`. State is a **5-D normalised vector** `[t*, q*, Δp*, spread*, imb*]` (the σ̂ realized-vol feature is disabled); the discrete action space has **6 actions** — a fraction of *remaining* inventory to sell from `ACTION_FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`. Reward = per-step IS contribution minus a quadratic trade penalty; the terminal step **forces full liquidation**. IS is reported in **basis points** (`× 1e4`).
- `simulated_env.py`: `SimConfig` plus four price models — `AlmgrenChrissEnv` (Gaussian, the sanity check), `MeanRevertingEnv` (OU), `JumpDiffusionEnv` (Poisson jumps → fat tails, where IQN-CVaR should win), and `RegimeSwitchingEnv` (HMM volatility). The **current** experiments use AC / mean-reverting / jump-diffusion; `RegimeSwitchingEnv` is exercised only by `tests/run_tests.py`.
- `taq_env.py`: `TAQEnv` / `TAQConfig` — replays real NYSE TAQ data (`data/processed/*.parquet`) with an impact model, one episode per trading window.

### `agents/`
- `baselines.py`: all comparison agents. Rule-based `TWAPAgent` and `AlmgrenChrissAgent` (analytical optimum, pre-computed schedule); learned `DQNAgent`, `DDQNAgent`, `QRDQNAgent`. All learned agents share `_DeepRLBase` + `DeepRLConfig` (same replay buffer, ε-schedule, target-net update) so comparisons are fair.
- `iqn_agents.py`: `IQNAgent` + `AgentConfig` — the paper's contribution. **One class, two policies**: `cvar_alpha=1.0` → IQN-neutral (τ ~ U[0,1]); `cvar_alpha<1.0` → IQN-CVaR_α (τ ~ U[0,α]). Training *always* uses τ ~ U[0,1]; the CVaR policy is obtained purely by restricting τ at **inference** — this is the "zero-cost risk control" claim. `_compute_loss` implements the quantile Huber loss.

### `networks/iqn_networks.py`
`IQNNetwork` = `StateEncoder` (ψ) ⊙ `CosineQuantileEmbedding` (φ) → `OutputMLP` (f), plus `NetworkConfig`. The element-wise `ψ(s) ⊙ φ(τ)` is what lets τ condition the whole computation (vs QR-DQN's fixed quantiles).

### `training/`
`replay_buffer.py` (`ReplayBuffer`, pre-allocated numpy arrays), `scheduler.py` (ε-decay, LR schedules), `trainer.py` (`Trainer`). Note: the experiment scripts define their **own** `train_agent` loop rather than using `Trainer`.

### `evaluation/`
- `metrics.py`: `cvar_alpha` (the headline metric), `var_alpha`, `gain_loss_ratio`, `sortino_ratio`, `mean/std/max_is`, `EpisodeTracker`, `MetricsSuite`, and `format_comparison_table`.
- `visualizer.py`: `Visualizer` — one method per paper figure (IS distributions, mean-CVaR frontier, training curves, inventory trajectories, quantile distribution, heatmap).

## Key design points when editing experiments

- **Weight sharing is the whole point.** `run_phase` trains only IQN-neutral (+ DQN/DDQN), then `share_iqn_weights()` copies IQN-neutral's online/target weights into every `IQN-CVaR_*` agent. Never train the CVaR variants separately — that would break the paper's claim.
- **Checkpoint selection** picks the episode with the lowest validation `CVaR_0.95_bps`, not the last one.
- `QRDQNAgent` is currently commented out in `run_simulation.py`'s `build_agents`.
- **Configs are dataclass-driven (D4).** `configs/*.yaml` are **reference-only, not loaded by any code**; the scripts construct configs from hardcoded dicts (`DEFAULT_SIM_CONFIG`, `DEFAULT_TRAIN`) and dataclass defaults. As of P1-T4 the YAML values match the dataclasses (IQN net 64/32), but the **dataclass defaults and script constants remain the source of truth**. Note: `run_simulation.DEFAULT_SIM_CONFIG` overrides a few `SimConfig` defaults at runtime (e.g. `ou_theta`, the jump params) — change jump calibration in **both** places.

## Outputs and layout

- `results/<env_name>/` (and `results_regime_50000_10000/`) hold `checkpoints/*.pt`, `figures/`, and `logs/` (`is_arrays.pkl`, `all_results.json`, `comparison_table.txt`). All gitignored.
- `data/raw/` and `data/processed/` are gitignored; regenerate via loaders.
- `unnecessary_files/` — archived/unused code (LOBSTER, Coinbase, Binance/Kaggle loaders, walk-forward variants). Gitignored; do not wire it back into the active pipeline without intent.
- `Code_main.ipynb` is a **standalone** Monte-Carlo derivatives-pricing notebook (regime-switching models), separate from the RL training pipeline.
- `project_explain.md` and `structure.txt` are helpful for intent but **partially stale** — they predate the jump-diffusion/TAQ refactor and reference old names (`iqn_agent.py`, `run_lobster.py`, regime-switching as the main experiment). Trust the code over these docs.
