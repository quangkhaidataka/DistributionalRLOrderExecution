# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the paper **"Risk-Averse Optimal Execution via Distributional Reinforcement Learning"** (`main_paper.tex`). It trains RL agents to liquidate a large block of shares over a fixed horizon of `N` decision periods while minimising **implementation shortfall (IS)**. The core contribution: using **IQN** (Implicit Quantile Networks, a distributional RL algorithm) to control **tail risk via CVaR** — something scalar methods (DQN/DDQN) cannot do — at *zero extra training cost*.

This is a scripts-and-modules research repo, **not an installable package**. There is no `requirements.txt`, `setup.py`, or `pyproject.toml`. Dependencies live in the conda env `finrl_env` (Python 3.11, `torch==2.2.2`, numpy, pandas, pyyaml, matplotlib). Scripts prepend the project root to `sys.path` themselves, so **run everything from the repo root** with `python3`.

## ⚠️ Known breakage: `envs/__init__.py`

`envs/__init__.py` still does `from envs.lobster_env import ...`, but `lobster_env.py` was moved to `unnecessary_files/`. Because importing *any* `envs` submodule runs the package `__init__`, **every `import envs` / `from envs...` currently raises `ModuleNotFoundError: No module named 'envs.lobster_env'`.** This breaks `experiments/run_simulation.py` and `tests/run_tests.py` at import time.

To fix, edit `envs/__init__.py`: comment out the `lobster_env` import (line 14) and the `LobsterEnv`/`LobsterConfig`/`LOBSTERLoader` entries in `__all__`. (Scripts that import submodules directly — `eval_simulation.py`, `run_tag.py`, `plots/` — also hit this, since the package `__init__` runs regardless.)

## Commands

Run from the repo root. Tests are standalone scripts (no pytest) that print a PASS/FAIL summary and exit non-zero on failure.

```bash
# Tests
python3 tests/run_tests.py       # environment contract, IS math, regimes (needs envs fix above)
python3 tests/test_agent.py      # IQN network + agent (shapes, CVaR≠neutral, save/load, loss decreases)
python3 tests/test_baseline.py   # TWAP, AC, DQN, DDQN, QR-DQN

# Phase 1 — simulated markets (needs envs fix above)
python3 experiments/run_simulation.py --env jump --episodes 30000   # ac | ou | jump | both
python3 experiments/eval_simulation.py --env jump_diffusion         # eval saved checkpoints, no training

# Phase 2 — real NYSE TAQ data (walk-forward)
python3 experiments/run_tag.py --stock AAPL     # note: file is run_tag.py; docstring calls it run_taq
python3 experiments/eval_taq.py --stock AAPL

# Paper figures from saved IS arrays
python3 plots/plot_is_subplots.py --env jump_diffusion
```

Training is CPU-only by default (device is hardcoded to `cpu` in `build_agents`; MPS/CUDA branches are commented out). A harmless `Failed to initialize NumPy: _ARRAY_API not found` warning appears because numpy 2.x is paired with torch 2.2.2 — it does not affect runs.

## Architecture

Pipeline: **Environment** (the trading MDP) → **Agents** (learn to liquidate) → **Training loop** → **Evaluation/metrics** → **Visualisation** (paper figures).

### `envs/` — the execution MDP
- `base_env.py`: `BaseExecutionEnv` (abstract), `EnvConfig`, and `ACTION_FRACS`. State is a **6-D normalised vector**; the discrete action picks a fraction of *remaining* inventory to sell from `ACTION_FRACS` (currently `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`). Reward = per-step IS contribution minus a quadratic trade penalty; the terminal step **forces full liquidation**. IS is reported in **basis points** (`× 1e4`).
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
- **Configs vs. defaults mismatch:** `configs/*.yaml` are richly documented but the scripts mostly construct configs from hardcoded dicts (`DEFAULT_SIM_CONFIG`, `DEFAULT_TRAIN`) and dataclass defaults, *not* from the YAML. Notably `AgentConfig` defaults to a small net (`hidden_dim=64`, `cos_embedding_dim=32`) while the YAML documents 128/64. Treat the **dataclass defaults and script constants** as the real source of truth, and the YAML as reference.

## Outputs and layout

- `results/<env_name>/` (and `results_regime_50000_10000/`) hold `checkpoints/*.pt`, `figures/`, and `logs/` (`is_arrays.pkl`, `all_results.json`, `comparison_table.txt`). All gitignored.
- `data/raw/` and `data/processed/` are gitignored; regenerate via loaders.
- `unnecessary_files/` — archived/unused code (LOBSTER, Coinbase, Binance/Kaggle loaders, walk-forward variants). Gitignored; do not wire it back into the active pipeline without intent.
- `Code_main.ipynb` is a **standalone** Monte-Carlo derivatives-pricing notebook (regime-switching models), separate from the RL training pipeline.
- `project_explain.md` and `structure.txt` are helpful for intent but **partially stale** — they predate the jump-diffusion/TAQ refactor and reference old names (`iqn_agent.py`, `run_lobster.py`, regime-switching as the main experiment). Trust the code over these docs.
