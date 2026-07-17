# VERIFY.md — acceptance checks for Phase 1 + Phase 2

Branch: `fix/unify-arch-jd-recalibration` (off `main`). Run all commands **from the repo root** with the `finrl_env` python. Nothing here launches a long training run except the clearly-marked smoke runs in §B.

Legend for expected output: ✅ = must see this.

---

## Task coverage (status / files / which check)

| Task | What | Files touched | Verified by |
|------|------|---------------|-------------|
| **P1-T0** | Unblock `envs` imports (drop moved lobster_env) | `envs/__init__.py` | A1 |
| **P1-T1** | Unify DQN/DDQN width 128→64 | `agents/baselines.py` | A2 |
| **P1-T2** | Param-count guard (11,462 / 5,190) | `agents/param_utils.py`, `agents/baselines.py`, `agents/iqn_agents.py`, `experiments/run_simulation.py`, `run_tag.py`, `eval_taq.py`, `eval_simulation.py` | A2 |
| **P1-T3** | Jump-intensity fix + recalibration (stress) | `envs/simulated_env.py`, `experiments/run_simulation.py`, `tests/run_tests.py` | A3 |
| **P1-T4** | Config hygiene (YAML = reference-only) | `configs/sim_config.yaml`, `configs/agent_config.yaml` | A4 |
| **P1-T5** | Stale dim/action comments + tests (5-D / 6 actions) | `envs/base_env.py`, `networks/iqn_networks.py`, `agents/iqn_agents.py`, `tests/run_tests.py` | A5 |
| **P1-T6** | Remove per-env arch branch in eval_simulation | `experiments/eval_simulation.py` | A6 |
| **P1-T7** | `--device {cpu,mps}` flag | `experiments/run_simulation.py`, `run_tag.py` | A7 |
| **N9** | Delete dead half-spread block in taq_env | `envs/taq_env.py` | A8 |
| **P2-T1** | R1 multi-seed orchestration + aggregation | `experiments/run_seeds.py`, `aggregate_seeds.py`, `exp_utils.py`, `tests/test_aggregate.py` | A9, B (run) |
| **P2-T2** | R2 CVaR-α sweep (eval-only) | `experiments/sweep_cvar_alpha.py` | A10, B (run) |
| **P2-T3** | R3 jump sensitivity grid | `experiments/run_jump_sensitivity.py` | A10, B (run) |
| **P2-T4** | R4 impact misspecification (eval-only) | `experiments/run_impact_misspec.py` | A10, B (run) |
| **P2-T5** | R5 width ablation (optional) | `experiments/run_width_ablation.py` | A10, B (run) |
| **P2-T6** | `--smoke` flags + shared config dump | `experiments/run_simulation.py`, `run_tag.py` (+ `exp_utils.py`) | A11, B1–B3 |
| **T1** | JD recalibration → symmetric (μ_J=0), provisional λ=0.05/σ=0.12 | `envs/simulated_env.py`, `experiments/run_simulation.py` | A12 |
| **T2** | Jump-calibration scan + pre-registered criteria | `experiments/scan_jump_calibration.py` | A13 |
| **T3** | Jump-rate test: computed P-band + symmetric assertion | `tests/run_tests.py` | A14 (in A3/A5) |
| **T4** | Sweep selects best-by-val-CVaR checkpoint | `experiments/sweep_cvar_alpha.py`, `RUNBOOK.md` | A15 |
| **T5** | RUNBOOK JD-recalibration section + run_seeds jump overrides | `RUNBOOK.md`, `experiments/run_seeds.py` | inspection + `run_seeds --help` (jump flags) |

---

## A. Fast checks (< 1 min total) — no training

Run them all at once:

```bash
# --- A0: everything compiles ---
python3 -m py_compile \
  envs/__init__.py envs/base_env.py envs/simulated_env.py envs/taq_env.py \
  agents/baselines.py agents/iqn_agents.py agents/param_utils.py networks/iqn_networks.py \
  experiments/run_simulation.py experiments/run_tag.py experiments/eval_simulation.py \
  experiments/eval_taq.py experiments/exp_utils.py experiments/run_seeds.py \
  experiments/aggregate_seeds.py experiments/sweep_cvar_alpha.py \
  experiments/run_jump_sensitivity.py experiments/run_impact_misspec.py \
  experiments/run_width_ablation.py tests/run_tests.py tests/test_aggregate.py \
  && echo "A0 OK: all compile"

# --- A1 (P1-T0): envs imports work ---
python3 -c "from envs import AlmgrenChrissEnv, JumpDiffusionEnv, MeanRevertingEnv, SimConfig; from envs.taq_env import TAQEnv; print('A1 OK')"

# --- A2 (P1-T1/T2): unified param counts + drift guard ---
python3 -c "
import torch
from agents.baselines import DQNAgent, DeepRLConfig
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.param_utils import expected_counts, assert_param_count, mlp_param_count
ei, em = expected_counts(5, 6)
d = DQNAgent(DeepRLConfig(), 5, 6, device=torch.device('cpu'), seed=0)
i = IQNAgent(AgentConfig(1.0), 5, 6, device=torch.device('cpu'), seed=0)
assert_param_count(d, em, 'DQN'); assert_param_count(i, ei, 'IQN')
assert (em, ei) == (5190, 11462)
assert mlp_param_count(5, 6, hidden_dim=128) == 18566  # drift would be caught
print('A2 OK: DQN=5190 IQN=11462 (drift@128=18566)')"

# --- A3 (P1-T3) + A5 (P1-T5): env tests incl jump rate + 5-D/6-action asserts ---
python3 tests/run_tests.py | tail -3

# --- A4 (P1-T4): YAML parses + matches dataclasses ---
python3 -c "
import yaml
from agents.iqn_agents import AgentConfig
from agents.baselines import DeepRLConfig
for f in ['configs/sim_config.yaml','configs/agent_config.yaml']: yaml.safe_load(open(f))
a, r = AgentConfig(), DeepRLConfig()
assert (a.hidden_dim, a.cos_embedding_dim, a.target_update_freq) == (64, 32, 500)
assert r.hidden_dim == 64
print('A4 OK: yaml parses; dataclasses hidden=64 cos=32 target=500')"

# --- A6 (P1-T6): no architecture branch left in eval_simulation ---
( grep -q "hidden_dim=128" experiments/eval_simulation.py && echo "A6 FAIL: branch remains" ) || echo "A6 OK: no 128 branch"

# --- A7 (P1-T7): --device flag present in both runners ---
python3 experiments/run_simulation.py --help 2>/dev/null | grep -q -- "--device {cpu,mps}" && \
python3 experiments/run_tag.py        --help 2>/dev/null | grep -q -- "--device {cpu,mps}" && echo "A7 OK: --device in both"

# --- A8 (N9): dead half-spread block gone; active fill preserved ---
( grep -q "half_spread" envs/taq_env.py && echo "A8 FAIL: half_spread remains" ) || \
  ( grep -q "p_exec = mid - impact" envs/taq_env.py && echo "A8 OK: dead block gone, fill preserved" )

# --- A9 (P2-T1): aggregation unit test ---
python3 tests/test_aggregate.py | tail -1

# --- A10 (P2-T2..T5): every new experiment script imports + argparses ---
for s in sweep_cvar_alpha run_jump_sensitivity run_impact_misspec run_width_ablation run_seeds; do
  python3 experiments/$s.py --help >/dev/null 2>&1 && echo "A10 OK: $s imports" || echo "A10 FAIL: $s"
done

# --- A11 (P2-T6): --smoke present in both base runners ---
python3 experiments/run_simulation.py --help 2>/dev/null | grep -q -- "--smoke" && \
python3 experiments/run_tag.py        --help 2>/dev/null | grep -q -- "--smoke" && echo "A11 OK: --smoke in both"
```

**Expected (✅):**
- `A0 OK: all compile`
- `A1 OK`
- `A2 OK: DQN=5190 IQN=11462 (drift@128=18566)`
- A3/A5 → `Results: 32 passed, 0 failed out of 32 tests` (includes Section 7 jump rate: `E[jumps/episode] ≈ λ_J·N`, computed-band `P(≥1 jump/episode) ≈ 1-exp(-λ_J·N) (±0.03)`, `Jumps are symmetric (jump_mean == 0.0)`, and Section 1 `state shape is (5,)`, `n_actions == 6`)
- `A4 OK: ...`
- `A6 OK: no 128 branch`
- `A7 OK: --device in both`
- `A8 OK: dead block gone, fill preserved`
- A9 → `8/8 checks passed`
- A10 → five `A10 OK: <script> imports`
- `A11 OK: --smoke in both`

---

## JD-recalibration fast checks (T1–T4) — < 30 s total, no training

```bash
# --- A12 (T1): jump defaults symmetric LOCKED (both places) ---
python3 -c "
import sys; sys.path.insert(0,'experiments')
from envs import SimConfig
import run_simulation as RS
c, d = SimConfig(), RS.DEFAULT_SIM_CONFIG
assert (c.jump_intensity, c.jump_mean, c.jump_std) == (0.05, 0.0, 0.16)
assert (d['jump_intensity'], d['jump_mean'], d['jump_std']) == (0.05, 0.0, 0.16)
print('A12 OK (T1): jump defaults LOCKED 0.05/0.0/0.16 in SimConfig + DEFAULT_SIM_CONFIG')" 2>/dev/null

# --- A13 (T2): scan criteria logic + dump-action index ---
python3 -c "
import sys; sys.path.insert(0,'experiments')
import scan_jump_calibration as S
assert S.DUMP_ACTION == 5
def cell(tw, std, dump, nn, cv):
    return {'TWAP':{'CVaR_0.95_bps':tw},
            'DQN':{'std_IS_bps':0.5,'dump_fraction':0.2,'CVaR_0.95_bps':9},
            'IQN-neutral':{'std_IS_bps':std,'dump_fraction':dump,'CVaR_0.95_bps':nn},
            'IQN-CVaR_0.95':{'std_IS_bps':std,'dump_fraction':dump,'CVaR_0.95_bps':cv}}
assert S.score_criteria(cell(2,0,1,2.08,2.08))['hard_pass'] is False   # degenerate
assert S.score_criteria(cell(8,0.9,0.2,6.5,5.9))['hard_pass'] is True  # healthy
print('A13 OK (T2): DUMP_ACTION=5; degenerate cell FAILs, healthy cell PASSes')" 2>/dev/null

# --- A14 (T3): Section 7 computed P-band + symmetric assertion ---
python3 tests/run_tests.py 2>/dev/null | grep -E "1-exp|symmetric"

# --- A15 (T4): sweep picks best-by-val-CVaR ckpt, not newest ---
python3 -c "
import sys, os, tempfile, json; sys.path.insert(0,'experiments')
import sweep_cvar_alpha as S
d = tempfile.mkdtemp(); ck = os.path.join(d,'checkpoints'); lg = os.path.join(d,'logs')
os.makedirs(ck); os.makedirs(lg)
json.dump({'eval_history':[{'episode':1000,'CVaR_0.95_bps':9.0},
                           {'episode':2000,'CVaR_0.95_bps':3.0},
                           {'episode':3000,'CVaR_0.95_bps':7.0}]},
          open(os.path.join(lg,'IQN-neutral_training.json'),'w'))
for ep in (1000,2000,3000): open(os.path.join(ck,f'IQN-neutral_ep{ep}.pt'),'w').close()
assert S.find_best_checkpoint(ck).name == 'IQN-neutral_ep2000.pt'   # min val CVaR, not newest
print('A15 OK (T4): sweep selects best-by-val-CVaR ep2000, not newest ep3000')" 2>/dev/null
```

**Expected (✅):**
- `A12 OK (T1): ...`
- `A13 OK (T2): ...`
- A14 → two ✓ lines: `P(≥1 jump/episode) ≈ 1-exp(-λ_J·N) (±0.03)` and `Jumps are symmetric (jump_mean == 0.0)`
- `A15 OK (T4): ...`

---

## Staged-execution equivalence (run_jd_staged.py) — ~2 min, brief CPU training

The staged (one-agent-per-job) JD path must reproduce the monolithic `run_phase`
comparison table **exactly**. This is the acceptance gate for `run_jd_staged.py`.

```bash
SC=$(mktemp -d); PY=python3
COMMON="--episodes 100 --eval-freq 50 --checkpoint-freq 50 --eval-episodes 200 --seed 42 --device cpu"
# monolithic
$PY experiments/run_simulation.py --env jump $COMMON --results-dir "$SC/mono" >/dev/null 2>&1
# staged: one agent per job, then assemble
for A in DQN DDQN IQN-neutral; do
  $PY experiments/run_jd_staged.py --only-agent $A --out-dir "$SC/staged/jump_diffusion" $COMMON >/dev/null 2>&1
done
$PY experiments/run_jd_staged.py --assemble-eval --out-dir "$SC/staged/jump_diffusion" --device cpu >/dev/null 2>&1
diff "$SC/mono/jump_diffusion/logs/comparison_table.txt" \
     "$SC/staged/jump_diffusion/logs/comparison_table.txt" \
  && echo "STAGED EQUIVALENCE: PASS (tables byte-identical)"
```

**Expected (✅):** `STAGED EQUIVALENCE: PASS`. Runs on **CPU** by design (deterministic;
isolates staging correctness from any MPS non-determinism). The real retrain runs on
MPS — same device for every job (train + assemble), enforced by the run manifest.

---

## B. Smoke runs (~10 min total) — brief training; run after reviewing the diff

These do a little training (a few hundred–thousand gradient steps) to confirm the full pipeline runs end-to-end and prints the unified param counts. All write to the disposable `results/_smoke/` (gitignored). Swap `--device cpu` → `--device mps` to also smoke-test the MPS path.

```bash
# B1 — sim smoke, Almgren-Chriss (~1–2 min)
python3 experiments/run_simulation.py --smoke --env ac --device cpu

# B2 — sim smoke, jump-diffusion (~1–2 min)
python3 experiments/run_simulation.py --smoke --env jump --device cpu

# B3 — TAQ smoke, one fold (~1–3 min; needs data/processed/AAPL_2014.parquet)
python3 experiments/run_tag.py --smoke --device cpu

# B4 — agent unit tests (do a handful of gradient steps; ~1–2 min each)
python3 tests/test_agent.py
python3 tests/test_baseline.py
```

**Expected (✅):**
- B1/B2/B3 each print, at agent-build time:
  `[param-count] DQN : 5190 params (expected 5190)`, `DDQN : 5190`, `IQN-neutral : 11462`, `IQN-CVaR_* : 11462` — **no AssertionError**.
- Runs finish without NaN in the printed IS / loss; a comparison table prints.
- Outputs land under `results/_smoke/…` (and `results/_smoke/taq/AAPL` for B3). **Nothing under the real `results/*` is written.**
- B4: both test scripts end with `... passed, 0 failed`.

**Cleanup:** `rm -rf results/_smoke` (disposable; never commit it).

> Note: the smoke runs use `--episodes 100`, so no meaningful learning happens — they verify **plumbing**, not performance. Real training is Phase 3 (see `RUNBOOK.md`).
