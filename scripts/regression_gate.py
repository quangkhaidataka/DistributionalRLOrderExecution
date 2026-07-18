"""
scripts/regression_gate.py  (Design-v2 B1 — T1 HARD GATE)
---------------------------------------------------------
Prove the B1 refactor did NOT change legacy results.

It re-evaluates a saved legacy IQN-neutral checkpoint on the jump-diffusion sim
env under the EXACT protocol run_jd_staged's assemble-eval used (10k-episode
eval, seed = 42 + 99_999 = 100041) and compares the reproduced metrics against
the stored all_results.json reference numbers.

Because this is a deterministic reproduction (same seed, same checkpoint, same
env config), the numbers must match to floating-point noise (atol = 1e-3 bps).
Any metric outside tolerance means the refactor changed legacy behaviour: the
gate exits non-zero and prints the exact delta so the orchestrator can
investigate. Do NOT hide such a failure by loosening the tolerance.

Run:  python3 scripts/regression_gate.py
      -> prints a metric | stored | reproduced | delta | PASS/FAIL table,
         ends with 'REGRESSION GATE: PASS' and exit 0 (or FAIL + exit 1).

Env note: numpy 2.x + torch 2.2.2 prints a harmless
'Failed to initialize NumPy: _ARRAY_API not found' banner on torch import.
Ignore it. Never call tensor.numpy() in this codebase (use .tolist()).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── sys.path bootstrap (project root + experiments/, like every experiments/*.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))   # run_simulation, exp_utils

from envs import JumpDiffusionEnv, SimConfig
from run_simulation import build_agents, evaluate_all


# ============================================================================
# Fixed paths, reproduction protocol, and stored reference values
# ============================================================================

SEED_DIR   = PROJECT_ROOT / 'results' / '_seeds' / 'seed42' / 'jump_diffusion'
MANIFEST   = SEED_DIR / 'logs' / 'staged_manifest.json'          # sim_config source
CHECKPOINT = SEED_DIR / 'checkpoints' / 'IQN-neutral_ep7000.pt'  # legacy best ckpt
REFERENCE_JSON = SEED_DIR / 'logs' / 'all_results.json'          # provenance only

# Reproduction protocol — matches run_jd_staged assemble-eval (seed = seed+99_999).
SEED      = 42
EVAL_SEED = SEED + 99_999      # 100041
N_EVAL    = 10_000
DEVICE    = 'cpu'
STATE_DIM = 5                  # legacy 5-D normalised state
N_ACTIONS = 6                  # legacy 6-action remaining-inventory grid
ATOL_BPS  = 1e-3               # deterministic repro => float noise only

# Stored reference metrics: the EXACT all_results.json numbers for IQN-neutral.
REFERENCE = {
    'mean_IS_bps'  : 1.8064959831347238,
    'std_IS_bps'   : 0.7865051723405537,
    'CVaR_0.90_bps': 2.6997993082989065,
    'CVaR_0.95_bps': 3.4605254234128333,
    'max_IS_bps'   : 12.143337024087087,
}


def main() -> int:
    print('=' * 78)
    print('  REGRESSION GATE (Design-v2 B1 / T1) — legacy JD IQN-neutral reproduction')
    print('=' * 78)

    # ── 1. Build the JD env from the manifest sim_config (LEGACY 6-action mode).
    manifest = json.loads(MANIFEST.read_text())
    sim_config = manifest['sim_config']
    cfg = SimConfig(**sim_config)                 # no action_basis/action_fracs => legacy
    eval_env = JumpDiffusionEnv(cfg)
    print(f'  manifest    : {MANIFEST}')
    print(f'  checkpoint  : {CHECKPOINT}')
    print(f'  reference   : {REFERENCE_JSON}')
    print(f'  sim_config  : N={cfg.N} q0={cfg.q0} '
          f'jump=(lambda={cfg.jump_intensity}, mu={cfg.jump_mean}, sigma={cfg.jump_std})')
    print(f'  env         : {type(eval_env).__name__} '
          f'state_dim={eval_env.state_dim} n_actions={eval_env.n_actions} device={DEVICE}')

    assert eval_env.state_dim == STATE_DIM and eval_env.n_actions == N_ACTIONS, (
        f'expected legacy {STATE_DIM}-D/{N_ACTIONS}-action env, got '
        f'{eval_env.state_dim}-D/{eval_env.n_actions}-action')

    # ── 2. Build agents; keep ONLY IQN-neutral and load the legacy checkpoint.
    agents = build_agents(cfg, state_dim=STATE_DIM, n_actions=N_ACTIONS,
                          seed=SEED, device_str=DEVICE)
    agent = agents['IQN-neutral']
    agent.load(str(CHECKPOINT))

    # ── 3. Evaluate exactly as run_jd_staged assemble-eval did (seed=seed+99_999).
    print(f'\n  Evaluating IQN-neutral: n_eval={N_EVAL:,}, seed={EVAL_SEED} '
          f'(~1-2 min on {DEVICE})...\n')
    all_results, _ = evaluate_all({'IQN-neutral': agent}, eval_env,
                                  n_eval=N_EVAL, seed=EVAL_SEED)
    repro = all_results[0]

    # ── 4. Compare reproduced vs stored reference.
    print('\n' + '-' * 78)
    hdr = f'{"metric":<15}{"stored":>19}{"reproduced":>19}{"delta":>14}   verdict'
    print(hdr)
    print('-' * len(hdr))
    all_pass = True
    for metric, stored in REFERENCE.items():
        got = float(repro[metric])
        delta = got - stored
        ok = abs(delta) <= ATOL_BPS
        all_pass = all_pass and ok
        print(f'{metric:<15}{stored:>19.12g}{got:>19.12g}{delta:>14.2e}   '
              f'{"PASS" if ok else "FAIL"}')
    print('-' * len(hdr))

    verdict = 'PASS' if all_pass else 'FAIL'
    print(f'\nREGRESSION GATE: {verdict}   '
          f'(atol={ATOL_BPS:g} bps, seed={EVAL_SEED}, n_eval={N_EVAL:,})')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
