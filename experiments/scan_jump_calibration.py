"""
experiments/scan_jump_calibration.py
------------------------------------
Jump-diffusion calibration scan (Phase-3 helper — TRAINS models; do not run
during code review).

Selects the final (lambda_J, sigma_J) for the SYMMETRIC (mu_J = 0)
jump-diffusion stress env against PRE-REGISTERED acceptance criteria, after the
first symmetric-vs-adverse recalibration. The Merton compound-Poisson model
(per-period rate) is unchanged; only (lambda_J, sigma_J) are scanned.

Grid : lambda_J in {0.05, 0.10} x sigma_J in {0.08, 0.12, 0.16} dollars,
       mu_J = 0, seed 42  (6 cells).
Per cell (smoke-scale, 5,000 training episodes):
  - train DQN and IQN-neutral, share weights to IQN-CVaR_0.95
  - evaluate TWAP, DQN, IQN-neutral, IQN-CVaR_0.95 on 2,000 episodes
  - record + score the acceptance criteria.

Acceptance criteria (HARD = a,b,c ; SOFT = d, report only):
  (a) Non-degeneracy : Std IS > 0.05 bps for EVERY learned agent.
  (b) Dump fraction  : < 50% of IQN-neutral eval episodes take full
                       liquidation (ACTION_FRACS index N_ACTIONS-1) as their
                       FIRST action. (Logged for all learned agents.)
  (c) Meaningful tail: TWAP CVaR_0.95 in [4, 15] bps.
  (d) Differentiation: IQN-CVaR_0.95 CVaR_0.95 <= IQN-neutral CVaR_0.95 (soft).

Output: results/_jump_scan/cell_l<lam>_s<sig>/ (config.json, metrics.json) and
results/_jump_scan/summary.{txt,csv} with one row per cell, PASS/FAIL per
criterion, and a RECOMMENDED cell (all hard criteria pass; tie-break = largest
IQN-neutral - IQN-CVaR CVaR_0.95 gap). Refuses non-empty output dirs.

Runtime ~8-10 min/cell on MPS (~1 h total). Usage:
    python experiments/scan_jump_calibration.py --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from envs import JumpDiffusionEnv, SimConfig
from envs.base_env import N_ACTIONS
from agents.baselines import TWAPAgent, DQNAgent, DeepRLConfig
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.param_utils import expected_counts, assert_param_count
from evaluation.metrics import cvar_alpha
from exp_utils import dump_config_json, refuse_if_nonempty
import run_simulation as RS

LAMBDAS = [0.05, 0.10]
SIGMAS = [0.08, 0.12, 0.16]
DUMP_ACTION = N_ACTIONS - 1          # index of full-liquidation action (=5)
LEARNED = ['DQN', 'IQN-neutral', 'IQN-CVaR_0.95']

# Acceptance thresholds
STD_MIN_BPS = 0.05                    # (a)
DUMP_FRAC_MAX = 0.50                  # (b)
TAIL_LO, TAIL_HI = 4.0, 15.0         # (c) TWAP CVaR_0.95 band (bps)


# ---------------------------------------------------------------------------
# Evaluation that also records the first action (for the dump fraction)
# ---------------------------------------------------------------------------

def eval_agent(agent, env, n_eval, seed):
    env.seed(seed)
    is_bps, first_actions = [], []
    for _ in range(n_eval):
        state = env.reset()
        if hasattr(agent, 'reset') and callable(agent.reset):
            agent.reset()
        first = None
        done, info = False, {}
        while not done:
            a = agent.select_action(state, eval_mode=True)
            if first is None:
                first = a
            state, _, done, info = env.step(a)
        first_actions.append(int(first))
        is_bps.append(float(info['implementation_shortfall']) * 1e4)
    is_bps = np.asarray(is_bps)
    first_actions = np.asarray(first_actions)
    return {
        'mean_IS_bps': float(is_bps.mean()),
        'std_IS_bps': float(is_bps.std(ddof=1)) if len(is_bps) > 1 else 0.0,
        'CVaR_0.95_bps': float(cvar_alpha(is_bps, 0.95)),
        'max_IS_bps': float(is_bps.max()),
        'dump_fraction': float((first_actions == DUMP_ACTION).mean()),
    }


def score_criteria(m):
    a = all(m[n]['std_IS_bps'] > STD_MIN_BPS for n in LEARNED)
    b = m['IQN-neutral']['dump_fraction'] < DUMP_FRAC_MAX
    c = TAIL_LO <= m['TWAP']['CVaR_0.95_bps'] <= TAIL_HI
    d = m['IQN-CVaR_0.95']['CVaR_0.95_bps'] <= m['IQN-neutral']['CVaR_0.95_bps']
    gap = m['IQN-neutral']['CVaR_0.95_bps'] - m['IQN-CVaR_0.95']['CVaR_0.95_bps']
    return {'a_nondegeneracy': bool(a), 'b_dumpfrac': bool(b), 'c_tail': bool(c),
            'd_diff_soft': bool(d), 'hard_pass': bool(a and b and c),
            'cvar_gap': float(gap)}


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------

def run_cell(lam, sig, episodes, n_eval, seed, device_str, out_root, idx, total):
    out = out_root / f'cell_l{lam}_s{sig}'
    refuse_if_nonempty(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*64}\n  CELL {idx}/{total}: lambda={lam} sigma={sig} $  (mu=0)\n{"="*64}')

    cfg_dict = dict(RS.DEFAULT_SIM_CONFIG)
    cfg_dict['jump_intensity'] = lam
    cfg_dict['jump_mean'] = 0.0
    cfg_dict['jump_std'] = sig
    cfg = SimConfig(**cfg_dict)

    train_env = JumpDiffusionEnv(cfg)
    eval_env = JumpDiffusionEnv(cfg)
    sd, na = train_env.state_dim, train_env.n_actions
    device = torch.device(device_str)

    agents = {
        'TWAP': TWAPAgent(cfg),
        'DQN': DQNAgent(DeepRLConfig(), sd, na, device=device, seed=seed),
        'IQN-neutral': IQNAgent(AgentConfig(cvar_alpha=1.0), sd, na, device=device, seed=seed + 3),
        'IQN-CVaR_0.95': IQNAgent(AgentConfig(cvar_alpha=0.95), sd, na, device=device, seed=seed + 3),
    }
    exp_iqn, exp_mlp = expected_counts(sd, na)
    assert_param_count(agents['DQN'], exp_mlp, 'DQN')
    assert_param_count(agents['IQN-neutral'], exp_iqn, 'IQN-neutral')

    # Train the final agents (checkpoint_dir=None -> use the final policy, NOT
    # best-by-val-CVaR; the val-CVaR restore is exactly what biases toward dump).
    for name in ['DQN', 'IQN-neutral']:
        print(f'  training {name} ({episodes} episodes) ...')
        RS.train_agent(agents[name], train_env, episodes, eval_env=eval_env,
                       eval_freq=max(episodes // 2, 1), n_eval=200,
                       checkpoint_dir=None, seed=seed)

    agents['IQN-CVaR_0.95'].online_net.load_state_dict(
        agents['IQN-neutral'].online_net.state_dict())
    agents['IQN-CVaR_0.95'].target_net.load_state_dict(
        agents['IQN-neutral'].target_net.state_dict())

    metrics = {name: eval_agent(agent, eval_env, n_eval, seed + 99_999)
               for name, agent in agents.items()}
    crit = score_criteria(metrics)

    print(f'  TWAP CVaR95={metrics["TWAP"]["CVaR_0.95_bps"]:.3f} | '
          f'IQN-neutral: Std={metrics["IQN-neutral"]["std_IS_bps"]:.3f} '
          f'dump={metrics["IQN-neutral"]["dump_fraction"]:.2f} '
          f'CVaR95={metrics["IQN-neutral"]["CVaR_0.95_bps"]:.3f} | '
          f'gap={crit["cvar_gap"]:+.3f}')
    print(f'  criteria: (a)nondeg={crit["a_nondegeneracy"]} '
          f'(b)dump<50%={crit["b_dumpfrac"]} (c)tail={crit["c_tail"]} '
          f'-> HARD {"PASS" if crit["hard_pass"] else "FAIL"} | (d)diff={crit["d_diff_soft"]}')

    dump_config_json({'kind': 'jump_scan_cell', 'lambda_J': lam, 'sigma_J': sig,
                      'jump_mean': 0.0, 'episodes': episodes, 'n_eval': n_eval,
                      'seed': seed, 'device': device_str}, out / 'config.json')
    with open(out / 'metrics.json', 'w') as f:
        json.dump({'metrics': metrics, 'criteria': crit}, f, indent=2)

    return {'lambda_J': lam, 'sigma_J': sig, 'metrics': metrics, 'criteria': crit}


# ---------------------------------------------------------------------------
# Summary + recommendation
# ---------------------------------------------------------------------------

def write_summary(cells, out_root):
    passing = [c for c in cells if c['criteria']['hard_pass']]
    rec = max(passing, key=lambda c: c['criteria']['cvar_gap']) if passing else None

    csv_rows = []
    for c in cells:
        m, cr = c['metrics'], c['criteria']
        csv_rows.append({
            'lambda_J': c['lambda_J'], 'sigma_J': c['sigma_J'],
            'TWAP_CVaR95': round(m['TWAP']['CVaR_0.95_bps'], 4),
            'DQN_std': round(m['DQN']['std_IS_bps'], 4),
            'DQN_dump': round(m['DQN']['dump_fraction'], 3),
            'IQNn_std': round(m['IQN-neutral']['std_IS_bps'], 4),
            'IQNn_dump': round(m['IQN-neutral']['dump_fraction'], 3),
            'IQNn_CVaR95': round(m['IQN-neutral']['CVaR_0.95_bps'], 4),
            'IQNcvar_std': round(m['IQN-CVaR_0.95']['std_IS_bps'], 4),
            'IQNcvar_dump': round(m['IQN-CVaR_0.95']['dump_fraction'], 3),
            'IQNcvar_CVaR95': round(m['IQN-CVaR_0.95']['CVaR_0.95_bps'], 4),
            'a_nondeg': cr['a_nondegeneracy'], 'b_dump': cr['b_dumpfrac'],
            'c_tail': cr['c_tail'], 'd_diff': cr['d_diff_soft'],
            'hard_pass': cr['hard_pass'], 'cvar_gap': round(cr['cvar_gap'], 4),
            'recommended': bool(rec is not None and rec is c),
        })

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    lines = ['Jump-diffusion calibration scan (mu_J=0, symmetric)',
             'Criteria: (a) Std IS>0.05 bps all learned; (b) IQN-neutral dump<50%;',
             '          (c) TWAP CVaR95 in [4,15] bps; (d, soft) IQN-CVaR CVaR95 <= IQN-neutral.',
             '']
    hdr = (f'{"lam":>5} {"sig":>5} | {"TWAP_CV95":>9} {"IQNn_std":>8} {"IQNn_dump":>9} '
           f'{"IQNn_CV95":>9} {"gap":>7} | {"a":>2} {"b":>2} {"c":>2} {"HARD":>5} {"d":>2}')
    lines += [hdr, '-' * len(hdr)]
    for c in cells:
        m, cr = c['metrics'], c['criteria']
        mark = '  <== RECOMMENDED' if (rec is not None and rec is c) else ''
        lines.append(
            f'{c["lambda_J"]:>5} {c["sigma_J"]:>5} | '
            f'{m["TWAP"]["CVaR_0.95_bps"]:>9.3f} {m["IQN-neutral"]["std_IS_bps"]:>8.3f} '
            f'{m["IQN-neutral"]["dump_fraction"]:>9.2f} {m["IQN-neutral"]["CVaR_0.95_bps"]:>9.3f} '
            f'{cr["cvar_gap"]:>+7.3f} | '
            f'{"Y" if cr["a_nondegeneracy"] else "N":>2} {"Y" if cr["b_dumpfrac"] else "N":>2} '
            f'{"Y" if cr["c_tail"] else "N":>2} {"PASS" if cr["hard_pass"] else "FAIL":>5} '
            f'{"Y" if cr["d_diff_soft"] else "N":>2}{mark}')
    lines.append('')
    if rec is not None:
        lines.append(f'RECOMMENDED: lambda_J={rec["lambda_J"]}, sigma_J={rec["sigma_J"]} '
                     f'(all hard criteria pass; largest CVaR gap = {rec["criteria"]["cvar_gap"]:+.3f} bps)')
    else:
        lines.append('RECOMMENDED: NONE — no cell passed all hard criteria. '
                     'Widen the grid (e.g. larger sigma_J or lambda_J) or relax thresholds.')

    (out_root / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print(f'\nWrote {out_root}/summary.{{txt,csv}}')


def main():
    ap = argparse.ArgumentParser(description='Jump-diffusion calibration scan')
    ap.add_argument('--episodes', type=int, default=5000, help='Training episodes per learned agent')
    ap.add_argument('--eval-episodes', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--lambdas', nargs='+', type=float, default=LAMBDAS)
    ap.add_argument('--sigmas', nargs='+', type=float, default=SIGMAS)
    args = ap.parse_args()

    out_root = PROJECT_ROOT / 'results' / '_jump_scan'
    grid = [(lam, sig) for lam in args.lambdas for sig in args.sigmas]
    cells = []
    for i, (lam, sig) in enumerate(grid, 1):
        cells.append(run_cell(lam, sig, args.episodes, args.eval_episodes,
                              args.seed, args.device, out_root, i, len(grid)))
    write_summary(cells, out_root)


if __name__ == '__main__':
    main()
