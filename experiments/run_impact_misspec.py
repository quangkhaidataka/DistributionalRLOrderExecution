"""
experiments/run_impact_misspec.py
---------------------------------
R4 — impact misspecification (EVAL-ONLY, no retraining).

Agents are trained under the BASE impact model (eta, gamma). Here they are
evaluated in environments whose temporary/permanent impact coefficients are
each scaled by {0.5, 1.0, 2.0} — a 3x3 grid — to probe robustness to a
misspecified impact model. The agents' weights (and the AC baseline's assumed
model) stay at the base calibration; only the evaluation environment changes.

Writes results/impact_misspec_<env>/:
  - config.json                       : grid spec
  - cell_e<es>_g<gs>/all_results.json : per-cell metrics + config.json
  - matrix.txt / matrix.csv           : per-agent 3x3 CVaR_0.95 (and mean IS)

Usage (Phase-3, after checkpoints exist):
    python experiments/run_impact_misspec.py --env jump
    python experiments/run_impact_misspec.py --env ac --n-eval 5000
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from envs import AlmgrenChrissEnv, MeanRevertingEnv, JumpDiffusionEnv, SimConfig
from agents.baselines import (TWAPAgent, AlmgrenChrissAgent,
                              DQNAgent, DDQNAgent, DeepRLConfig)
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.param_utils import expected_counts, assert_param_count
from exp_utils import dump_config_json
import run_simulation as RS

SCALES = [0.5, 1.0, 2.0]
ENV_CLASSES = {'ac': AlmgrenChrissEnv, 'ou': MeanRevertingEnv, 'jump': JumpDiffusionEnv}
ENV_NAMES = {'ac': 'almgren_chriss', 'ou': 'mean_reverting', 'jump': 'jump_diffusion'}
LEARNED_WITH_CKPT = ['DQN', 'DDQN', 'IQN-neutral']


def _scaled_cfg(eta_scale, gamma_scale):
    d = dict(RS.DEFAULT_SIM_CONFIG)
    d['eta'] = d['eta'] * eta_scale
    d['gamma'] = d['gamma'] * gamma_scale
    return SimConfig(**d)


def _load_ckpt(agent, name, ckpt_dir):
    best = ckpt_dir / f'{name}_best.pt'
    if best.exists():
        agent.load(str(best))
        return True
    eps = sorted(glob.glob(str(ckpt_dir / f'{name}_ep*.pt')))
    if eps:
        agent.load(eps[-1])
        return True
    print(f'  WARNING: no checkpoint for {name} in {ckpt_dir}')
    return False


def build_agents(base_cfg, state_dim, n_actions, seed, device, ckpt_dir):
    agents = {}
    agents['TWAP'] = TWAPAgent(base_cfg)
    agents['AC'] = AlmgrenChrissAgent(base_cfg, risk_aversion=1e-6)
    agents['DQN'] = DQNAgent(DeepRLConfig(), state_dim, n_actions, device=device, seed=seed)
    agents['DDQN'] = DDQNAgent(DeepRLConfig(), state_dim, n_actions, device=device, seed=seed + 1)
    agents['IQN-neutral'] = IQNAgent(AgentConfig(cvar_alpha=1.0), state_dim, n_actions,
                                     device=device, seed=seed + 3)
    agents['IQN-CVaR_0.95'] = IQNAgent(AgentConfig(cvar_alpha=0.95), state_dim, n_actions,
                                       device=device, seed=seed + 3)

    exp_iqn, exp_mlp = expected_counts(state_dim, n_actions)
    for n in ['DQN', 'DDQN']:
        assert_param_count(agents[n], exp_mlp, n)
    for n in ['IQN-neutral', 'IQN-CVaR_0.95']:
        assert_param_count(agents[n], exp_iqn, n)

    for name in LEARNED_WITH_CKPT:
        _load_ckpt(agents[name], name, ckpt_dir)
    # IQN-CVaR shares IQN-neutral weights (D-design)
    agents['IQN-CVaR_0.95'].online_net.load_state_dict(
        agents['IQN-neutral'].online_net.state_dict())
    agents['IQN-CVaR_0.95'].target_net.load_state_dict(
        agents['IQN-neutral'].target_net.state_dict())
    return agents


def main():
    ap = argparse.ArgumentParser(description='R4 impact misspecification (eval-only)')
    ap.add_argument('--env', required=True, choices=list(ENV_CLASSES.keys()))
    ap.add_argument('--ckpt-dir', default=None,
                    help='Checkpoint dir (default results/<env_name>/checkpoints)')
    ap.add_argument('--n-eval', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    env_name = ENV_NAMES[args.env]
    env_cls = ENV_CLASSES[args.env]
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else \
        PROJECT_ROOT / 'results' / env_name / 'checkpoints'

    base_cfg = _scaled_cfg(1.0, 1.0)
    probe = env_cls(base_cfg)
    sd, na = probe.state_dim, probe.n_actions
    device = torch.device('cpu')

    agents = build_agents(base_cfg, sd, na, args.seed, device, ckpt_dir)

    out = PROJECT_ROOT / 'results' / f'impact_misspec_{env_name}'
    out.mkdir(parents=True, exist_ok=True)
    dump_config_json({'kind': 'impact_misspec', 'env': env_name,
                      'eta_scales': SCALES, 'gamma_scales': SCALES,
                      'n_eval': args.n_eval, 'seed': args.seed,
                      'ckpt_dir': str(ckpt_dir)}, out / 'config.json')

    # cell -> {agent: metrics}
    grid = {}
    csv_rows = []
    for es in SCALES:
        for gs in SCALES:
            eval_env = env_cls(_scaled_cfg(es, gs))
            all_results, _ = RS.evaluate_all(agents, eval_env, n_eval=args.n_eval,
                                             seed=args.seed + 99_999)
            cell = out / f'cell_e{es}_g{gs}'
            (cell / 'logs').mkdir(parents=True, exist_ok=True)
            serial = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in r.items()} for r in all_results]
            with open(cell / 'logs' / 'all_results.json', 'w') as f:
                json.dump(serial, f, indent=2)
            dump_config_json({'eta_scale': es, 'gamma_scale': gs,
                              'eta': RS.DEFAULT_SIM_CONFIG['eta'] * es,
                              'gamma': RS.DEFAULT_SIM_CONFIG['gamma'] * gs},
                             cell / 'config.json')
            grid[(es, gs)] = {r['agent_name']: r for r in all_results}
            for r in all_results:
                csv_rows.append({'agent': r['agent_name'], 'eta_scale': es,
                                 'gamma_scale': gs,
                                 'mean_IS_bps': float(r['mean_IS_bps']),
                                 'CVaR_0.95_bps': float(r['CVaR_0.95_bps'])})
            print(f'  cell eta*{es} gamma*{gs} done')

    _write_matrices(out, grid, csv_rows)
    print(f'\nWrote {out}/ (matrix.txt, matrix.csv, {len(SCALES)**2} cells)')


def _write_matrices(out, grid, csv_rows):
    agent_names = list(next(iter(grid.values())).keys())
    lines = ['Impact misspecification — CVaR_0.95 (bps), rows=eta_scale, cols=gamma_scale',
             '(each agent trained at eta_scale=gamma_scale=1.0)', '']
    for a in agent_names:
        lines.append(f'Agent: {a}')
        lines.append('  eta\\gamma ' + ''.join(f'{gs:>10}' for gs in SCALES))
        for es in SCALES:
            cells = ''.join(f'{grid[(es, gs)][a]["CVaR_0.95_bps"]:>10.4f}' for gs in SCALES)
            lines.append(f'  {es:<9}' + cells)
        lines.append('')
    (out / 'matrix.txt').write_text('\n'.join(lines) + '\n')
    with open(out / 'matrix.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['agent', 'eta_scale', 'gamma_scale',
                                          'mean_IS_bps', 'CVaR_0.95_bps'])
        w.writeheader()
        w.writerows(csv_rows)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
