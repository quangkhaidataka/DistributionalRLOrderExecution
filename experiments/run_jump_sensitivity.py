"""
experiments/run_jump_sensitivity.py
-----------------------------------
R3 — jump-parameter sensitivity (Phase-3 batch script — TRAINS models).

Sweeps the jump-diffusion intensity over three levels and compares a compact
agent set {TWAP, DQN, IQN-neutral, IQN-CVaR_0.95} at each. Requires the P1-T3
fix (per-period intensity). One seed.

Levels (per-period Poisson rate; P(>=1 jump/episode) with N=5):
    low  : lambda = 0.01  (~5%)
    base : lambda = 0.03  (~14%, the default stress calibration)
    high : lambda = 0.06  (~26%)

Writes results/jump_sensitivity_<level>/ (config.json, all_results.json,
table.txt, checkpoints/) and a cross-level summary
results/jump_sensitivity_summary.{txt,csv}.

Usage:
    python experiments/run_jump_sensitivity.py --episodes 30000 --device mps
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
from agents.baselines import TWAPAgent, DQNAgent, DeepRLConfig
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.param_utils import expected_counts, assert_param_count
from evaluation.metrics import format_comparison_table
from exp_utils import dump_config_json, refuse_if_nonempty
import run_simulation as RS

LEVELS = [('low', 0.01), ('base', 0.03), ('high', 0.06)]
AGENTS_REPORTED = ['TWAP', 'DQN', 'IQN-neutral', 'IQN-CVaR_0.95']


def build_agents(env_cfg, state_dim, n_actions, seed, device):
    agents = {}
    agents['TWAP'] = TWAPAgent(env_cfg)
    agents['DQN'] = DQNAgent(DeepRLConfig(), state_dim, n_actions, device=device, seed=seed)
    agents['IQN-neutral'] = IQNAgent(AgentConfig(cvar_alpha=1.0), state_dim, n_actions,
                                     device=device, seed=seed + 3)
    agents['IQN-CVaR_0.95'] = IQNAgent(AgentConfig(cvar_alpha=0.95), state_dim, n_actions,
                                       device=device, seed=seed + 3)
    exp_iqn, exp_mlp = expected_counts(state_dim, n_actions)
    assert_param_count(agents['DQN'], exp_mlp, 'DQN')
    assert_param_count(agents['IQN-neutral'], exp_iqn, 'IQN-neutral')
    assert_param_count(agents['IQN-CVaR_0.95'], exp_iqn, 'IQN-CVaR_0.95')
    return agents


def run_level(level, lam, episodes, n_eval, seed, device_str):
    out = PROJECT_ROOT / 'results' / f'jump_sensitivity_{level}'
    refuse_if_nonempty(out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)

    cfg_dict = dict(RS.DEFAULT_SIM_CONFIG)
    cfg_dict['jump_intensity'] = lam
    cfg = SimConfig(**cfg_dict)
    train_env = JumpDiffusionEnv(cfg)
    eval_env = JumpDiffusionEnv(cfg)
    sd, na = train_env.state_dim, train_env.n_actions
    device = torch.device(device_str)

    print(f'\n=== jump_sensitivity level={level} (lambda={lam}) ===')
    agents = build_agents(cfg, sd, na, seed, device)

    for name in ['DQN', 'IQN-neutral']:
        print(f'  training {name} ...')
        RS.train_agent(agents[name], train_env, episodes, eval_env=eval_env,
                       eval_freq=RS.DEFAULT_TRAIN['eval_freq'], n_eval=200,
                       checkpoint_dir=out / 'checkpoints',
                       checkpoint_freq=RS.DEFAULT_TRAIN['checkpoint_freq'], seed=seed)

    # IQN-CVaR shares IQN-neutral weights (zero-cost risk control)
    agents['IQN-CVaR_0.95'].online_net.load_state_dict(
        agents['IQN-neutral'].online_net.state_dict())
    agents['IQN-CVaR_0.95'].target_net.load_state_dict(
        agents['IQN-neutral'].target_net.state_dict())

    all_results, _ = RS.evaluate_all(agents, eval_env, n_eval=n_eval, seed=seed + 99_999)

    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    serial = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in r.items()} for r in all_results]
    with open(logs / 'all_results.json', 'w') as f:
        json.dump(serial, f, indent=2)
    (logs / 'table.txt').write_text(format_comparison_table(all_results, bps=True))
    dump_config_json({'kind': 'jump_sensitivity', 'level': level, 'jump_intensity': lam,
                      'jump_mean': cfg.jump_mean, 'jump_std': cfg.jump_std,
                      'episodes': episodes, 'seed': seed, 'device': device_str},
                     out / 'config.json')
    return {r['agent_name']: r for r in all_results}


def write_summary(by_level):
    """Cross-level CVaR_0.95 / mean_IS comparison for the reported agents."""
    out_txt = PROJECT_ROOT / 'results' / 'jump_sensitivity_summary.txt'
    out_csv = PROJECT_ROOT / 'results' / 'jump_sensitivity_summary.csv'
    levels = [lv for lv, _ in LEVELS if lv in by_level]

    lines = ['Jump-parameter sensitivity — CVaR_0.95 (bps) [mean IS] by level', '']
    header = f'{"Agent":<16s}' + ''.join(f' | {lv:>18s}' for lv in levels)
    lines += [header, '-' * len(header)]
    csv_rows = []
    for a in AGENTS_REPORTED:
        row = f'{a:<16s}'
        for lv in levels:
            r = by_level[lv].get(a, {})
            cv = r.get('CVaR_0.95_bps', float('nan'))
            mn = r.get('mean_IS_bps', float('nan'))
            row += f' | {cv:>8.4f} [{mn:>6.4f}]'
            csv_rows.append({'agent': a, 'level': lv,
                             'CVaR_0.95_bps': cv, 'mean_IS_bps': mn})
        lines.append(row)
    out_txt.write_text('\n'.join(lines) + '\n')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['agent', 'level', 'CVaR_0.95_bps', 'mean_IS_bps'])
        w.writeheader()
        w.writerows(csv_rows)
    print('\n' + '\n'.join(lines))
    print(f'\nWrote {out_txt} and {out_csv}')


def main():
    ap = argparse.ArgumentParser(description='R3 jump-parameter sensitivity')
    ap.add_argument('--episodes', type=int, default=None,
                    help='Training episodes per learned agent (default 30000)')
    ap.add_argument('--eval-episodes', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--levels', nargs='+', default=[lv for lv, _ in LEVELS],
                    choices=[lv for lv, _ in LEVELS])
    args = ap.parse_args()

    episodes = args.episodes or RS.DEFAULT_TRAIN['n_episodes']
    lam_by_level = dict(LEVELS)
    by_level = {}
    for level in args.levels:
        by_level[level] = run_level(level, lam_by_level[level], episodes,
                                    args.eval_episodes, args.seed, args.device)
    write_summary(by_level)


if __name__ == '__main__':
    main()
