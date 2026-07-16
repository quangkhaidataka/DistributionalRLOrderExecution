"""
experiments/run_width_ablation.py
---------------------------------
R5 (OPTIONAL, appendix) — DQN/DDQN hidden-width ablation (Phase-3 batch script;
TRAINS models).

Trains DQN and DDQN at hidden_dim in {64, 128} on the AC and jump-diffusion
sim envs (one seed) to show the unified width=64 is not disadvantaging the
scalar baselines. The param-count guard is reused at the ablated width
(mlp_param_count(state_dim, n_actions, hidden_dim=width)).

Writes results/width_ablation/<env>_h<width>/ (config, all_results, table,
checkpoints) and a summary results/width_ablation/summary.{txt,csv}.

Usage:
    python experiments/run_width_ablation.py --episodes 30000 --device mps
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

from envs import AlmgrenChrissEnv, JumpDiffusionEnv, SimConfig
from agents.baselines import DQNAgent, DDQNAgent, DeepRLConfig
from agents.param_utils import mlp_param_count, assert_param_count
from evaluation.metrics import format_comparison_table
from exp_utils import dump_config_json, refuse_if_nonempty
import run_simulation as RS

WIDTHS = [64, 128]
ENVS = {'ac': ('almgren_chriss', AlmgrenChrissEnv),
        'jump': ('jump_diffusion', JumpDiffusionEnv)}


def run_cell(env_key, width, episodes, n_eval, seed, device_str):
    env_name, env_cls = ENVS[env_key]
    out = PROJECT_ROOT / 'results' / 'width_ablation' / f'{env_name}_h{width}'
    refuse_if_nonempty(out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(**RS.DEFAULT_SIM_CONFIG)
    train_env = env_cls(cfg)
    eval_env = env_cls(cfg)
    sd, na = train_env.state_dim, train_env.n_actions
    device = torch.device(device_str)

    print(f'\n=== width_ablation {env_name} hidden={width} ===')
    rl_cfg = DeepRLConfig(hidden_dim=width)
    agents = {
        'DQN': DQNAgent(rl_cfg, sd, na, device=device, seed=seed),
        'DDQN': DDQNAgent(rl_cfg, sd, na, device=device, seed=seed + 1),
    }
    exp = mlp_param_count(sd, na, hidden_dim=width)
    for name, agent in agents.items():
        assert_param_count(agent, exp, f'{name}@{width}')
        RS.train_agent(agent, train_env, episodes, eval_env=eval_env,
                       eval_freq=RS.DEFAULT_TRAIN['eval_freq'], n_eval=200,
                       checkpoint_dir=out / 'checkpoints',
                       checkpoint_freq=RS.DEFAULT_TRAIN['checkpoint_freq'], seed=seed)

    all_results, _ = RS.evaluate_all(agents, eval_env, n_eval=n_eval, seed=seed + 99_999)
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    serial = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in r.items()} for r in all_results]
    with open(logs / 'all_results.json', 'w') as f:
        json.dump(serial, f, indent=2)
    (logs / 'table.txt').write_text(format_comparison_table(all_results, bps=True))
    dump_config_json({'kind': 'width_ablation', 'env': env_name, 'hidden_dim': width,
                      'param_count': exp, 'episodes': episodes, 'seed': seed,
                      'device': device_str}, out / 'config.json')
    return {r['agent_name']: r for r in all_results}


def main():
    ap = argparse.ArgumentParser(description='R5 DQN/DDQN width ablation (optional)')
    ap.add_argument('--episodes', type=int, default=None)
    ap.add_argument('--eval-episodes', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--envs', nargs='+', default=list(ENVS.keys()), choices=list(ENVS.keys()))
    ap.add_argument('--widths', nargs='+', type=int, default=WIDTHS)
    args = ap.parse_args()

    episodes = args.episodes or RS.DEFAULT_TRAIN['n_episodes']
    rows = []
    for env_key in args.envs:
        for width in args.widths:
            res = run_cell(env_key, width, episodes, args.eval_episodes,
                           args.seed, args.device)
            for agent in ['DQN', 'DDQN']:
                r = res.get(agent, {})
                rows.append({'env': ENVS[env_key][0], 'hidden_dim': width, 'agent': agent,
                             'mean_IS_bps': float(r.get('mean_IS_bps', float('nan'))),
                             'CVaR_0.95_bps': float(r.get('CVaR_0.95_bps', float('nan')))})

    out = PROJECT_ROOT / 'results' / 'width_ablation'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['env', 'hidden_dim', 'agent',
                                          'mean_IS_bps', 'CVaR_0.95_bps'])
        w.writeheader()
        w.writerows(rows)
    lines = ['Width ablation — DQN/DDQN mean IS / CVaR_0.95 (bps)', '']
    lines.append(f'{"env":<16s}{"hidden":>8s}{"agent":>8s}{"mean_IS":>12s}{"CVaR95":>12s}')
    for r in rows:
        lines.append(f'{r["env"]:<16s}{r["hidden_dim"]:>8d}{r["agent"]:>8s}'
                     f'{r["mean_IS_bps"]:>12.4f}{r["CVaR_0.95_bps"]:>12.4f}')
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print(f'\nWrote {out}/summary.{{txt,csv}}')


if __name__ == '__main__':
    main()
