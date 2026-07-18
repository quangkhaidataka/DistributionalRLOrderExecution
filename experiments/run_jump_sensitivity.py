"""
experiments/run_jump_sensitivity.py
-----------------------------------
R3 — jump-parameter sensitivity (Phase-3 batch script — TRAINS models).

Sweeps the jump-diffusion intensity over three levels and compares a compact
agent set {TWAP, DQN, IQN-neutral, IQN-CVaR_0.95} at each. Requires the P1-T3
fix (per-period intensity). One seed.

Levels (per-period Poisson rate; P(>=1 jump/episode) with N=5):
    Re-centred on the LOCKED calibration (λ_J=0.05, σ_J=0.16, μ_J=0):
    low  : lambda = 0.025  (P(≥1 jump/ep) ≈ 12%)
    base : lambda = 0.05   (≈ 22%, the locked calibration)
    high : lambda = 0.10   (≈ 39%)
    σ_J is held at the locked 0.16 (from DEFAULT_SIM_CONFIG); only λ_J varies.

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
from envs.base_env import N_ACTIONS
from agents.baselines import TWAPAgent, DQNAgent, DDQNAgent, DeepRLConfig
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.param_utils import expected_counts, assert_param_count
from evaluation.metrics import format_comparison_table
from exp_utils import dump_config_json, refuse_if_nonempty
import run_simulation as RS

LEVELS = [('low', 0.025), ('base', 0.05), ('high', 0.10)]   # re-centred on locked λ_J=0.05
AGENTS_REPORTED = ['TWAP', 'DQN', 'DDQN', 'IQN-neutral', 'IQN-CVaR_0.95']  # DDQN added (E2)


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


def run_ddqn_level(level, lam, episodes, n_eval, seed, device_str):
    """E2: train DDQN at one λ level into a NEW subdir (does not touch the
    existing 4-agent level dir), eval it, and record its dump fraction."""
    out = PROJECT_ROOT / 'results' / f'jump_sensitivity_{level}' / 'ddqn'
    refuse_if_nonempty(out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)

    cfg_dict = dict(RS.DEFAULT_SIM_CONFIG)
    cfg_dict['jump_intensity'] = lam
    cfg = SimConfig(**cfg_dict)
    train_env, eval_env = JumpDiffusionEnv(cfg), JumpDiffusionEnv(cfg)
    sd, na = train_env.state_dim, train_env.n_actions
    device = torch.device(device_str)

    print(f'\n=== jump_sensitivity DDQN level={level} (lambda={lam}, sigma={cfg.jump_std}) ===')
    ddqn = DDQNAgent(DeepRLConfig(), sd, na, device=device, seed=seed + 1)
    assert_param_count(ddqn, expected_counts(sd, na)[1], 'DDQN')

    RS.train_agent(ddqn, train_env, episodes, eval_env=eval_env,
                   eval_freq=RS.DEFAULT_TRAIN['eval_freq'], n_eval=200,
                   checkpoint_dir=out / 'checkpoints',
                   checkpoint_freq=RS.DEFAULT_TRAIN['checkpoint_freq'], seed=seed)

    all_results, trackers = RS.evaluate_all({'DDQN': ddqn}, eval_env,
                                            n_eval=n_eval, seed=seed + 99_999)
    result = all_results[0]

    # dump fraction: fraction of episodes whose FIRST action is full liquidation.
    eval_env.seed(seed + 99_999)
    RS._seed_global_rng(seed + 99_999)
    firsts = []
    for _ in range(n_eval):
        s = eval_env.reset()
        a = ddqn.select_action(s, eval_mode=True)
        firsts.append(int(a))
        done = False
        while not done:
            s, _, done, _ = eval_env.step(a)
            if not done:
                a = ddqn.select_action(s, eval_mode=True)
    dump = float((np.array(firsts) == (N_ACTIONS - 1)).mean())

    serial = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
              for k, v in result.items()}
    with open(out / 'ddqn_result.json', 'w') as f:
        json.dump({'result': serial, 'dump_fraction': dump}, f, indent=2)
    dump_config_json({'kind': 'jump_sensitivity_ddqn', 'level': level,
                      'jump_intensity': lam, 'jump_std': cfg.jump_std,
                      'jump_mean': cfg.jump_mean, 'episodes': episodes,
                      'seed': seed, 'device': device_str, 'dump_fraction': dump},
                     out / 'config.json')
    print(f'  DDQN level={level}: Std={result["std_IS_bps"]:.4f} '
          f'CVaR95={result["CVaR_0.95_bps"]:.4f} dump={dump:.3f}')
    return {'DDQN': serial, '_ddqn_dump': dump}


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

    # E2 — DDQN collapse check (Std IS + first-action dump fraction) per level.
    if any('DDQN' in by_level[lv] for lv in levels):
        lines += ['', 'DDQN collapse check — Std IS (bps) [dump fraction] by level:', '']
        drow = f'{"DDQN":<16s}'
        for lv in levels:
            r = by_level[lv].get('DDQN', {})
            std = r.get('std_IS_bps', float('nan'))
            dump = r.get('_dump_fraction', float('nan'))
            drow += f' | {std:>8.4f} [{dump:>5.3f}]'
        lines.append(drow)

    out_txt.write_text('\n'.join(lines) + '\n')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['agent', 'level', 'CVaR_0.95_bps', 'mean_IS_bps'])
        w.writeheader()
        w.writerows(csv_rows)
    print('\n' + '\n'.join(lines))
    print(f'\nWrote {out_txt} and {out_csv}')


def load_level(level):
    """Reconstruct one level's {agent: result} from all_results.json, merging the
    E2 DDQN result (ddqn/ddqn_result.json) if present."""
    base = PROJECT_ROOT / 'results' / f'jump_sensitivity_{level}'
    d = {r['agent_name']: r for r in json.load(open(base / 'logs' / 'all_results.json'))}
    ddqn_p = base / 'ddqn' / 'ddqn_result.json'
    if ddqn_p.exists():
        payload = json.load(open(ddqn_p))
        r = payload['result']
        r.setdefault('agent_name', 'DDQN')
        r['_dump_fraction'] = payload.get('dump_fraction')
        d['DDQN'] = r
    return d


def main():
    ap = argparse.ArgumentParser(description='R3 jump-parameter sensitivity')
    ap.add_argument('--episodes', type=int, default=None,
                    help='Training episodes per learned agent (default 30000)')
    ap.add_argument('--eval-episodes', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--levels', nargs='+', default=[lv for lv, _ in LEVELS],
                    choices=[lv for lv, _ in LEVELS])
    ap.add_argument('--summarize', action='store_true',
                    help='Build the cross-level summary from existing level dirs '
                         '(no training). Use after staging one level per job.')
    ap.add_argument('--ddqn-level', nargs='+', default=None,
                    choices=[lv for lv, _ in LEVELS],
                    help='E2: train DDQN at these level(s) into results/'
                         'jump_sensitivity_<level>/ddqn/ (new subdir; no overwrite).')
    args = ap.parse_args()

    all_levels = [lv for lv, _ in LEVELS]
    lam_by_level = dict(LEVELS)
    episodes = args.episodes or RS.DEFAULT_TRAIN['n_episodes']

    if args.summarize:
        write_summary({lv: load_level(lv) for lv in args.levels})
        return

    if args.ddqn_level:
        for level in args.ddqn_level:
            run_ddqn_level(level, lam_by_level[level], episodes,
                           args.eval_episodes, args.seed, args.device)
        print(f'Trained DDQN for {args.ddqn_level}. Run --summarize to fold into the table.')
        return

    by_level = {}
    for level in args.levels:
        by_level[level] = run_level(level, lam_by_level[level], episodes,
                                    args.eval_episodes, args.seed, args.device)
    if set(args.levels) == set(all_levels):
        write_summary(by_level)
    else:
        print(f'Ran levels {args.levels}. Run --summarize after all levels for '
              f'the cross-level table.')


if __name__ == '__main__':
    main()
