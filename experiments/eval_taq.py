"""
experiments/eval_taq.py
------------------------
Evaluate existing TAQ checkpoints on test set without retraining.

Usage:
    python experiments/eval_taq.py --stock AAPL
    python experiments/eval_taq.py --stock AAPL --n-eval 5000
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.taq_env import TAQEnv, TAQConfig
from envs.base_env import EnvConfig
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.baselines import (
    TWAPAgent, AlmgrenChrissAgent,
    DQNAgent, DDQNAgent, DeepRLConfig,
)
from evaluation.metrics import EpisodeTracker, format_comparison_table


# ============================================================================
# CONFIG — must match run_taq.py
# ============================================================================
N        = 5
T        = 60.0
Q0       = 5000
A        = 0.0001
ETA      = 1e-5
GAMMA_I  = 2.5e-7
SIGMA    = 0.001
DISCOUNT = 0.99
SEED     = 42


def run_eval(stock: str, n_eval: int):
    cfg = TAQConfig(
        N=N, T=T, q0=Q0, p0=100.0,
        eta=ETA, gamma=GAMMA_I, a=A,
        sigma=SIGMA, discount=DISCOUNT,
        data_dir='data/processed', stock=stock, year=2014,
    )

    # Load test dates
    parquet_path = Path(cfg.data_dir) / f'{stock}_{cfg.year}.parquet'
    full_df = pd.read_parquet(parquet_path, columns=['date'])
    all_dates = sorted(full_df['date'].unique().tolist())
    test_dates = [d for d in all_dates if int(d[5:7]) in [10, 11, 12]]

    test_env = TAQEnv(cfg, test_dates)
    state_dim = test_env.state_dim
    n_actions = test_env.n_actions
    device = torch.device('cpu')

    # Find checkpoint directory
    out_dir = PROJECT_ROOT / 'results' / 'taq' / stock
    ckpt_dir = out_dir / 'fold1' / 'checkpoints'
    print(f'  Checkpoint dir: {ckpt_dir}')

    # Build agents
    base_cfg = EnvConfig(
        N=N, T=T, q0=Q0, p0=100.0,
        eta=ETA, gamma=GAMMA_I, a=A,
        sigma=SIGMA, discount=DISCOUNT,
    )

    agents = {}
    agents['TWAP'] = TWAPAgent(base_cfg)
    agents['AC'] = AlmgrenChrissAgent(base_cfg, risk_aversion=1e-6)

    rl_cfg = DeepRLConfig()
    agents['DQN'] = DQNAgent(rl_cfg, state_dim, n_actions, device=device, seed=SEED)
    agents['DDQN'] = DDQNAgent(rl_cfg, state_dim, n_actions, device=device, seed=SEED+1)

    iqn_cfg = AgentConfig(cvar_alpha=1.0)
    agents['IQN-neutral'] = IQNAgent(iqn_cfg, state_dim, n_actions, device=device, seed=SEED+3)

    # Load checkpoints
    for name in ['DQN', 'DDQN', 'IQN-neutral']:
        best_file = ckpt_dir / f'{name}_best.pt'
        if best_file.exists():
            agents[name].load(str(best_file))
        else:
            print(f'  WARNING: {best_file} not found!')

    # CVaR variants share IQN-neutral weights
    for alpha in [0.90, 0.95]:
        cvar_cfg = AgentConfig(cvar_alpha=alpha)
        name = f'IQN-CVaR_{alpha:.2f}'
        agents[name] = IQNAgent(cvar_cfg, state_dim, n_actions, device=device, seed=SEED+3)
        agents[name].online_net.load_state_dict(agents['IQN-neutral'].online_net.state_dict())
        agents[name].target_net.load_state_dict(agents['IQN-neutral'].target_net.state_dict())

    # Evaluate all agents
    eval_seed = SEED + 99_999
    all_results = []
    trackers = {}

    for name, agent in agents.items():
        test_env.seed(eval_seed)
        tracker = EpisodeTracker()

        for _ in range(n_eval):
            state = test_env.reset()
            if hasattr(agent, 'reset') and callable(agent.reset):
                agent.reset()
            tracker.begin_episode()
            done = False
            info = {}
            while not done:
                action = agent.select_action(state, eval_mode=True)
                state, reward, done, info = test_env.step(action)
                tracker.step(reward, info)
            tracker.end_episode(info)

        results = tracker.compute_metrics()
        results['agent_name'] = name
        all_results.append(results)
        trackers[name] = tracker
        print(f'  {name}: mean_IS={results["mean_IS_bps"]:.4f}, '
              f'CVaR95={results["CVaR_0.95_bps"]:.4f}')

    # Print table
    table = format_comparison_table(all_results, bps=True)
    print(f'\n  Test Results ({n_eval} episodes):')
    print(table)

    # Save IS arrays for plotting
    is_dict = {}
    for name, tracker in trackers.items():
        is_dict[name] = np.array(tracker.is_values) * 1e4
    pkl_path = out_dir / 'fold1' / 'is_arrays.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(is_dict, f)
    print(f'\n  IS arrays saved to {pkl_path}')

    # Save results JSON
    fold_results = []
    for r in all_results:
        fold_results.append({
            'agent': r['agent_name'],
            'mean_IS': float(r['mean_IS_bps']),
            'std_IS': float(r['std_IS_bps']),
            'CVaR_90': float(r['CVaR_0.90_bps']),
            'CVaR_95': float(r['CVaR_0.95_bps']),
            'max_IS': float(r['max_IS_bps']),
            'GL': float(r['GL_ratio']),
        })
    with open(out_dir / 'fold1' / 'eval_results.json', 'w') as f:
        json.dump(fold_results, f, indent=2)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--stock', type=str, default='AAPL')
    p.add_argument('--n-eval', type=int, default=2000)
    args = p.parse_args()

    print(f'\n  Evaluating {args.stock} ({args.n_eval} episodes)...\n')
    run_eval(args.stock, args.n_eval)