"""
experiments/run_taq.py
-----------------------
Phase 2: Walk-forward backtest on NYSE TAQ data.

Follows Ning et al. (2021) methodology with IQN extension.
Uses mid-price + impact model on NBBO data from WRDS.

Walk-forward CV with expanding training window:
    Fold 1: Train Jan-Jun,  Val Jul-Aug,    Test Sep-Oct
    Fold 2: Train Jan-Aug,  Val Sep-Oct,    Test Nov-Dec

Results saved to results/taq/{stock}/

Usage:
    python experiments/run_taq.py
    python experiments/run_taq.py --episodes 5000 --eval-episodes 200
    python experiments/run_taq.py --stock MSFT
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

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
from evaluation.metrics import (
    EpisodeTracker, format_comparison_table, format_table_row,
)
from evaluation.visualizer import Visualizer


# ============================================================================
# ███  CONFIG  ███
# ============================================================================

STOCK = 'AAPL'
YEAR  = 2014

# MDP settings
# With:
N        = 5
T        = 60.0      # 60-minute execution window (12-min steps)     # 30-minute execution window (3-min steps)
Q0       = 5000      # shares to liquidate
A        = 0.0001     # quadratic penalty
ETA = 1e-5 # temporary impact coefficient
GAMMA_I  = 2.5e-7    # permanent impact (for AC baseline only)
SIGMA    = 0.001     # baseline vol
DISCOUNT = 0.99

# Training
N_EPISODES      = 50_000
EVAL_FREQ       = 1_000
CHECKPOINT_FREQ = 1_000
SEED            = 42

# Validation & test
N_VAL_EVAL  = 300
N_TEST_EVAL = 10_000

# Walk-forward folds: (train_months, val_months, test_months)
# Months are 1-indexed
FOLDS = [
    ([1,2,3,4,5,6,7], [8,9], [10,11,12]),
]

# ============================================================================
# ███  END CONFIG  ███
# ============================================================================


def get_dates_for_months(all_dates: List[str], months: List[int]) -> List[str]:
    """Filter dates by month numbers."""
    return [d for d in all_dates if int(d[5:7]) in months]


def build_taq_config() -> TAQConfig:
    return TAQConfig(
        N=N, T=T, q0=Q0, p0=100.0,
        eta=ETA, gamma=GAMMA_I, a=A,
        sigma=SIGMA, discount=DISCOUNT,
        data_dir='data/processed',
        stock=STOCK, year=YEAR,
    )


def build_all_agents(state_dim: int, n_actions: int, seed: int):
    """Build all agents including baselines and IQN variants."""

    device = torch.device('cpu')
    # if torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # else:
    #     device = torch.device('cpu')
    print(f'  Device: {device}')

    agents = {}

    # Rule-based baselines
    base_cfg = EnvConfig(
        N=N, T=T, q0=Q0, p0=100.0,
        eta=ETA, gamma=GAMMA_I, a=A,
        sigma=SIGMA, discount=DISCOUNT,
    )
    agents['TWAP'] = TWAPAgent(base_cfg)
    agents['AC']   = AlmgrenChrissAgent(base_cfg, risk_aversion=1e-6)

    # Deep RL baselines
    rl_cfg = DeepRLConfig()
    agents['DQN']  = DQNAgent(rl_cfg, state_dim, n_actions, device=device, seed=seed)
    agents['DDQN'] = DDQNAgent(rl_cfg, state_dim, n_actions, device=device, seed=seed+1)

    # IQN
    iqn_cfg = AgentConfig(cvar_alpha=1.0)
    agents['IQN-neutral'] = IQNAgent(
        iqn_cfg, state_dim, n_actions, device=device, seed=seed+3)

    # CVaR variants
    for alpha in [0.90, 0.95]:
        cvar_cfg = AgentConfig(cvar_alpha=alpha)
        agents[f'IQN-CVaR_{alpha:.2f}'] = IQNAgent(
            cvar_cfg, state_dim, n_actions, device=device, seed=seed+3)

    return agents


def train_with_validation(agent, train_env, val_env, n_episodes,
                          ckpt_dir, seed):
    """Train agent, select best checkpoint by val CVaR₉₅."""
    train_env.seed(seed)
    val_env.seed(seed + 10_000)

    best_cvar = float('inf')
    best_ckpt = None
    t_start = time.time()

    log = {
        'losses': [],
        'episode_rewards': [],
        'eval_history': [],
    }

    for ep in range(1, n_episodes + 1):
        state = train_env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = train_env.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                log['losses'].append(loss)
            ep_reward += reward
            state = next_state

        log['episode_rewards'].append(ep_reward)

        # Periodic validation
        if ep % EVAL_FREQ == 0:
            val_results = quick_eval(agent, val_env, N_VAL_EVAL, seed + ep)
            val_results['episode'] = ep
            log['eval_history'].append(val_results)
            elapsed = time.time() - t_start
            eps_str = getattr(agent, 'epsilon', 0.0)
            print(f'    [{agent.name}] ep={ep:>6d} | '
                  f'val_IS={val_results["mean_IS_bps"]:>8.4f} | '
                  f'val_CVaR₉₅={val_results["CVaR_0.95_bps"]:>8.4f} | '
                  f'ε={eps_str:.3f} | {elapsed:.0f}s')

            cvar95 = val_results['CVaR_0.95_bps']
            if cvar95 < best_cvar:
                best_cvar = cvar95
                best_ckpt = ckpt_dir / f'{agent.name}_best.pt'
                agent.save(str(best_ckpt))

        # Periodic checkpoint
        if ep % CHECKPOINT_FREQ == 0:
            agent.save(str(ckpt_dir / f'{agent.name}_ep{ep}.pt'))

    # Restore best
    if best_ckpt is not None and best_ckpt.exists():
        agent.load(str(best_ckpt))
        print(f'    Restored best (val CVaR₉₅={best_cvar:.4f})')

    log['wall_time'] = time.time() - t_start
    return log

def quick_eval(agent, env, n_eval, seed):
    """Quick evaluation returning metrics dict."""
    env.seed(seed)
    tracker = EpisodeTracker()

    for _ in range(n_eval):
        state = env.reset()
        if hasattr(agent, 'reset') and callable(agent.reset):
            agent.reset()
        tracker.begin_episode()
        done = False
        info = {}
        while not done:
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, info = env.step(action)
            tracker.step(reward, info)
        tracker.end_episode(info)

    return tracker.compute_metrics()


def evaluate_all(agents, env, n_eval, seed):
    """Evaluate all agents, return results and trackers."""
    all_results = []
    trackers = {}

    for name, agent in agents.items():
        env.seed(seed)
        tracker = EpisodeTracker()

        for _ in range(n_eval):
            state = env.reset()
            if hasattr(agent, 'reset') and callable(agent.reset):
                agent.reset()
            tracker.begin_episode()
            done = False
            info = {}
            while not done:
                action = agent.select_action(state, eval_mode=True)
                state, reward, done, info = env.step(action)
                tracker.step(reward, info)
            tracker.end_episode(info)

        results = tracker.compute_metrics()
        results['agent_name'] = name
        all_results.append(results)
        trackers[name] = tracker

    return all_results, trackers


def share_iqn_weights(agents):
    """Copy IQN-neutral weights to CVaR variants."""
    neutral = agents.get('IQN-neutral')
    if neutral is None:
        return
    for name, agent in agents.items():
        if name.startswith('IQN-CVaR') and hasattr(agent, 'online_net'):
            agent.online_net.load_state_dict(neutral.online_net.state_dict())
            agent.target_net.load_state_dict(neutral.target_net.state_dict())
            print(f'  Copied IQN-neutral → {name}')


# ============================================================================
# Main pipeline
# ============================================================================

def run():
    cfg = build_taq_config()
    out_dir = Path('results') / 'taq' / STOCK
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all dates from the parquet
    parquet_path = Path(cfg.data_dir) / f'{cfg.stock}_{cfg.year}.parquet'
    full_df = pd.read_parquet(parquet_path, columns=['date'])
    all_dates = sorted(full_df['date'].unique().tolist())
    n_dates = len(all_dates)

    print(f'\n{"="*70}')
    print(f'  TAQ Walk-Forward: {STOCK} {YEAR}')
    print(f'  q0={Q0} shares | N={N} | T={T}min | η={ETA}')
    print(f'  Total trading days: {n_dates}')
    print(f'  Folds: {len(FOLDS)}')
    print(f'{"="*70}')

    # Collect results across folds
    fold_results = []

    for fold_idx, (train_months, val_months, test_months) in enumerate(FOLDS):
        fold_num = fold_idx + 1

        train_dates = get_dates_for_months(all_dates, train_months)
        val_dates   = get_dates_for_months(all_dates, val_months)
        test_dates  = get_dates_for_months(all_dates, test_months)

        print(f'\n{"━"*70}')
        print(f'  Fold {fold_num}:')
        print(f'    Train months {train_months}: {len(train_dates)} days')
        print(f'    Val months {val_months}: {len(val_dates)} days')
        print(f'    Test months {test_months}: {len(test_dates)} days')
        print(f'{"━"*70}')

        if len(train_dates) == 0 or len(val_dates) == 0 or len(test_dates) == 0:
            print(f'  Skipping fold {fold_num}: insufficient data')
            continue

        # Build environments
        train_env = TAQEnv(cfg, train_dates)
        val_env   = TAQEnv(cfg, val_dates)
        test_env  = TAQEnv(cfg, test_dates)

        state_dim = train_env.state_dim
        n_actions = train_env.n_actions

        # Build fresh agents for this fold
        agents = build_all_agents(state_dim, n_actions, SEED)

        # Train learned agents
        fold_ckpt_dir = out_dir / f'fold{fold_num}' / 'checkpoints'
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        learned = {name: agent for name, agent in agents.items()
                   if hasattr(agent, 'update') and hasattr(agent, 'store')
                   and name not in ('TWAP', 'AC')
                   and not name.startswith('IQN-CVaR')}

        print(f'\n  Training {len(learned)} agents...')
        training_logs = {}
        for name, agent in learned.items():
            print(f'\n  ── {name} {"─"*(50 - len(name))}')
            log = train_with_validation(
                agent, train_env, val_env, N_EPISODES,
                fold_ckpt_dir, SEED,
            )
            training_logs[name] = log

        # Share IQN weights to CVaR variants
        share_iqn_weights(agents)

        # Test
        print(f'\n  Testing on months {test_months} '
              f'({N_TEST_EVAL} episodes)...')
        all_results, trackers = evaluate_all(
            agents, test_env, N_TEST_EVAL, SEED + 99_999)
        
        # Save raw IS arrays for plotting
        import pickle
        is_dict = {}
        for name, tracker in trackers.items():
            is_dict[name] = np.array(tracker.is_values) * 1e4
        with open(out_dir / f'fold{fold_num}' / 'is_arrays.pkl', 'wb') as f:
            pickle.dump(is_dict, f)

        # Print fold table
        table = format_comparison_table(all_results, bps=True)
        print(f'\n  Fold {fold_num} Test Results:')
        print(table)

        # Save fold results
        fold_log = out_dir / f'fold{fold_num}' / 'logs'
        fold_log.mkdir(parents=True, exist_ok=True)
        with open(fold_log / 'table.txt', 'w') as f:
            f.write(f'Fold {fold_num}: Train months {train_months}, '
                    f'Val {val_months}, Test {test_months}\n\n')
            f.write(table)

        # Save figures
        fold_fig = out_dir / f'fold{fold_num}' / 'figures'
        fold_fig.mkdir(parents=True, exist_ok=True)
        viz = Visualizer(save_dir=str(fold_fig), bps=True)
        viz.plot_is_distributions(all_results, trackers=trackers)
        viz.plot_mean_cvar_frontier(all_results, alpha=0.95)
        viz.plot_inventory_trajectories(trackers, q0=Q0, n_periods=N)
        viz.plot_comparison_heatmap(all_results)

        # Quantile distribution for IQN
        for name, agent in agents.items():
            if hasattr(agent, 'online_net') and 'IQN-neutral' in name:
                sample_state = np.array([0.5, 0.5, 0.0, 0.01, 0.0],
                                        dtype=np.float32)
                viz.plot_quantile_distribution(agent, sample_state)
                break

        import matplotlib.pyplot as plt
        plt.close('all')

        # Training curves
        for name, log in training_logs.items():
            viz_tc = Visualizer(save_dir=str(fold_fig), bps=True)
            viz_tc.plot_training_curves(log)
            for ext in ['pdf', 'png']:
                src = fold_fig / f'training_curves.{ext}'
                dst = fold_fig / f'training_curves_{name}.{ext}'
                if src.exists():
                    src.rename(dst)
            print(f'  ✓ Training curves ({name})')

        # Collect for summary
        for r in all_results:
            fold_results.append({
                'fold': fold_num,
                'train_months': str(train_months),
                'val_months': str(val_months),
                'test_months': str(test_months),
                'agent': r['agent_name'],
                'mean_IS': float(r['mean_IS_bps']),
                'std_IS': float(r['std_IS_bps']),
                'CVaR_90': float(r['CVaR_0.90_bps']),
                'CVaR_95': float(r['CVaR_0.95_bps']),
                'max_IS': float(r['max_IS_bps']),
                'GL': float(r['GL_ratio']),
            })

    # ======================================================================
    # Summary across folds
    # ======================================================================

    if not fold_results:
        print('\nNo folds completed!')
        return

    print(f'\n{"="*70}')
    print(f'  TAQ WALK-FORWARD SUMMARY: {STOCK} {YEAR}')
    print(f'{"="*70}')

    # Collect unique agent names preserving order
    agent_names = []
    seen = set()
    for r in fold_results:
        if r['agent'] not in seen:
            agent_names.append(r['agent'])
            seen.add(r['agent'])

    # Header
    n_folds = len(FOLDS)
    header = f'{"Agent":<22s}'
    for i in range(1, n_folds + 1):
        header += f' | Fold {i} IS | Fold {i} CVaR₉₅'
    header += ' | Mean IS ± Std | Mean CVaR₉₅ ± Std'
    print(header)
    print('─' * len(header))

    for name in agent_names:
        folds = [r for r in fold_results if r['agent'] == name]
        row = f'{name:<22s}'

        is_vals = []
        cvar_vals = []

        for i in range(1, n_folds + 1):
            fold_data = [r for r in folds if r['fold'] == i]
            if fold_data:
                is_val = fold_data[0]['mean_IS']
                cvar_val = fold_data[0]['CVaR_95']
                is_vals.append(is_val)
                cvar_vals.append(cvar_val)
                row += f' | {is_val:>8.4f} | {cvar_val:>11.4f}'
            else:
                row += f' | {"N/A":>8s} | {"N/A":>11s}'

        if is_vals:
            mean_is = np.mean(is_vals)
            std_is = np.std(is_vals)
            mean_cvar = np.mean(cvar_vals)
            std_cvar = np.std(cvar_vals)
            row += (f' | {mean_is:>6.4f}±{std_is:<6.4f}'
                    f' | {mean_cvar:>8.4f}±{std_cvar:<8.4f}')

        print(row)

    # Save summary
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(fold_results, f, indent=2)

    with open(out_dir / 'summary_table.txt', 'w') as f:
        f.write(f'TAQ Walk-Forward Summary: {STOCK} {YEAR}\n')
        f.write(f'q0={Q0}, N={N}, T={T}, eta={ETA}, a={A}\n\n')

        for name in agent_names:
            folds = [r for r in fold_results if r['agent'] == name]
            is_vals = [r['mean_IS'] for r in folds]
            cvar_vals = [r['CVaR_95'] for r in folds]

            if is_vals:
                f.write(f'{name:<22s}: '
                        f'Mean IS = {np.mean(is_vals):>8.4f} ± '
                        f'{np.std(is_vals):.4f} bps, '
                        f'CVaR₉₅ = {np.mean(cvar_vals):>8.4f} ± '
                        f'{np.std(cvar_vals):.4f} bps '
                        f'({len(is_vals)} folds)\n')

    print(f'\nAll results saved to {out_dir}/')


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(
        description='Walk-forward CV on NYSE TAQ data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--stock', type=str, default=None,
                   help='Stock ticker (AAPL, MSFT, GOOG)')
    p.add_argument('--episodes', type=int, default=None,
                   help='Training episodes per agent per fold')
    p.add_argument('--eval-episodes', type=int, default=None,
                   help='Test episodes per fold')
    args = p.parse_args()

    if args.stock:
        STOCK = args.stock
    if args.episodes:
        N_EPISODES = args.episodes
    if args.eval_episodes:
        N_TEST_EVAL = args.eval_episodes

    run()