"""
plots/plot_is_subplots.py
--------------------------
Generate paper-quality IS distribution subplots (2x3 grid).
Each agent gets its own panel for clarity.

Usage:
    python plots/plot_is_subplots.py --env jump_diffusion
    python plots/plot_is_subplots.py --env taq --stock AAPL
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.base_env import EnvConfig, N_ACTIONS
from evaluation.metrics import EpisodeTracker, cvar_alpha


# ============================================================================
# CONFIG
# ============================================================================

# Agents to plot (in order)
AGENTS = ['TWAP', 'AC', 'DQN', 'DDQN', 'IQN-neutral', 'IQN-CVaR_0.95']

# Colors — professional palette
COLORS = {
    'TWAP':           '#4C72B0',   # steel blue
    'AC':             '#DD8452',   # warm orange
    'DQN':            '#55A868',   # muted green
    'DDQN':           '#C44E52',   # muted red
    'IQN-neutral':    '#8172B3',   # muted purple
    'IQN-CVaR_0.95':  '#CCB974',   # gold
    'IQN-CVaR_0.90':  '#64B5CD',   # light blue
}

BINS = 60


# def load_simulation_results(env_name: str) -> Dict[str, np.ndarray]:
#     import pickle
#     pkl_path = PROJECT_ROOT / 'results' / env_name / 'logs' / 'is_arrays.pkl'
#     with open(pkl_path, 'rb') as f:
#         return pickle.load(f)    

def load_simulation_results(env_name: str) -> Dict[str, np.ndarray]:
    import pickle
    pkl_path = PROJECT_ROOT / 'results' / env_name / 'logs' / 'is_arrays.pkl'
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


# In plot_is_subplots.py, replace load_taq_results with:
def load_taq_results(stock):
    import pickle
    pkl_path = PROJECT_ROOT / 'results' / 'taq' / stock / 'fold1' / 'is_arrays.pkl'
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def plot_is_subplots(is_arrays: Dict[str, np.ndarray],
                     title: str, save_path: Path):
    """Create 2x3 subplot grid of IS distributions."""

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    # Compute global x-range for consistent axes
    # all_vals = np.concatenate(list(is_arrays.values()))
    # x_min = np.percentile(all_vals, 0.5)
    # x_max = np.percentile(all_vals, 99.5)

    agent_order = [a for a in AGENTS if a in is_arrays]

    for idx, name in enumerate(agent_order):
        ax = axes[idx]
        arr = is_arrays[name]
        color = COLORS.get(name, '#333333')

        # Histogram
        lo = np.percentile(arr, 1)
        hi = np.percentile(arr, 99)
        pad = (hi - lo) * 0.1
        lo -= pad
        hi += pad

        # Histogram
        ax.hist(arr, bins=BINS, range=(lo, hi),
                color=color, alpha=0.7, edgecolor='white',
                linewidth=0.5, density=True)

        # CVaR 95 vertical line
        c95 = cvar_alpha(arr, 0.95)
        ax.axvline(c95, color='#C44E52', linestyle='--', linewidth=2.0,
                   label=f'CVaR$_{{0.95}}$={c95:.2f}')

        # Mean vertical line
        m = arr.mean()
        ax.axvline(m, color='black', linestyle='-', linewidth=1.5,
                   label=f'Mean={m:.2f}')

        # Labels
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right',
                  framealpha=0.9, edgecolor='none')
        ax.set_xlabel('IS (bps)', fontsize=10)

        if idx % 3 == 0:
            ax.set_ylabel('Density', fontsize=10)

        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)

    # Remove unused axes
    for idx in range(len(agent_order), len(axes)):
        axes[idx].set_visible(False)

    # Share y-axis for RL agents (panels 2,3,4,5 = DQN, DDQN, IQN-neutral, IQN-CVaR)
    rl_indices = [i for i, name in enumerate(agent_order) if name not in ('TWAP', 'AC')]
    if rl_indices:
        max_y = max(axes[i].get_ylim()[1] for i in rl_indices)
        for i in rl_indices:
            axes[i].set_ylim(0, max_y)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path).replace('.pdf', '.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(str(save_path).replace('.pdf', '.png'),
                dpi=300, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close(fig)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='jump_diffusion',
                   choices=['jump_diffusion', 'almgren_chriss', 'taq'])
    p.add_argument('--stock', type=str, default='AAPL')
    args = p.parse_args()

    if args.env == 'taq':
        print(f'Loading TAQ results for {args.stock}...')
        is_arrays = load_taq_results(args.stock)
        save_path = PROJECT_ROOT / 'results' / 'taq' / f'{args.stock}_is_subplots.pdf'
        title = f'IS Distribution by Agent — {args.stock} (NYSE TAQ 2014)'
    else:
        print(f'Loading simulation results ({args.env})...')
        is_arrays = load_simulation_results(args.env)
        save_path = PROJECT_ROOT / 'results' / args.env / 'figures' / 'is_subplots.pdf'
        title = f'IS Distribution by Agent — {args.env.replace("_", " ").title()}'

    plot_is_subplots(is_arrays, title, save_path)