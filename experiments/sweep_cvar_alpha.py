"""
experiments/sweep_cvar_alpha.py
-------------------------------
R2 — CVaR-alpha sweep (EVAL-ONLY, no retraining).

Loads a trained IQN-neutral checkpoint and evaluates it at a range of CVaR
truncation levels alpha. Only the inference-time tau range changes
(agent.cfg.cvar_alpha -> tau_high in IQNAgent.select_action); the network
weights are never modified. alpha=1.0 reproduces the risk-neutral policy — a
sanity check against the IQN-neutral row of the main results table.

Outputs, under results/_cvar_sweep/<tag>/:
  - frontier.png / frontier.pdf : mean-IS vs CVaR_0.95 frontier
  - sweep.csv                   : alpha, mean_IS_bps, CVaR_0.95_bps, ...
  - config.json                 : full run config

Usage (Phase-3, after checkpoints exist):
    python experiments/sweep_cvar_alpha.py --env jump
    python experiments/sweep_cvar_alpha.py --env taq --n-eval 5000
    python experiments/sweep_cvar_alpha.py --env ac --checkpoint <path>
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from agents.iqn_agents import IQNAgent, AgentConfig
from evaluation.metrics import EpisodeTracker
from exp_utils import dump_config_json

ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

SIM_ENV_NAMES = {'ac': 'almgren_chriss', 'ou': 'mean_reverting', 'jump': 'jump_diffusion'}


def build_env(env_key):
    """Return (env, tag, default_ckpt_dir) for a sim env key or 'taq'."""
    if env_key == 'taq':
        import run_tag as RT
        import pandas as pd
        RT.DEVICE = 'cpu'
        cfg = RT.build_taq_config()
        parquet = Path(cfg.data_dir) / f'{cfg.stock}_{cfg.year}.parquet'
        all_dates = sorted(pd.read_parquet(parquet, columns=['date'])['date'].unique().tolist())
        _, _, test_months = RT.FOLDS[0]
        test_dates = RT.get_dates_for_months(all_dates, test_months)
        env = RT.TAQEnv(cfg, test_dates)
        return env, f'taq_{cfg.stock}', PROJECT_ROOT / 'results' / 'taq' / cfg.stock / 'fold1' / 'checkpoints'

    from envs import AlmgrenChrissEnv, MeanRevertingEnv, JumpDiffusionEnv, SimConfig
    import run_simulation as RS
    env_name = SIM_ENV_NAMES[env_key]
    cfg = SimConfig(**RS.DEFAULT_SIM_CONFIG)
    cls = {'almgren_chriss': AlmgrenChrissEnv,
           'mean_reverting': MeanRevertingEnv,
           'jump_diffusion': JumpDiffusionEnv}[env_name]
    return cls(cfg), env_name, PROJECT_ROOT / 'results' / env_name / 'checkpoints'


def find_checkpoint(ckpt_arg, ckpt_dir):
    if ckpt_arg:
        return Path(ckpt_arg)
    best = ckpt_dir / 'IQN-neutral_best.pt'
    if best.exists():
        return best
    eps = sorted(glob.glob(str(ckpt_dir / 'IQN-neutral_ep*.pt')))
    if eps:
        return Path(eps[-1])
    raise FileNotFoundError(
        f'No IQN-neutral checkpoint found in {ckpt_dir} (pass --checkpoint).')


def evaluate_at_alpha(agent, env, n_eval, seed, alpha):
    """Eval the (fixed-weight) agent with CVaR truncation tau_high=alpha."""
    agent.cfg.cvar_alpha = alpha          # only changes inference-time tau range
    env.seed(seed)
    tr = EpisodeTracker()
    for _ in range(n_eval):
        state = env.reset()
        if hasattr(agent, 'reset') and callable(agent.reset):
            agent.reset()
        tr.begin_episode()
        done, info = False, {}
        while not done:
            a = agent.select_action(state, eval_mode=True)
            state, r, done, info = env.step(a)
            tr.step(r, info)
        tr.end_episode(info)
    return tr.compute_metrics()


def main():
    ap = argparse.ArgumentParser(description='R2 CVaR-alpha sweep (eval-only)')
    ap.add_argument('--env', required=True, choices=list(SIM_ENV_NAMES) + ['taq'])
    ap.add_argument('--checkpoint', default=None, help='IQN-neutral checkpoint (.pt)')
    ap.add_argument('--n-eval', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--alphas', nargs='+', type=float, default=ALPHAS)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    env, tag, ckpt_dir = build_env(args.env)
    ckpt = find_checkpoint(args.checkpoint, ckpt_dir)
    state_dim, n_actions = env.state_dim, env.n_actions

    agent = IQNAgent(AgentConfig(cvar_alpha=1.0), state_dim, n_actions,
                     device=torch.device('cpu'), seed=args.seed)
    agent.load(str(ckpt))
    print(f'Loaded {ckpt}  (env={tag}, state_dim={state_dim}, n_actions={n_actions})')

    rows = []
    for alpha in args.alphas:
        m = evaluate_at_alpha(agent, env, args.n_eval, args.seed + 99_999, alpha)
        rows.append({'alpha': alpha,
                     'mean_IS_bps': float(m['mean_IS_bps']),
                     'std_IS_bps': float(m['std_IS_bps']),
                     'CVaR_0.90_bps': float(m['CVaR_0.90_bps']),
                     'CVaR_0.95_bps': float(m['CVaR_0.95_bps']),
                     'max_IS_bps': float(m['max_IS_bps'])})
        print(f'  alpha={alpha:<4} mean_IS={rows[-1]["mean_IS_bps"]:.4f} '
              f'CVaR95={rows[-1]["CVaR_0.95_bps"]:.4f} bps')

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / 'results' / '_cvar_sweep' / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _plot_frontier(rows, tag, out_dir)
    dump_config_json({'kind': 'cvar_sweep', 'env': tag, 'checkpoint': str(ckpt),
                      'n_eval': args.n_eval, 'seed': args.seed,
                      'alphas': args.alphas}, out_dir / 'config.json')
    print(f'\nWrote {out_dir}/ (frontier.png/pdf, sweep.csv, config.json)')


def _plot_frontier(rows, tag, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cv = [r['CVaR_0.95_bps'] for r in rows]
    mn = [r['mean_IS_bps'] for r in rows]
    al = [r['alpha'] for r in rows]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(cv, mn, '-o', color='#1f77b4', lw=1.6, ms=6, zorder=3)
    for a, c, m in zip(al, cv, mn):
        ax.annotate(f'α={a:g}', (c, m), textcoords='offset points',
                    xytext=(6, 4), fontsize=8, color='#333333')
    ax.set_xlabel(r'CVaR$_{0.95}$ of implementation shortfall (bps)')
    ax.set_ylabel('Mean IS (bps)')
    ax.set_title(f'CVaR-α frontier — {tag}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'frontier.png', dpi=200)
    fig.savefig(out_dir / 'frontier.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
