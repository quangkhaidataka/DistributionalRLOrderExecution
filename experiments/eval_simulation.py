"""
experiments/eval_simulation.py
-------------------------------
Evaluate existing simulation checkpoints and save IS arrays for plotting.

Usage:
    python experiments/eval_simulation.py --env jump_diffusion
    python experiments/eval_simulation.py --env almgren_chriss
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.simulated_env import AlmgrenChrissEnv, JumpDiffusionEnv, SimConfig
from envs.base_env import EnvConfig
from agents.baselines import TWAPAgent, AlmgrenChrissAgent, DQNAgent, DDQNAgent, DeepRLConfig
from agents.iqn_agents import IQNAgent, AgentConfig
from evaluation.metrics import EpisodeTracker, format_comparison_table

import torch

# ============================================================================
# CONFIG — must match run_simulation.py
# ============================================================================
SIM_CONFIG = dict(
    N=5, T=60.0, q0=100_000, p0=100.0,
    sigma=0.00095, eta=2.5e-6, gamma=2.5e-7,
    discount=0.99, ou_theta=0.15, ou_mu=100.0,
    jump_intensity=0.3, jump_mean=-0.002, jump_std=0.005,
)

SEED = 42
N_EVAL = 10_000

# Specific checkpoints that produced the reported results
# Change these if your best checkpoints are different
CHECKPOINTS = {
    'jump_diffusion': {
        'DQN':         'DQN_ep18000.pt',
        'DDQN':        'DDQN_ep3000.pt',
        'IQN-neutral': 'IQN-neutral_ep9000.pt',
    },
    'almgren_chriss': {
        'DQN':         None,  # will find best automatically
        'DDQN':        None,
        'IQN-neutral': None,
    },
}


def evaluate_agent(agent, env, n_eval, seed):
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
    return tracker


def run(env_name: str):
    cfg = SimConfig(**SIM_CONFIG)

    if env_name == 'jump_diffusion':
        env = JumpDiffusionEnv(cfg)
    elif env_name == 'almgren_chriss':
        env = AlmgrenChrissEnv(cfg)
    else:
        raise ValueError(f'Unknown env: {env_name}')

    state_dim = env.state_dim
    n_actions = env.n_actions
    device = torch.device('cpu')

    results_dir = PROJECT_ROOT / 'results' / env_name
    ckpt_dir = results_dir / 'checkpoints'
    log_dir = results_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build agents
    agents = {}
    agents['TWAP'] = TWAPAgent(cfg)
    agents['AC'] = AlmgrenChrissAgent(cfg, risk_aversion=1e-6)

    rl_cfg = DeepRLConfig()
    agents['DQN'] = DQNAgent(rl_cfg, state_dim, n_actions, device=device, seed=SEED)
    agents['DDQN'] = DDQNAgent(rl_cfg, state_dim, n_actions, device=device, seed=SEED+1)

    if env_name == 'jump_diffusion':
        iqn_cfg = AgentConfig(cvar_alpha=1.0, hidden_dim=128, cos_embedding_dim=64)
    else:
        iqn_cfg = AgentConfig(cvar_alpha=1.0)
    agents['IQN-neutral'] = IQNAgent(iqn_cfg, state_dim, n_actions, device=device, seed=SEED+3)

    # Load specific checkpoints
    ckpt_map = CHECKPOINTS.get(env_name, {})
    for name in ['DQN', 'DDQN', 'IQN-neutral']:
        ckpt_file = ckpt_map.get(name)
        if ckpt_file:
            path = ckpt_dir / ckpt_file
        else:
            # Try _best.pt, then last checkpoint
            path = ckpt_dir / f'{name}_best.pt'
            if not path.exists():
                import glob
                files = sorted(glob.glob(str(ckpt_dir / f'{name}_ep*.pt')))
                path = Path(files[-1]) if files else None

        if path and path.exists():
            agents[name].load(str(path))
            print(f'  Loaded {path.name}')
        else:
            print(f'  WARNING: no checkpoint for {name}')

    # IQN-CVaR variants share weights
    for alpha in [0.90, 0.95]:
        if env_name == 'jump_diffusion':
            cvar_cfg = AgentConfig(cvar_alpha=alpha, hidden_dim=128, cos_embedding_dim=64)
        else:
            cvar_cfg = AgentConfig(cvar_alpha=alpha)
        name = f'IQN-CVaR_{alpha:.2f}'
        agents[name] = IQNAgent(cvar_cfg, state_dim, n_actions, device=device, seed=SEED+3)
        agents[name].online_net.load_state_dict(agents['IQN-neutral'].online_net.state_dict())
        agents[name].target_net.load_state_dict(agents['IQN-neutral'].target_net.state_dict())

    # Evaluate
    eval_seed = SEED + 99_999
    all_results = []
    is_dict = {}

    for name, agent in agents.items():
        print(f'  Evaluating {name}...')
        tracker = evaluate_agent(agent, env, N_EVAL, eval_seed)
        results = tracker.compute_metrics()
        results['agent_name'] = name
        all_results.append(results)
        is_dict[name] = np.array(tracker.is_values) * 1e4
        print(f'    mean={results["mean_IS_bps"]:.4f}, '
              f'CVaR95={results["CVaR_0.95_bps"]:.4f}')

    # Print table
    table = format_comparison_table(all_results, bps=True)
    print(f'\n  Results ({N_EVAL} episodes):')
    print(table)

    # Save IS arrays
    pkl_path = log_dir / 'is_arrays.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(is_dict, f)
    print(f'\n  IS arrays saved to {pkl_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='jump_diffusion',
                   choices=['jump_diffusion', 'almgren_chriss'])
    args = p.parse_args()

    print(f'\n  Evaluating {args.env}...\n')
    run(args.env)