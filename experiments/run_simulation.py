"""
experiments/run_simulation.py
-----------------------------
Phase 1: Simulated environment experiments.

This script produces the core results for the paper:
    - Table 3:  Agent comparison (mean IS, CVaR, VaR, GL ratio)
    - Figure 2: IS distribution histograms
    - Figure 3: Mean-CVaR efficient frontier
    - Figure 4: Training curves (loss + reward)
    - Figure 5: Inventory trajectories
    - Figure 6: Quantile distribution visualisation
    - Figure 7: Regime-conditioned IS comparison

Experimental design:
    Phase 1A — Almgren-Chriss (Gaussian IS)
        Purpose: sanity check. All agents should be close.
        Expected: IQN-neutral ≈ DDQN ≈ AC optimal.
        CVaR advantage should be minimal (Gaussian tails).

    Phase 1B — Regime-switching (fat-tailed IS)
        Purpose: demonstrate IQN-CVaR advantage.
        Expected: IQN-CVaR shows clear CVaR reduction.
        The Gaussian mixture from regime switching creates
        fat tails that distributional RL can exploit.

Reproducibility:
    - All random seeds are fixed and logged
    - Environment and agent configs are saved to results/logs/
    - Checkpoints saved every checkpoint_freq episodes
    - Full metrics saved as JSON for post-hoc analysis

Hardware target: MacBook Air M2, 8GB RAM
    - Training: ~15 min per agent on Regime-Switching env
    - Evaluation: ~2 min per agent (500 episodes)
    - Total Phase 1: ~2-3 hours

Usage:
    cd iqn_execution
    python experiments/run_simulation.py
    python experiments/run_simulation.py --env regime --episodes 20000
    python experiments/run_simulation.py --eval-only --checkpoint results/checkpoints/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project imports ─────────────────────────────────────────────────────────
# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from envs import (
    AlmgrenChrissEnv, MeanRevertingEnv, JumpDiffusionEnv,
    SimConfig, N_ACTIONS,
)
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.baselines import (
    TWAPAgent, AlmgrenChrissAgent, DQNAgent, DDQNAgent,
    QRDQNAgent, DeepRLConfig,
)
from evaluation.metrics import (
    MetricsSuite, EpisodeTracker,
    format_comparison_table, format_table_row,
)
from evaluation.visualizer import Visualizer


# ============================================================================
# Default hyperparameters
# ============================================================================

# Environment (calibrated to liquid NASDAQ large-cap)
DEFAULT_SIM_CONFIG = dict(
    N       = 5,
    T       = 60.0,
    q0      = 100_000,
    p0      = 100.0,
    sigma   = 0.00095,
    eta     = 2.5e-6,
    gamma   = 2.5e-7,
    discount= 0.99,
    ou_theta= 0.15,
    ou_mu   = 100.0,
    jump_intensity = 0.3,
    jump_mean      = -0.002,
    jump_std       = 0.005)

# Training
DEFAULT_TRAIN = dict(
    n_episodes       = 30_000,    # total training episodes
    eval_freq        = 1_000,     # evaluate every N episodes
    checkpoint_freq  = 1_000,     # save checkpoint every N episodes
    n_eval_episodes  = 10000,       # episodes per evaluation
    seed             = 42,
)

# CVaR levels to test
CVAR_ALPHAS = [0.90, 0.95]


# ============================================================================
# Training loop
# ============================================================================

def train_agent(
    agent,
    env,
    n_episodes    : int,
    eval_env      = None,
    eval_freq     : int   = 1000,
    n_eval        : int   = 500,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_freq: int  = 5000,
    seed          : int   = 42,
) -> Dict:
    """
    Train a single agent on the given environment.

    Returns a training log dict with keys:
        'losses', 'episode_rewards', 'episode_is',
        'eval_history', 'wall_time'

    The training loop is identical for ALL learned agents
    (DQN, DDQN, QR-DQN, IQN). This ensures fair comparison:
    same number of episodes, same environment seed sequence,
    same evaluation protocol.
    """
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed + 10_000)

    log = {
        'losses'          : [],
        'episode_rewards' : [],
        'episode_is'      : [],
        'eval_history'    : [],
        'wall_time'       : 0.0,
    }

    t_start   = time.time()
    total_steps = 0

    for ep in range(1, n_episodes + 1):
        state     = env.reset()
        ep_reward = 0.0
        done      = False
        info      = {}

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)

            loss = agent.update()
            if loss is not None:
                log['losses'].append(loss)

            ep_reward += reward
            state      = next_state
            total_steps += 1

        log['episode_rewards'].append(ep_reward)
        is_val = info.get('implementation_shortfall', -ep_reward)
        log['episode_is'].append(is_val)

        # ── Periodic evaluation ─────────────────────────────────
        if eval_env is not None and ep % eval_freq == 0:
            suite   = MetricsSuite(n_eval=n_eval, seed=seed + ep)
            results = suite.evaluate(agent, eval_env)
            results['episode'] = ep
            results['total_steps'] = total_steps
            log['eval_history'].append(results)

            elapsed = time.time() - t_start
            eps_str = getattr(agent, 'epsilon', 0.0)
            print(f'  [{agent.name}] ep={ep:>6d} | '
                  f'IS={results["mean_IS_bps"]:>6.2f}±{results["std_IS_bps"]:<5.2f} bps | '
                  f'CVaR₉₅={results["CVaR_0.95_bps"]:>6.2f} bps | '
                  f'ε={eps_str:.3f} | '
                  f'{elapsed:.0f}s')

        # ── Checkpoint ──────────────────────────────────────────
        if checkpoint_dir is not None and ep % checkpoint_freq == 0:
            ckpt_path = checkpoint_dir / f'{agent.name}_ep{ep}.pt'
            agent.save(str(ckpt_path))


    log['wall_time'] = time.time() - t_start

    # Restore best checkpoint based on validation CVaR_0.95
    if checkpoint_dir is not None and log['eval_history']:
        best_eval = min(log['eval_history'],
                        key=lambda x: x.get('CVaR_0.95_bps', float('inf')))
        best_ep = best_eval['episode']
        best_ckpt = checkpoint_dir / f'{agent.name}_ep{best_ep}.pt'
        if best_ckpt.exists():
            agent.load(str(best_ckpt))
            print(f'  Restored best checkpoint: ep={best_ep} '
                  f'(CVaR₉₅={best_eval["CVaR_0.95_bps"]:.2f} bps)')
        else:
            print(f'  Warning: best checkpoint ep={best_ep} not found on disk')

    return log



# ============================================================================
# Agent factory
# ============================================================================

def build_agents(
    env_config : SimConfig,
    state_dim  : int,
    n_actions  : int,
    seed       : int = 42,
    cvar_alphas: List[float] = None,
) -> Dict[str, object]:
    """
    Construct all agents for the experiment.

    Returns an OrderedDict-style dict preserving the paper's
    ablation order: TWAP → AC → DQN → DDQN → QR-DQN → IQN-neutral → IQN-CVaR

    All learned agents share identical hyperparameters
    (lr, batch_size, buffer_size, network size) for fair comparison.
    The ONLY difference between them is the loss function / architecture.
    """
    import torch
    device = torch.device('cpu')
    # if torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # elif torch.cuda.is_available():
    #     device = torch.device('cuda')

    if cvar_alphas is None:
        cvar_alphas = CVAR_ALPHAS

    agents = {}

    # ── Rule-based baselines ──────────────────────────────────
    agents['TWAP'] = TWAPAgent(env_config)
    agents['AC']   = AlmgrenChrissAgent(env_config, risk_aversion=1e-6)

    # ── Deep RL baselines ─────────────────────────────────────
    rl_cfg = DeepRLConfig()

    agents['DQN'] = DQNAgent(
        rl_cfg, state_dim, n_actions, device=device, seed=seed,
    )
    agents['DDQN'] = DDQNAgent(
        rl_cfg, state_dim, n_actions, device=device, seed=seed + 1,
    )
    # agents['QR-DQN'] = QRDQNAgent(
    #     rl_cfg, state_dim, n_actions, device=device, seed=seed + 2,
    # )

    # ── IQN agents (our contribution) ─────────────────────────
    #    All IQN variants share the same training — only
    #    inference-time τ sampling differs (cvar_alpha).
    #    We train ONE network with cvar_alpha=1.0 (risk-neutral
    #    training), then create CVaR variants by copying weights
    #    and changing cvar_alpha at inference time.
    #    This is the paper's key design: zero-cost risk control.

    iqn_cfg_neutral = AgentConfig(cvar_alpha=1.0)
    agents['IQN-neutral'] = IQNAgent(
        iqn_cfg_neutral, state_dim, n_actions, device=device, seed=seed + 3,
    )

    for alpha in cvar_alphas:
        iqn_cfg_cvar = AgentConfig(cvar_alpha=alpha)
        name = f'IQN-CVaR_{alpha:.2f}'
        agents[name] = IQNAgent(
            iqn_cfg_cvar, state_dim, n_actions, device=device, seed=seed + 3,
        )

    return agents


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_all(
    agents     : Dict[str, object],
    env,
    n_eval     : int  = 500,
    seed       : int  = 42,
) -> Tuple[List[Dict], Dict[str, EpisodeTracker]]:
    """
    Evaluate all agents on the given environment.

    Returns:
        all_results: list of metric dicts (one per agent)
        trackers:    dict mapping agent_name → EpisodeTracker
                     (needed for visualisation)
    """
    all_results = []
    trackers    = {}

    for name, agent in agents.items():
        print(f'  Evaluating {name}...')
        env.seed(seed)

        tracker = EpisodeTracker()
        for ep in range(n_eval):
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

        print(f'    {format_table_row(results)}')

    return all_results, trackers


# ============================================================================
# Regime-conditioned evaluation (Phase 1B only)
# ============================================================================

def evaluate_by_regime(
    agents  : Dict[str, object],
    env     : RegimeSwitchingEnv,
    n_eval  : int = 500,
    seed    : int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Evaluate agents and partition IS by the dominant volatility
    regime during each episode.

    The regime is hidden from the agent (by design). We record it
    post-hoc for analysis: does the agent behave differently when
    volatility is high? Does CVaR protection help more in stress?

    Returns:
        {agent_name: {'normal': is_array, 'stress': is_array}}
    """
    regime_results = {}

    for name, agent in agents.items():
        env.seed(seed)
        normal_is = []
        stress_is = []

        for ep in range(n_eval):
            state = env.reset()
            if hasattr(agent, 'reset') and callable(agent.reset):
                agent.reset()

            regime_counts = {'normal': 0, 'stress': 0}
            done = False
            info = {}

            while not done:
                action = agent.select_action(state, eval_mode=True)
                state, reward, done, info = env.step(action)
                # Record the hidden regime at each step
                regime_counts[env.current_regime] += 1

            is_val = info.get('implementation_shortfall', 0.0)

            # Classify episode by dominant regime
            if regime_counts['stress'] > regime_counts['normal']:
                stress_is.append(is_val)
            else:
                normal_is.append(is_val)

        regime_results[name] = {
            'normal': np.array(normal_is),
            'stress': np.array(stress_is),
        }
        n_s = len(stress_is)
        n_n = len(normal_is)
        print(f'  {name}: {n_n} normal, {n_s} stress episodes')

    return regime_results


# ============================================================================
# Copy IQN weights for CVaR variants
# ============================================================================

def share_iqn_weights(agents: Dict[str, object]) -> None:
    """
    Copy trained IQN-neutral weights to all IQN-CVaR variants.

    This implements the paper's key claim: CVaR policy is obtained
    for FREE by simply changing the τ sampling range at inference.
    No retraining needed.
    """
    neutral = agents.get('IQN-neutral')
    if neutral is None:
        return

    for name, agent in agents.items():
        if name.startswith('IQN-CVaR') and hasattr(agent, 'online_net'):
            agent.online_net.load_state_dict(
                neutral.online_net.state_dict()
            )
            agent.target_net.load_state_dict(
                neutral.target_net.state_dict()
            )
            print(f'  Copied IQN-neutral weights → {name}')


# ============================================================================
# Full experiment pipeline
# ============================================================================

def run_phase(
    env_name       : str,
    sim_config     : SimConfig,
    n_episodes     : int,
    n_eval         : int,
    eval_freq      : int,
    checkpoint_freq: int,
    seed           : int,
    results_dir    : Path,
    eval_only      : bool = False,
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Run one complete experimental phase (AC or Regime-Switching).

    Steps:
        1. Create environments (train + eval)
        2. Build all agents
        3. Train learned agents (or load from checkpoint)
        4. Share IQN-neutral weights → CVaR variants
        5. Evaluate all agents
        6. Generate figures and tables
        7. Save everything to results/
    """
    phase_dir  = results_dir / env_name
    ckpt_dir   = phase_dir / 'checkpoints'
    fig_dir    = phase_dir / 'figures'
    log_dir    = phase_dir / 'logs'

    for d in [ckpt_dir, fig_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Environments ──────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  Phase 1{"A" if env_name == "almgren_chriss" else "B"}: '
          f'{env_name.replace("_", " ").title()}')
    print(f'{"="*60}')

    if env_name == 'almgren_chriss':
        train_env = AlmgrenChrissEnv(sim_config)
        eval_env  = AlmgrenChrissEnv(sim_config)
    # else:
    #     train_env = RegimeSwitchingEnv(sim_config)
    #     eval_env  = RegimeSwitchingEnv(sim_config)
    elif env_name == 'mean_reverting':
        train_env = MeanRevertingEnv(sim_config)
        eval_env  = MeanRevertingEnv(sim_config)
    else:
        train_env = JumpDiffusionEnv(sim_config)
        eval_env  = JumpDiffusionEnv(sim_config)

    state_dim = train_env.state_dim
    n_actions = train_env.n_actions

    # ── Step 2: Agents ────────────────────────────────────────
    print('\nBuilding agents...')
    agents = build_agents(sim_config, state_dim, n_actions, seed=seed)

    # ── Step 3: Training ──────────────────────────────────────
    training_logs = {}
    learned_agents = {
        name: agent for name, agent in agents.items()
        if hasattr(agent, 'update') and hasattr(agent, 'store')
        and not isinstance(agent, (TWAPAgent, AlmgrenChrissAgent))
        and not name.startswith('IQN-CVaR')
    }

    if not eval_only:
        print(f'\nTraining {len(learned_agents)} learned agents '
              f'for {n_episodes:,} episodes each...\n')

        for name, agent in learned_agents.items():
            print(f'── Training {name} {"─"*(45 - len(name))}')
            log = train_agent(
                agent          = agent,
                env            = train_env,
                n_episodes     = n_episodes,
                eval_env       = eval_env,
                eval_freq      = eval_freq,
                n_eval         = min(n_eval, 200),   # lighter eval during training
                checkpoint_dir = ckpt_dir,
                checkpoint_freq= checkpoint_freq,
                seed           = seed,
            )
            training_logs[name] = log
            print(f'  Done in {log["wall_time"]:.0f}s '
                  f'({len(log["losses"]):,} gradient steps)\n')

        # Save training logs
        for name, log in training_logs.items():
            log_path = log_dir / f'{name}_training.json'
            serialisable = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in log.items()
            }
            # Convert numpy floats in eval_history
            for entry in serialisable.get('eval_history', []):
                for ek, ev in entry.items():
                    if isinstance(ev, (np.floating, np.integer)):
                        entry[ek] = float(ev)
            with open(log_path, 'w') as f:
                json.dump(serialisable, f, indent=2, default=str)

    elif checkpoint_path is not None:
        print(f'\nLoading checkpoints from {checkpoint_path}...')
        ckpt_p = Path(checkpoint_path)
        for name, agent in learned_agents.items():
            candidates = sorted(ckpt_p.glob(f'{name}_ep*.pt'))
            if candidates:
                agent.load(str(candidates[-1]))

    # ── Step 4: Weight sharing ────────────────────────────────
    print('\nSharing IQN-neutral weights to CVaR variants...')
    share_iqn_weights(agents)

    # ── Step 5: Final evaluation ──────────────────────────────
    print(f'\nFinal evaluation ({n_eval} episodes per agent)...')
    all_results, trackers = evaluate_all(
        agents, eval_env, n_eval=n_eval, seed=seed + 99_999,
    )

    # Save raw IS arrays for plotting
    import pickle
    is_dict = {}
    for name, tracker in trackers.items():
        is_dict[name] = np.array(tracker.is_values) * 1e4
    with open(log_dir / 'is_arrays.pkl', 'wb') as f:
        pickle.dump(is_dict, f)

    # Regime analysis (only for regime-switching env)
    # regime_results = None
    # if env_name == 'regime_switching':
    #     print('\nRegime-conditioned evaluation...')
    #     regime_results = evaluate_by_regime(
    #         agents, eval_env, n_eval=n_eval, seed=seed + 99_999,
    #     )

    # ── Step 6: Tables ────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  Results: {env_name}')
    print(f'{"="*60}')
    table = format_comparison_table(all_results, bps=True)
    print(table)

    # Save table
    with open(log_dir / 'comparison_table.txt', 'w') as f:
        f.write(table)

    # Save raw results as JSON
    results_json = []
    for r in all_results:
        serialisable = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                serialisable[k] = float(v)
            elif isinstance(v, np.ndarray):
                serialisable[k] = v.tolist()
            else:
                serialisable[k] = v
        results_json.append(serialisable)
    with open(log_dir / 'all_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    # ── Step 7: Figures ───────────────────────────────────────
    print('\nGenerating figures...')
    viz = Visualizer(save_dir=str(fig_dir), bps=True)

    # Fig 2: IS distributions
    viz.plot_is_distributions(all_results, trackers=trackers)
    print('  ✓ IS distributions')

    # Fig 3: Mean-CVaR frontier
    viz.plot_mean_cvar_frontier(all_results, alpha=0.95)
    print('  ✓ Mean-CVaR frontier')

    # Fig 4: Training curves (for each learned agent)
    for name, log in training_logs.items():
        viz_agent = Visualizer(save_dir=str(fig_dir), bps=True)
        viz_agent.plot_training_curves(log)
        # Rename output to include agent name
        for ext in ['pdf', 'png']:
            src = fig_dir / f'training_curves.{ext}'
            dst = fig_dir / f'training_curves_{name}.{ext}'
            if src.exists():
                src.rename(dst)
        print(f'  ✓ Training curves ({name})')

    # Fig 5: Inventory trajectories
    viz.plot_inventory_trajectories(
        trackers, q0=sim_config.q0, n_periods=sim_config.N,
    )
    print('  ✓ Inventory trajectories')

    # Fig 6: Quantile distribution (IQN only)
    iqn_neutral = agents.get('IQN-neutral')
    if iqn_neutral is not None:
        # Use a representative state: t*=0.5, q*=0.5, rest=0
        sample_state = np.array([0.5, 0.5, 0.0, 0.01, 0.0],
                                dtype=np.float32)
        viz.plot_quantile_distribution(iqn_neutral, sample_state)
        print('  ✓ Quantile distribution')

    # Fig 7: Regime comparison (Phase 1B only)
    regime_results = None
    if regime_results is not None:
        viz.plot_regime_comparison(regime_results)
        print('  ✓ Regime comparison')

    # Fig 8: Comparison heatmap
    viz.plot_comparison_heatmap(all_results)
    print('  ✓ Comparison heatmap')

    # Fig 10: Reward per step
    viz.plot_reward_per_step(trackers, n_periods=sim_config.N)
    print('  ✓ Reward per step')

    print(f'\nAll results saved to {phase_dir}/')

    import matplotlib.pyplot as plt
    plt.close('all')


# ============================================================================
# CLI entry point
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Phase 1: Simulated environment experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # p.add_argument('--env', type=str, default='both',
    #                choices=['ac', 'regime', 'both'],
    #                help='Which environment to run: '
    #                     'ac=Almgren-Chriss, regime=Regime-Switching, '
    #                     'both=run both sequentially')
    p.add_argument('--env', type=str, default='both',
                   choices=['ac', 'ou', 'jump', 'both'],
                   help='ac=Almgren-Chriss, ou=Mean-Reverting, jump=Jump-Diffusion, both=ac+jump')
    
    p.add_argument('--episodes', type=int, default=DEFAULT_TRAIN['n_episodes'],
                   help='Total training episodes per agent')
    p.add_argument('--eval-episodes', type=int,
                   default=DEFAULT_TRAIN['n_eval_episodes'],
                   help='Episodes per evaluation round')
    p.add_argument('--eval-freq', type=int,
                   default=DEFAULT_TRAIN['eval_freq'],
                   help='Evaluate every N training episodes')
    p.add_argument('--seed', type=int, default=DEFAULT_TRAIN['seed'],
                   help='Global random seed')
    p.add_argument('--results-dir', type=str, default='results',
                   help='Base directory for all outputs')
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training, load checkpoints and evaluate')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Checkpoint directory for --eval-only')
    return p.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    sim_config  = SimConfig(**DEFAULT_SIM_CONFIG)

    # Save config for reproducibility
    (results_dir / 'logs').mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'logs' / 'sim_config.json', 'w') as f:
        json.dump(asdict(sim_config), f, indent=2)

    envs_to_run = []
    # if args.env in ('ac', 'both'):
    #     envs_to_run.append('almgren_chriss')
    # if args.env in ('regime', 'both'):
    #     envs_to_run.append('regime_switching')

    if args.env in ('ac', 'both'):
        envs_to_run.append('almgren_chriss')
    if args.env in ('ou','both'):
        envs_to_run.append('mean_reverting')
    if args.env in ('jump', 'both'):
        envs_to_run.append('jump_diffusion')

    for env_name in envs_to_run:
        run_phase(
            env_name        = env_name,
            sim_config      = sim_config,
            n_episodes      = args.episodes,
            n_eval          = args.eval_episodes,
            eval_freq       = args.eval_freq,
            checkpoint_freq = DEFAULT_TRAIN['checkpoint_freq'],
            seed            = args.seed,
            results_dir     = results_dir,
            eval_only       = args.eval_only,
            checkpoint_path = args.checkpoint,
        )

    print('\n' + '=' * 60)
    print('  Phase 1 complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()