"""
training/trainer.py
-------------------
Centralised training loop for all agents and environments.

The Trainer owns the train-eval-checkpoint-log cycle. It works
with ANY combination of agent and environment that follow the
project's interface contract:

    Agent interface:
        agent.select_action(state, eval_mode=False) → int
        agent.store(state, action, reward, next_state, done) → None
        agent.update()  → Optional[float]
        agent.save(path) / agent.load(path)
        agent.name → str

    Environment interface:
        env.reset()  → state (np.ndarray)
        env.step(action) → (state, reward, done, info)
        env.seed(seed)

    Rule-based agents (TWAP, AC) also work: store() and update()
    are no-ops, so the Trainer just collects evaluation metrics.

Design choices:
    - Single Trainer class, not one-per-agent:
      Avoids duplicating the training loop across experiment scripts.
      run_simulation.py and run_lobster.py instantiate a Trainer
      and call trainer.train(agent, env).

    - TrainerConfig dataclass holds ALL training hyperparameters:
      n_episodes, eval_freq, checkpoint_freq, etc.
      This is the single source of truth — no magic numbers.

    - TrainingLog dataclass accumulates all training data:
      losses, rewards, IS values, evaluation snapshots.
      Serialisable to JSON for post-hoc analysis.

    - Optional LR and epsilon schedulers (from scheduler.py):
      If provided, the Trainer applies them at each step.
      If not, the agent's internal schedules are used unchanged.

    - Early stopping on evaluation plateau:
      If CVaR_0.95 hasn't improved for `patience` eval rounds,
      training stops and the best checkpoint is restored.

Usage:
    from training.trainer import Trainer, TrainerConfig

    cfg = TrainerConfig(n_episodes=15_000, eval_freq=1_000)
    trainer = Trainer(cfg)
    log = trainer.train(agent, train_env, eval_env=eval_env)

    # log.losses, log.episode_rewards, log.eval_history, ...
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import numpy as np

from evaluation.metrics import MetricsSuite, EpisodeTracker, format_table_row
from training.scheduler import (
    EpsilonScheduler, EpsilonConfig,
    LRScheduler, LRConfig,
)


# ============================================================================
# Training configuration
# ============================================================================

@dataclass
class TrainerConfig:
    """
    All parameters controlling the training loop.

    Separating training config from agent config keeps concerns clean:
    - AgentConfig: network size, tau samples, loss function details
    - TrainerConfig: how many episodes, when to evaluate, where to save
    """
    # ── Training length ───────────────────────────────────────
    n_episodes       : int   = 15_000    # total training episodes
    max_steps_per_ep : int   = 200       # safety cap (prevents infinite loops)

    # ── Evaluation ────────────────────────────────────────────
    eval_freq        : int   = 1_000     # evaluate every N episodes
    n_eval_episodes  : int   = 200       # episodes per mid-training eval
    n_final_eval     : int   = 500       # episodes for final evaluation

    # ── Checkpointing ─────────────────────────────────────────
    checkpoint_freq  : int   = 5_000     # save every N episodes
    checkpoint_dir   : str   = 'results/checkpoints'

    # ── Logging ───────────────────────────────────────────────
    log_freq         : int   = 100       # print progress every N episodes
    log_dir          : str   = 'results/logs'

    # ── Reproducibility ───────────────────────────────────────
    seed             : int   = 42

    # ── Early stopping ────────────────────────────────────────
    early_stopping   : bool  = False
    patience         : int   = 5         # eval rounds without improvement
    monitor_metric   : str   = 'CVaR_0.95_bps'   # lower = better
    # 'CVaR_0.95_bps' means we stop when CVaR₉₅ stops decreasing.
    # This is the key metric for the paper.

    # ── Scheduling (optional overrides) ───────────────────────
    use_lr_schedule  : bool  = False
    lr_strategy      : str   = 'cosine_warmup'
    lr_warmup_steps  : int   = 1_000
    lr_min           : float = 1e-6

    use_eps_schedule : bool  = False     # override agent's epsilon
    eps_strategy     : str   = 'linear'


# ============================================================================
# Training log
# ============================================================================

@dataclass
class TrainingLog:
    """
    Accumulates all data from a training run.

    Designed to be JSON-serialisable for post-hoc analysis.
    The experiment scripts save this to results/logs/.
    """
    losses          : List[float]              = field(default_factory=list)
    episode_rewards : List[float]              = field(default_factory=list)
    episode_is      : List[float]              = field(default_factory=list)
    episode_lengths : List[int]                = field(default_factory=list)
    eval_history    : List[Dict[str, Any]]     = field(default_factory=list)
    epsilon_history : List[float]              = field(default_factory=list)
    lr_history      : List[float]              = field(default_factory=list)
    wall_time       : float                    = 0.0
    total_steps     : int                      = 0
    total_episodes  : int                      = 0
    best_metric     : float                    = float('inf')
    best_episode    : int                      = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dict."""
        d = asdict(self)
        # numpy types → Python native
        for key, val in d.items():
            if isinstance(val, list) and val and isinstance(val[0], (np.floating, np.integer)):
                d[key] = [float(v) for v in val]
            elif isinstance(val, (np.floating, np.integer)):
                d[key] = float(val)
        # Sanitise eval_history entries
        clean_eval = []
        for entry in d.get('eval_history', []):
            clean = {}
            for k, v in entry.items():
                if isinstance(v, (np.floating, np.integer)):
                    clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                else:
                    clean[k] = v
            clean_eval.append(clean)
        d['eval_history'] = clean_eval
        return d

    def save(self, path: str) -> None:
        """Save log as JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> 'TrainingLog':
        """Load log from JSON."""
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ============================================================================
# Callback protocol
# ============================================================================

class TrainerCallback:
    """
    Optional callback hooks for custom behaviour during training.

    Subclass and override any method. Useful for:
        - Custom logging (wandb, tensorboard)
        - Dynamic hyperparameter tuning
        - Visualisation during training
    """

    def on_train_start(self, trainer: 'Trainer', agent, env) -> None:
        pass

    def on_episode_end(
        self,
        trainer  : 'Trainer',
        agent,
        episode  : int,
        reward   : float,
        is_val   : float,
        info     : Dict,
    ) -> None:
        pass

    def on_eval_end(
        self,
        trainer : 'Trainer',
        agent,
        episode : int,
        results : Dict[str, float],
    ) -> None:
        pass

    def on_train_end(self, trainer: 'Trainer', agent, log: TrainingLog) -> None:
        pass


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """
    Centralised training loop for any agent + environment pair.

    The Trainer is stateless between calls to train() — all state
    is captured in the returned TrainingLog. This makes it safe
    to train multiple agents sequentially with the same Trainer.

    Training loop structure (per episode):
        1. env.reset() → initial state
        2. Loop until done:
           a. agent.select_action(state) → action
           b. env.step(action) → (next_state, reward, done, info)
           c. agent.store(state, action, reward, next_state, done)
           d. agent.update() → loss or None
           e. (optional) apply LR/epsilon schedule
        3. Log episode reward, IS, length
        4. (periodic) Evaluate on eval_env
        5. (periodic) Save checkpoint
        6. (optional) Check early stopping

    Args:
        cfg:       TrainerConfig with all training parameters
        callbacks: list of TrainerCallback instances
    """

    def __init__(
        self,
        cfg       : TrainerConfig = None,
        callbacks : Optional[List[TrainerCallback]] = None,
    ):
        self.cfg = cfg or TrainerConfig()
        self.callbacks = callbacks or []

        # Build schedulers
        self._eps_scheduler = None
        self._lr_scheduler  = None

    def train(
        self,
        agent,
        env,
        eval_env : Optional[Any] = None,
    ) -> TrainingLog:
        """
        Run the full training loop.

        Args:
            agent:    any agent implementing the project's interface
            env:      training environment (reset/step/seed)
            eval_env: separate evaluation environment (optional)

        Returns:
            TrainingLog with all training data
        """
        cfg = self.cfg
        log = TrainingLog()

        # ── Setup ─────────────────────────────────────────────
        env.seed(cfg.seed)
        if eval_env is not None:
            eval_env.seed(cfg.seed + 10_000)

        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Build schedulers if configured
        self._build_schedulers(agent)

        # Early stopping state
        best_metric     = float('inf')
        best_ckpt_path  = None
        patience_counter = 0

        t_start     = time.time()
        global_step = 0

        # Notify callbacks
        for cb in self.callbacks:
            cb.on_train_start(self, agent, env)

        agent_name = getattr(agent, 'name', type(agent).__name__)
        print(f'[Trainer] Starting training: {agent_name}')
        print(f'  Episodes: {cfg.n_episodes:,} | '
              f'Eval every {cfg.eval_freq} | '
              f'Checkpoint every {cfg.checkpoint_freq}')

        # ── Main training loop ────────────────────────────────
        for ep in range(1, cfg.n_episodes + 1):
            state     = env.reset()
            ep_reward = 0.0
            ep_steps  = 0
            done      = False
            info      = {}

            # Reset rule-based agents
            if hasattr(agent, 'reset') and callable(agent.reset):
                agent.reset()

            while not done and ep_steps < cfg.max_steps_per_ep:
                # ── Action selection ──────────────────────────
                action = agent.select_action(state)

                # ── Environment step ──────────────────────────
                next_state, reward, done, info = env.step(action)

                # ── Store transition ──────────────────────────
                agent.store(state, action, reward, next_state, done)

                # ── Gradient update ───────────────────────────
                loss = agent.update()
                if loss is not None:
                    log.losses.append(loss)

                # ── Apply LR schedule ─────────────────────────
                if self._lr_scheduler is not None and hasattr(agent, 'optimizer'):
                    lr = self._lr_scheduler.apply(agent.optimizer, global_step)
                    if ep_steps == 0 and ep % cfg.log_freq == 0:
                        log.lr_history.append(lr)

                ep_reward += reward
                state      = next_state
                ep_steps  += 1
                global_step += 1

            # ── Episode bookkeeping ───────────────────────────
            is_val = info.get('implementation_shortfall')
            if is_val is None:
                is_val = -ep_reward

            log.episode_rewards.append(ep_reward)
            log.episode_is.append(is_val)
            log.episode_lengths.append(ep_steps)

            # Record epsilon
            if self._eps_scheduler is not None:
                eps_val = self._eps_scheduler.value(global_step)
                log.epsilon_history.append(eps_val)
            elif hasattr(agent, 'epsilon'):
                log.epsilon_history.append(agent.epsilon)

            # Callbacks
            for cb in self.callbacks:
                cb.on_episode_end(self, agent, ep, ep_reward, is_val, info)

            # ── Periodic logging ──────────────────────────────
            if ep % cfg.log_freq == 0:
                recent_is  = log.episode_is[-cfg.log_freq:]
                recent_rew = log.episode_rewards[-cfg.log_freq:]
                mean_is  = np.mean(recent_is) * 10_000
                mean_rew = np.mean(recent_rew)
                eps_str  = f'{log.epsilon_history[-1]:.3f}' if log.epsilon_history else 'N/A'
                elapsed  = time.time() - t_start
                loss_str = f'{np.mean(log.losses[-100:]):.6f}' if log.losses else 'N/A'

                print(f'  ep={ep:>6d} | '
                      f'IS={mean_is:>7.2f} bps | '
                      f'rew={mean_rew:>+8.5f} | '
                      f'loss={loss_str} | '
                      f'ε={eps_str} | '
                      f'{elapsed:.0f}s')

            # ── Periodic evaluation ───────────────────────────
            if eval_env is not None and ep % cfg.eval_freq == 0:
                eval_results = self._evaluate(
                    agent, eval_env,
                    n_eval = cfg.n_eval_episodes,
                    seed   = cfg.seed + ep,
                )
                eval_results['episode']     = ep
                eval_results['total_steps'] = global_step
                log.eval_history.append(eval_results)

                print(f'  ── EVAL ep={ep}: {format_table_row(eval_results)}')

                # Callbacks
                for cb in self.callbacks:
                    cb.on_eval_end(self, agent, ep, eval_results)

                # ── Early stopping check ──────────────────────
                if cfg.early_stopping:
                    current_metric = eval_results.get(cfg.monitor_metric, float('inf'))

                    if current_metric < best_metric:
                        best_metric      = current_metric
                        patience_counter = 0
                        log.best_metric  = best_metric
                        log.best_episode = ep
                        # Save best checkpoint
                        best_ckpt_path = ckpt_dir / f'{agent_name}_best.pt'
                        if hasattr(agent, 'save'):
                            agent.save(str(best_ckpt_path))
                    else:
                        patience_counter += 1

                    if patience_counter >= cfg.patience:
                        print(f'  ── EARLY STOPPING at ep={ep} '
                              f'(no improvement in {cfg.monitor_metric} '
                              f'for {cfg.patience} eval rounds)')
                        # Restore best checkpoint
                        if best_ckpt_path is not None and hasattr(agent, 'load'):
                            agent.load(str(best_ckpt_path))
                            print(f'  Restored best checkpoint from ep={log.best_episode}')
                        break

            # ── Periodic checkpoint ───────────────────────────
            if ep % cfg.checkpoint_freq == 0 and hasattr(agent, 'save'):
                ckpt_path = ckpt_dir / f'{agent_name}_ep{ep}.pt'
                agent.save(str(ckpt_path))

        # ── Training complete ─────────────────────────────────
        log.wall_time      = time.time() - t_start
        log.total_steps    = global_step
        log.total_episodes = ep

        # Final evaluation
        if eval_env is not None:
            print(f'\n  Final evaluation ({cfg.n_final_eval} episodes)...')
            final_results = self._evaluate(
                agent, eval_env,
                n_eval = cfg.n_final_eval,
                seed   = cfg.seed + 999_999,
            )
            final_results['episode'] = ep
            final_results['is_final'] = True
            log.eval_history.append(final_results)
            print(f'  FINAL: {format_table_row(final_results)}')

        # Save training log
        log_path = log_dir / f'{agent_name}_training_log.json'
        log.save(str(log_path))

        # Callbacks
        for cb in self.callbacks:
            cb.on_train_end(self, agent, log)

        print(f'[Trainer] {agent_name} done: '
              f'{log.total_episodes:,} episodes, '
              f'{log.total_steps:,} steps, '
              f'{log.wall_time:.0f}s')

        return log

    # ------------------------------------------------------------------
    # Internal: evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        agent,
        eval_env,
        n_eval: int,
        seed: int,
    ) -> Dict[str, float]:
        """Run evaluation episodes and return metrics dict."""
        suite = MetricsSuite(n_eval=n_eval, seed=seed)
        return suite.evaluate(agent, eval_env)

    # ------------------------------------------------------------------
    # Internal: build schedulers
    # ------------------------------------------------------------------

    def _build_schedulers(self, agent) -> None:
        """Build epsilon and LR schedulers from config."""
        cfg = self.cfg

        if cfg.use_eps_schedule:
            # Estimate total steps for schedule length
            estimated_steps = cfg.n_episodes * 15  # ~15 steps/episode for N=10
            eps_cfg = EpsilonConfig(
                strategy    = cfg.eps_strategy,
                start       = getattr(agent, 'cfg', None) and agent.cfg.epsilon_start or 1.0,
                end         = getattr(agent, 'cfg', None) and agent.cfg.epsilon_end or 0.01,
                decay_steps = getattr(agent, 'cfg', None) and agent.cfg.epsilon_decay_steps or estimated_steps,
            )
            self._eps_scheduler = EpsilonScheduler(eps_cfg)
            print(f'  Epsilon schedule: {self._eps_scheduler}')
        else:
            self._eps_scheduler = None

        if cfg.use_lr_schedule and hasattr(agent, 'optimizer'):
            # Get base LR from agent's optimizer
            base_lr = agent.optimizer.param_groups[0]['lr']
            estimated_steps = cfg.n_episodes * 15
            lr_cfg = LRConfig(
                strategy     = cfg.lr_strategy,
                base_lr      = base_lr,
                min_lr       = cfg.lr_min,
                warmup_steps = cfg.lr_warmup_steps,
                total_steps  = estimated_steps,
            )
            self._lr_scheduler = LRScheduler(lr_cfg)
            print(f'  LR schedule: {self._lr_scheduler}')
        else:
            self._lr_scheduler = None


# ============================================================================
# Convenience: train multiple agents
# ============================================================================

def train_all(
    agents     : Dict[str, Any],
    env,
    eval_env   : Optional[Any] = None,
    cfg        : TrainerConfig = None,
    skip_rule_based: bool = True,
) -> Dict[str, TrainingLog]:
    """
    Train all agents sequentially with the same Trainer config.

    Rule-based agents (agents without an 'update' that returns a loss,
    i.e., TWAP and AC) are skipped by default since they don't learn.
    Set skip_rule_based=False to still run them through the loop
    (useful for collecting baseline evaluation metrics).

    Args:
        agents:          dict mapping name → agent
        env:             training environment
        eval_env:        evaluation environment (optional)
        cfg:             TrainerConfig (shared across all agents)
        skip_rule_based: if True, skip agents that don't learn

    Returns:
        dict mapping name → TrainingLog
    """
    if cfg is None:
        cfg = TrainerConfig()

    trainer = Trainer(cfg)
    logs    = {}

    for name, agent in agents.items():
        # Check if agent is learnable
        # An agent is "learned" if it has BOTH store() and a neural
        # network (online_net or replay buffer). Rule-based agents
        # like TWAP/AC have store() as a no-op but no network.
        has_network = (
            hasattr(agent, 'online_net')   # IQN, QR-DQN
            or hasattr(agent, 'replay')    # DQN, DDQN
        )
        is_learned = (
            hasattr(agent, 'store')
            and hasattr(agent, 'update')
            and has_network
        )

        if skip_rule_based and not is_learned:
            print(f'[train_all] Skipping {name} (rule-based)')
            continue

        print(f'\n{"═"*60}')
        print(f'  Training: {name}')
        print(f'{"═"*60}')

        log = trainer.train(agent, env, eval_env=eval_env)
        logs[name] = log

    return logs