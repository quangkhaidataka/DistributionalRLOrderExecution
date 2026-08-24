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
# Resumable-state primitive (Design-v2 B1)
# ============================================================================
#
# These two module-level helpers are the primitive the Design-v2 staged
# runners (B2) use to make a split `--episodes 0:b` followed by `b:c` run
# reproduce a single `0:c` run. Equivalence requires restoring, at the split
# boundary, everything that steers the stochastic trajectory:
#
#   1. the agent's learned parameters + optimizer state (online/target nets),
#   2. the agent's `_step` counter (drives ε-decay AND target-net updates),
#   3. the GLOBAL RNG state (torch + numpy, and torch.cuda if present) — the
#      ε-greedy coin flips, τ sampling, replay sampling, and env noise all
#      draw from these generators.
#
# Everything is bundled into ONE file via `torch.save`.
#
# LIMITATION — the replay BUFFER is intentionally NOT persisted here. It can be
# large and is not needed for the staged v2 split, where each job either trains
# one agent fully or resumes from this state with a freshly warmed buffer.
# Because the warm buffer's contents differ from an uninterrupted run's buffer,
# resumed updates are only *equivalent-in-distribution* (same RNG streams, same
# weights, same step clock), not byte-identical, until the buffer re-fills.
# torch.save / torch.get_rng_state / torch.set_rng_state do NOT touch the
# numpy↔torch bridge, so they are safe under this repo's numpy-2.x + torch-2.2.2
# pairing (never call `tensor.numpy()`).

def _capture_agent_state(agent) -> Dict[str, Any]:
    """Capture a learned agent's serialisable state_dicts + step counter.

    Works for every learned agent in this repo (IQNAgent and the _DeepRLBase
    subclasses all expose online_net / target_net / optimizer / _step). Missing
    attributes are skipped so rule-based agents degrade gracefully.
    """
    state: Dict[str, Any] = {}
    if hasattr(agent, 'online_net'):
        state['online_net'] = agent.online_net.state_dict()
    if hasattr(agent, 'target_net'):
        state['target_net'] = agent.target_net.state_dict()
    if hasattr(agent, 'optimizer'):
        state['optimizer'] = agent.optimizer.state_dict()
    if hasattr(agent, '_step'):
        state['_step'] = agent._step
    return state


def save_resumable_state(agent, path, extra: Optional[dict] = None) -> None:
    """Persist agent + RNG state so a split training run resumes equivalently.

    Writes a single `torch.save` bundle containing the agent's own checkpoint
    (network + optimizer state_dicts and the `_step` counter) and the global
    RNG state (torch, numpy, and torch.cuda when available), plus any caller
    `extra` dict (e.g. `{'episode': b}`) for bookkeeping.

    This is the primitive the Design-v2 staged runners (B2) use for
    `--episodes a:b` resume. The replay buffer is NOT saved (see module note):
    resume is equivalent-in-distribution once the warm buffer re-fills, not
    byte-identical. Uses torch.get_rng_state / np.random.get_state, which do
    not hit the numpy↔torch bridge, so it is safe under numpy-2.x + torch-2.2.2.

    Args:
        agent: any learned agent (online_net/target_net/optimizer/_step).
        path : destination file for the single bundled checkpoint.
        extra: optional JSON-ish dict round-tripped back by load_resumable_state.
    """
    import torch  # local import: keep module import-light + torch-optional

    rng_state: Dict[str, Any] = {
        'torch': torch.get_rng_state(),        # CPU ByteTensor (no numpy bridge)
        'numpy': np.random.get_state(),        # tuple; pickled by torch.save
    }
    if torch.cuda.is_available():
        rng_state['torch_cuda'] = torch.cuda.get_rng_state_all()

    bundle = {
        'format_version': 1,
        'agent_state'   : _capture_agent_state(agent),
        'rng_state'     : rng_state,
        'extra'         : dict(extra) if extra else {},
    }
    torch.save(bundle, path)


def load_resumable_state(agent, path) -> dict:
    """Restore agent + RNG state saved by `save_resumable_state`; return `extra`.

    Loads the bundle onto CPU (map_location='cpu') — `load_state_dict` then
    copies weights into the agent's on-device parameters in place, and the
    torch RNG ByteTensor must live on CPU for `torch.set_rng_state`. Restores
    the agent's networks/optimizer, the `_step` counter, and the global torch/
    numpy(/cuda) RNG streams, so subsequent draws continue as if uninterrupted.

    Args:
        agent: the agent instance to restore into (same architecture as saved).
        path : bundle written by save_resumable_state.

    Returns:
        The `extra` dict passed at save time (or {} if none).
    """
    import torch  # local import: keep module import-light + torch-optional

    # CPU load: the torch RNG ByteTensor must be CPU for set_rng_state, and
    # load_state_dict cross-copies CPU weights into on-device params fine.
    bundle = torch.load(path, map_location='cpu')

    agent_state = bundle.get('agent_state', {})
    if hasattr(agent, 'online_net') and 'online_net' in agent_state:
        agent.online_net.load_state_dict(agent_state['online_net'])
    if hasattr(agent, 'target_net') and 'target_net' in agent_state:
        agent.target_net.load_state_dict(agent_state['target_net'])
    if hasattr(agent, 'optimizer') and 'optimizer' in agent_state:
        agent.optimizer.load_state_dict(agent_state['optimizer'])
    if hasattr(agent, '_step') and '_step' in agent_state:
        agent._step = agent_state['_step']

    rng_state = bundle.get('rng_state', {})
    if 'torch' in rng_state:
        torch.set_rng_state(rng_state['torch'])
    if 'numpy' in rng_state:
        np.random.set_state(rng_state['numpy'])
    if 'torch_cuda' in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state['torch_cuda'])

    return bundle.get('extra', {})


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
    # checkpoint_freq is config-driven (this field is the single source of
    # truth) — nothing hard-codes the interval in the loop below. Default kept
    # at 5_000 to preserve legacy behaviour; the Design-v2 runner (B2) passes
    # 2_000 explicitly rather than relying on a code-level constant.
    checkpoint_freq  : int   = 5_000     # save every N episodes
    checkpoint_dir   : str   = 'results/checkpoints'

    # ── Replay buffer (record / passthrough) ──────────────────
    # Recorded here so the training-run config captures the intended replay
    # capacity in one place. NOTE: the Trainer does NOT build the buffer —
    # each learned agent constructs its own ReplayBuffer from AgentConfig/
    # DeepRLConfig.replay_capacity. This field is a passthrough/record for the
    # Design-v2 runner and provenance; it does not itself resize any buffer.
    replay_capacity  : int   = 100_000   # intended replay capacity (record only)

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
        start_episode : int = 1,
    ) -> TrainingLog:
        """
        Run the full training loop.

        Args:
            agent:    any agent implementing the project's interface
            env:      training environment (reset/step/seed)
            eval_env: separate evaluation environment (optional)
            start_episode: 1-indexed episode to begin the loop at. Defaults to
                1, which reproduces the legacy behaviour byte-for-byte
                (`range(1, n_episodes+1)`). Passing a value > 1 lets a
                Design-v2 staged runner (B2) resume a split `--episodes a:b`
                run at episode `a` after restoring state via
                `load_resumable_state`; the episode-numbered checkpoints and
                periodic eval/log cadence then line up with a single run.

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
        # start_episode defaults to 1 → range(1, n_episodes+1), identical to
        # the legacy loop. `ep` is pre-seeded so the post-loop bookkeeping
        # (log.total_episodes = ep) is well-defined even if the range is empty.
        ep = start_episode - 1
        for ep in range(start_episode, cfg.n_episodes + 1):
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
            # Estimate total UPDATE-STEPS for schedule length. The `* 15` is a
            # heuristic steps/episode factor (valid for horizons N≈10-20); it is
            # an ESTIMATE used only to size the ε schedule, NOT an episode budget
            # and NOT a hard cap on training length.
            estimated_steps = cfg.n_episodes * 15  # ~15 steps/episode for N≈10-20
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
            # Same heuristic estimate as above (~15 update-steps/episode for
            # N≈10-20) — used only to size the LR schedule, not an episode budget.
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