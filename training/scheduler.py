"""
training/scheduler.py
---------------------
Schedule controllers for epsilon-greedy exploration and learning rate.

These are standalone objects that the Trainer queries at each step,
decoupling scheduling logic from agent internals. Agents already
have built-in epsilon decay (via _get_epsilon), but the Trainer
can optionally override with these more flexible schedulers.

Supported schedules:
    EpsilonScheduler:
        - Linear decay (default, matches Dabney et al. 2018)
        - Exponential decay (faster convergence for small problems)
        - Cosine annealing (smooth, avoids sudden policy changes)

    LRScheduler:
        - Constant (baseline)
        - Linear warmup + cosine decay (standard in deep learning)
        - Step decay (reduce by factor every N steps)

Design choices:
    - Pure Python (no torch.optim.lr_scheduler dependency) so the
      same scheduler works for any optimizer or RL library
    - Each scheduler is a callable: scheduler(step) → value
    - Immutable after construction: all parameters set at init
    - The Trainer calls scheduler.step() and applies the returned
      value to the agent's epsilon or optimizer LR

Usage:
    # Epsilon schedule
    eps_sched = EpsilonScheduler(
        strategy='linear', start=1.0, end=0.01, decay_steps=10_000,
    )
    for step in range(total_steps):
        epsilon = eps_sched.value(step)

    # LR schedule
    lr_sched = LRScheduler(
        strategy='cosine_warmup', base_lr=1e-4, min_lr=1e-6,
        warmup_steps=1000, total_steps=100_000,
    )
    for step in range(total_steps):
        lr = lr_sched.value(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


# ============================================================================
# Epsilon scheduler
# ============================================================================

@dataclass
class EpsilonConfig:
    """Configuration for epsilon-greedy exploration schedule."""
    strategy    : str   = 'linear'     # 'linear', 'exponential', 'cosine'
    start       : float = 1.0          # initial ε
    end         : float = 0.01         # final ε (floor)
    decay_steps : int   = 10_000       # steps over which to decay


class EpsilonScheduler:
    """
    Epsilon-greedy exploration schedule.

    Controls the exploration-exploitation tradeoff during training.
    Early training: high ε → random exploration to fill replay buffer.
    Late training:  low ε  → greedy exploitation of learned Q-values.

    Strategy details:
        linear:
            ε(t) = start + (end - start) · min(t / decay_steps, 1)
            Standard in DQN literature. Simple, predictable.
            Used by Dabney et al. (2018) and Ning et al. (2021).

        exponential:
            ε(t) = end + (start - end) · exp(-t / (decay_steps / 5))
            Faster initial decay, slower convergence to floor.
            Good for small state spaces where exploitation pays early.

        cosine:
            ε(t) = end + ½(start - end)(1 + cos(π · min(t/decay_steps, 1)))
            Smooth annealing, no sharp transitions.
            Avoids sudden policy changes that can destabilise training.

    Why use this instead of the agent's built-in epsilon?
        - Flexibility: change schedule without modifying agent code
        - Consistency: same scheduler object used across all agents
        - Logging: Trainer can log ε at each step for diagnostics
    """

    def __init__(self, cfg: EpsilonConfig = None, **kwargs):
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = EpsilonConfig(**kwargs)

    def value(self, step: int) -> float:
        """
        Compute ε at the given training step.

        Args:
            step: current global training step (0-indexed)

        Returns:
            epsilon value in [self.cfg.end, self.cfg.start]
        """
        t     = step
        start = self.cfg.start
        end   = self.cfg.end
        T     = max(self.cfg.decay_steps, 1)

        if self.cfg.strategy == 'linear':
            progress = min(t / T, 1.0)
            return start + progress * (end - start)

        elif self.cfg.strategy == 'exponential':
            # Time constant: ε reaches ~1% of (start-end) at t ≈ decay_steps
            tau = T / 5.0
            return end + (start - end) * math.exp(-t / tau)

        elif self.cfg.strategy == 'cosine':
            progress = min(t / T, 1.0)
            return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))

        else:
            raise ValueError(f'Unknown epsilon strategy: {self.cfg.strategy}')

    def __repr__(self) -> str:
        c = self.cfg
        return (f'EpsilonScheduler({c.strategy}, '
                f'{c.start}→{c.end} over {c.decay_steps} steps)')


# ============================================================================
# Learning rate scheduler
# ============================================================================

@dataclass
class LRConfig:
    """Configuration for learning rate schedule."""
    strategy    : str   = 'constant'     # 'constant', 'cosine_warmup', 'step_decay'
    base_lr     : float = 1e-4           # initial / peak learning rate
    min_lr      : float = 1e-6           # floor LR (for cosine/step)
    warmup_steps: int   = 0              # linear warmup period
    total_steps : int   = 100_000        # total training steps
    step_size   : int   = 30_000         # for step_decay: decay every N steps
    step_gamma  : float = 0.5            # for step_decay: multiply LR by this


class LRScheduler:
    """
    Learning rate schedule controller.

    Applied externally by the Trainer — modifies optimizer LR
    at each step without coupling to torch.optim.lr_scheduler.

    Strategy details:
        constant:
            LR(t) = base_lr
            Simplest baseline. Fine for short training runs.

        cosine_warmup:
            Warmup phase (t < warmup_steps):
                LR(t) = base_lr · (t / warmup_steps)
            Cosine phase (t >= warmup_steps):
                LR(t) = min_lr + ½(base_lr - min_lr)(1 + cos(π · progress))
                where progress = (t - warmup_steps) / (total_steps - warmup_steps)

            The warmup prevents large initial gradient steps when the
            network is randomly initialised (common in transformer training,
            also helpful for RL with LayerNorm).

        step_decay:
            LR(t) = max(base_lr · γ^⌊t / step_size⌋, min_lr)
            Classic staircase decay. Coarse-grained but robust.

    Why schedule LR in RL?
        Financial RL has non-stationary reward distributions (the
        agent's own behaviour changes the effective data distribution).
        Starting with a higher LR and decaying helps:
        - Early: fast initial learning from diverse exploration data
        - Late:  fine-grained refinement near the optimal policy
        - Warmup: prevents catastrophic early updates with bad
          random initialisation + financial reward noise
    """

    def __init__(self, cfg: LRConfig = None, **kwargs):
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = LRConfig(**kwargs)

    def value(self, step: int) -> float:
        """
        Compute learning rate at the given step.

        Args:
            step: current global training step

        Returns:
            learning rate value
        """
        t       = step
        base_lr = self.cfg.base_lr
        min_lr  = self.cfg.min_lr
        warmup  = self.cfg.warmup_steps
        total   = max(self.cfg.total_steps, 1)

        if self.cfg.strategy == 'constant':
            return base_lr

        elif self.cfg.strategy == 'cosine_warmup':
            # Linear warmup
            if t < warmup:
                return base_lr * (t / max(warmup, 1))

            # Cosine decay
            decay_steps = max(total - warmup, 1)
            progress    = min((t - warmup) / decay_steps, 1.0)
            return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        elif self.cfg.strategy == 'step_decay':
            n_decays = t // max(self.cfg.step_size, 1)
            lr = base_lr * (self.cfg.step_gamma ** n_decays)
            return max(lr, min_lr)

        else:
            raise ValueError(f'Unknown LR strategy: {self.cfg.strategy}')

    def apply(self, optimizer, step: int) -> float:
        """
        Compute LR and apply it to all optimizer param groups.

        Convenience method — equivalent to:
            lr = scheduler.value(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        Args:
            optimizer: torch.optim.Optimizer
            step:      current global step

        Returns:
            The applied learning rate
        """
        lr = self.value(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def __repr__(self) -> str:
        c = self.cfg
        return (f'LRScheduler({c.strategy}, '
                f'base={c.base_lr}, min={c.min_lr}, '
                f'warmup={c.warmup_steps}, total={c.total_steps})')