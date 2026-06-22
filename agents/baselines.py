"""
agents/baselines.py
--------------------
All benchmark agents for comparison against IQN in the paper.

Hierarchy
─────────
BaseAgent  (abstract)
  ├── TWAPAgent          — time-weighted average price schedule
  ├── AlmgrenChrissAgent — analytical mean-variance optimal schedule
  ├── DQNAgent           — Deep Q-Network (Mnih et al. 2015)
  └── DDQNAgent          — Double DQN   (Van Hasselt et al. 2016)
  └── QRDQNAgent         — Quantile-Regression DQN (Dabney et al. 2017)

Paper role of each benchmark
─────────────────────────────
TWAP            : universal industry baseline. Every execution paper
                  measures improvement vs TWAP.
                  Reference: Bertsimas & Lo (1998), Almgren & Chriss (2001)

AlmgrenChriss   : the gold-standard analytical benchmark for academic
                  execution papers. Shows what optimal mean-variance
                  control achieves under Gaussian assumptions.
                  Reference: Almgren & Chriss (2001), J. Risk 3(2):5-39

DQN             : first deep RL baseline (Mnih et al. 2015). Establishes
                  that RL can beat rule-based methods.
                  Reference: Nevmyvaka et al. (2006), Ning et al. (2021)

DDQN            : the direct predecessor to IQN in execution literature.
                  Ning et al. (2021) use DDQN as their main method.
                  This is the most important comparison for the paper.
                  Reference: Van Hasselt et al. (2016), Ning et al. (2021)

QR-DQN          : ablation step between DDQN and IQN.
                  Learns fixed discrete quantiles — proves the benefit
                  of distributional RL before adding continuous τ.
                  Reference: Dabney et al. (2017) "Distributional RL
                             with Quantile Regression", AAAI 2018

Ablation chain for paper Table 3:
    TWAP → AC → DQN → DDQN → QR-DQN → IQN-neutral → IQN-CVaR_0.9 → IQN-CVaR_0.95

Interface contract (identical for ALL agents):
    agent.select_action(state, eval_mode)  → int in {0,1,2,3,4}
    agent.store(s, a, r, s', done)         → None  (no-op for rule-based)
    agent.update()                         → float | None
    agent.name                             → str
    agent.save(path) / agent.load(path)    → None (no-op for rule-based)

Tensor shape conventions (consistent with iqn_agent.py):
    B  = batch_size
    A  = n_actions = 5
    N  = n_quantiles (QR-DQN specific)
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.base_env import EnvConfig, ACTION_FRACS
from training.replay_buffer import ReplayBuffer, ReplayConfig


# ============================================================================
# Abstract base agent
# ============================================================================

class BaseAgent(ABC):
    """
    Interface contract shared by ALL agents (rule-based and learned).

    Having a single interface means:
    - The training loop and evaluator never branch on agent type
    - Paper results tables are produced by the same evaluation code
    - Ablation is just swapping one agent for another
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Return action index in {0, 1, 2, 3, 4}."""
        ...

    def store(self, state, action, reward, next_state, done) -> None:
        """Store transition. Rule-based agents ignore this."""
        pass

    def update(self) -> Optional[float]:
        """Gradient update. Rule-based agents return None."""
        return None

    def save(self, path: str) -> None:
        """Save checkpoint. Rule-based agents are stateless."""
        pass

    def load(self, path: str) -> None:
        """Load checkpoint. Rule-based agents are stateless."""
        pass

    @property
    @abstractmethod
    def name(self) -> str: ...


# ============================================================================
# 1. TWAP Agent
# ============================================================================

class TWAPAgent(BaseAgent):
    """
    Time-Weighted Average Price execution.

    Policy: sell exactly q0/N shares in each of the N periods.
    This is the canonical "do-nothing intelligent" baseline.
    Every execution paper in the literature reports improvement vs TWAP.

    Implementation note on action mapping:
        The environment's discrete action set is {0%, 25%, 50%, 75%, 100%}
        of REMAINING inventory. TWAP requires selling a fixed fraction of
        the ORIGINAL inventory each period, not a fixed fraction of remaining.

        Period k fraction of remaining q_k:
            f_k = (q0/N) / q_k

        But q_k decreases as we sell, so we can't use a fixed action index.
        The correct implementation computes the exact TWAP quantity each
        period and selects the action whose fraction of REMAINING inventory
        is closest to the desired quantity.

        This is the correct way to implement TWAP in a discrete-action env.

    Reference:
        Bertsimas & Lo (1998), "Optimal Control of Execution Costs"
        Almgren & Chriss (2001), Section 2
    """

    def __init__(self, env_config: EnvConfig):
        self.cfg             = env_config
        self.twap_per_period = env_config.q0 / env_config.N  # fixed shares/period
        self._period         = 0

    def reset(self) -> None:
        """Must be called at the start of each episode."""
        self._period = 0

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Find the action that executes closest to the cumulative TWAP target.

        TWAP requires selling q0/N shares in each of N periods. With a
        discrete action set {0%, 25%, 50%, 75%, 100%} of REMAINING inventory,
        a naive "sell 10% of remaining" approach fails because 10% is not
        a member of the action set and the argmin maps to 0% (action=0).

        Correct approach — target absolute shares, not fractions:
            cumulative_target_k = (k + 1) * (q0 / N)
            deficit_k = cumulative_target_k - (q0 - q_remaining)
            Select action whose absolute sell = ACTION_FRACS[a] * q_remaining
            is closest to deficit_k.

        This ensures the agent stays on the TWAP schedule in absolute terms.
        """
        q_remaining = state[1] * self.cfg.q0   # denormalize q* → shares

        if q_remaining < 1e-6:
            return 0

        # Shares that should have been sold INCLUDING this period
        cumulative_target = (self._period + 1) * self.twap_per_period
        cumulative_sold   = self.cfg.q0 - q_remaining
        # Deficit: how many shares to sell to stay on schedule
        deficit  = float(np.clip(cumulative_target - cumulative_sold, 0.0, q_remaining))

        # Absolute shares each action would sell
        candidate_x = ACTION_FRACS * q_remaining   # (5,)

        # Select action minimising |sell - deficit|
        action = int(np.argmin(np.abs(candidate_x - deficit)))

        self._period += 1
        return action

    @property
    def name(self) -> str:
        return 'TWAP'


# ============================================================================
# 2. Almgren-Chriss Agent
# ============================================================================

class AlmgrenChrissAgent(BaseAgent):
    """
    Analytical Almgren-Chriss mean-variance optimal execution schedule.

    This is the closed-form solution to the execution problem under:
        - Linear permanent impact: δp = γ · x_t
        - Linear temporary impact: ε_t = η · (x_t / dt)
        - Gaussian price innovations
        - Mean-variance objective: min E[IS] + λ · Var[IS]

    Closed-form solution (Almgren & Chriss 2001, Theorem 1):
        Remaining inventory at time t_k:
            q(t_k) = q0 · sinh(κ(T - t_k)) / sinh(κT)

        Trade size at period k:
            n_k = q(t_k) - q(t_{k+1})

        where:
            κ² = λσ² / η̃
            η̃  = η - ½γdt   (effective temporary impact)

    Risk aversion λ controls the mean-variance tradeoff:
        λ → 0   : minimise E[IS] only → uniform execution (≈ TWAP)
        λ → ∞   : minimise Var[IS] → front-loaded (sell fast early)

    Derivation sketch:
        IS = Σ_k [γ·x_k·(q0 - Σ_{j<k} x_j) + η·x_k²/dt]  (trading costs)
           + Σ_k σ·√dt·ξ_k·(q0 - Σ_{j≤k} x_j)              (market risk)
        Variance contribution comes from the second term.
        Minimising E[IS] + λ·Var[IS] yields the sinh schedule above.

    Key parameter (calibration):
        We calibrate λ by matching the AC schedule to a target fraction
        of inventory liquidated in the first period. The default λ=1e-6
        gives a nearly-uniform schedule under our parameters (κ≈0.0007),
        which is the appropriate comparison for a 60-minute horizon.

    Reference:
        Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio
        transactions." Journal of Risk, 3(2), 5–39.
    """

    def __init__(self, env_config: EnvConfig, risk_aversion: float = 1e-6):
        self.cfg    = env_config
        self.lam    = risk_aversion

        # Pre-compute the entire schedule at initialisation.
        # Schedule is deterministic — no adaptation to market conditions.
        self._schedule = self._compute_schedule()
        self._period   = 0

    def _compute_schedule(self) -> np.ndarray:
        """
        Compute the full AC schedule n_1, ..., n_N at t=0.

        Uses the inventory-path formulation:
            q(t_k) = q0 · sinh(κ(T - t_k)) / sinh(κT)
            n_k = q(t_k) - q(t_{k+1})

        This is numerically more stable than the direct formula for
        small κ (near-uniform case) because sinh cancels cleanly.

        Returns:
            schedule: (N,) array of shares to sell each period.
                      Guaranteed to sum exactly to q0 by construction.
        """
        N  = self.cfg.N
        q0 = self.cfg.q0
        dt = self.cfg.dt
        T  = self.cfg.T

        # Effective temporary impact coefficient
        # η̃ = η - ½γdt   (removes double-counting of permanent impact)
        eta_tilde = self.cfg.eta - 0.5 * self.cfg.gamma * dt

        # Safety: if eta_tilde ≤ 0, fall back to TWAP
        if eta_tilde <= 0:
            return np.full(N, q0 / N, dtype=np.float64)

        # κ² = λσ² / η̃
        kappa2 = self.lam * (self.cfg.sigma ** 2) / eta_tilde

        # For κ² ≈ 0 (low risk aversion), sinh(κT) ≈ κT and the
        # schedule degenerates to uniform. Use TWAP in this limit.
        if kappa2 < 1e-14:
            return np.full(N, q0 / N, dtype=np.float64)

        kappa = np.sqrt(kappa2)

        # Inventory path at each decision time t_k = k * dt
        # q(t_k) = q0 · sinh(κ(T - t_k)) / sinh(κT)
        t_nodes = np.arange(N + 1, dtype=np.float64) * dt  # t_0, t_1, ..., t_N
        q_path  = q0 * np.sinh(kappa * (T - t_nodes)) / np.sinh(kappa * T)
        q_path[-1] = 0.0   # enforce q(T) = 0 exactly (numerical safety)

        # Trade sizes: n_k = q(t_k) - q(t_{k+1})
        schedule = np.diff(-q_path)   # n_k = q_k - q_{k+1}  > 0

        # Clip negative values (can occur due to floating point)
        schedule = np.clip(schedule, 0.0, None)

        # Renormalize to ensure sum = q0 exactly
        # (avoids any residual from floating-point accumulation)
        total = schedule.sum()
        if total > 0:
            schedule = schedule * (q0 / total)

        return schedule

    def reset(self) -> None:
        """Reset period counter for new episode."""
        self._period = 0

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Look up the pre-computed AC schedule for the current period.
        Convert shares-to-sell into the closest discrete action.

        The AC agent is completely open-loop: it ignores market state
        and simply executes the pre-computed schedule. This is the key
        limitation that RL agents overcome.
        """
        if self._period >= self.cfg.N:
            return 0

        q_remaining = state[1] * self.cfg.q0   # denormalize q*

        if q_remaining < 1e-6:
            self._period += 1
            return 0

        # Target shares for this period
        target_x = self._schedule[self._period]

        # Convert to fraction of remaining inventory
        target_frac = float(np.clip(target_x / (q_remaining + 1e-8), 0.0, 1.0))

        # Find closest discrete action
        diffs  = np.abs(ACTION_FRACS - target_frac)
        action = int(np.argmin(diffs))

        self._period += 1
        return action

    @property
    def schedule(self) -> np.ndarray:
        """The full pre-computed execution schedule (for diagnostics)."""
        return self._schedule.copy()

    @property
    def kappa(self) -> float:
        """The κ parameter controlling schedule shape."""
        eta_tilde = self.cfg.eta - 0.5 * self.cfg.gamma * self.cfg.dt
        if eta_tilde <= 0: return 0.0
        kappa2 = self.lam * (self.cfg.sigma ** 2) / eta_tilde
        return float(np.sqrt(max(kappa2, 0)))

    @property
    def name(self) -> str:
        return f'AC(λ={self.lam:.0e})'


# ============================================================================
# Shared components for learned agents (DQN, DDQN, QR-DQN)
# ============================================================================

@dataclass
class DeepRLConfig:
    """
    Shared hyperparameters for all deep RL baseline agents.
    Kept identical to AgentConfig in iqn_agent.py wherever possible
    so comparisons are fair (same lr, batch size, buffer, etc.)
    """
    # Training
    lr               : float = 1e-4
    gamma            : float = 0.99
    batch_size       : int   = 64
    target_update_freq: int  = 500
    grad_clip_norm   : float = 10.0

    # Exploration
    epsilon_start    : float = 1.0
    epsilon_end      : float = 0.01
    epsilon_decay_steps: int = 50_000

    # Replay buffer
    replay_capacity  : int   = 50_000
    replay_min_size  : int   = 256

    # Network architecture (matches IQN's for fair comparison)
    hidden_dim       : int   = 128
    n_hidden_layers  : int   = 2

    # QR-DQN specific
    n_quantiles      : int   = 51   # N fixed quantiles (paper default)
    huber_kappa      : float = 1.0


class _QMLP(nn.Module):
    """
    Shared MLP Q-network for DQN and DDQN.
    Architecture: state_dim → hidden → hidden → n_actions
    Matches StateEncoder + OutputMLP structure from IQN for fair comparison.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int,
                 n_layers: int):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.ReLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(hidden_dim, n_actions)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, state_dim) → q_values: (B, A)"""
        return self.net(state)


class _QRNetwork(nn.Module):
    """
    QR-DQN network: outputs N fixed quantile values per action.

    Architecture: state_dim → hidden → hidden → (n_actions × n_quantiles)
    Reshaped output: (B, n_quantiles, n_actions)

    Key difference from IQN:
        - Fixed pre-defined quantile levels τ_i = (2i-1)/(2N)
        - Network directly outputs quantile VALUES (no τ input)
        - IQN conditions on τ and can output any quantile level
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int,
                 n_layers: int, n_quantiles: int):
        super().__init__()
        self.n_actions   = n_actions
        self.n_quantiles = n_quantiles

        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.ReLU()]
            in_dim = hidden_dim
        # Output: n_actions × n_quantiles values
        layers += [nn.Linear(hidden_dim, n_actions * n_quantiles)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

        # Fixed quantile levels τ_i = (2i-1)/(2N), i=1..N
        # Midpoint rule from Dabney et al. (2017) Eq. (9)
        taus = (2 * torch.arange(1, n_quantiles + 1, dtype=torch.float32) - 1) \
               / (2 * n_quantiles)
        self.register_buffer('taus', taus)   # shape: (N,)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, state_dim)
        returns: quantiles (B, N, A)
        """
        B = state.shape[0]
        out = self.net(state)                                # (B, A*N)
        out = out.view(B, self.n_actions, self.n_quantiles)  # (B, A, N)
        out = out.transpose(1, 2)                            # (B, N, A)
        return out

    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Mean over quantiles → expected Q-value for action selection."""
        return self.forward(state).mean(dim=1)               # (B, A)


class _DeepRLBase(BaseAgent):
    """
    Shared training infrastructure for DQN, DDQN, QR-DQN.
    Subclasses only override _compute_loss().
    """

    def __init__(self, cfg: DeepRLConfig, state_dim: int, n_actions: int,
                 device: Optional[torch.device] = None, seed: int = 42):
        self.cfg       = cfg
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device    = device or self._auto_device()
        self._step     = 0

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build networks (overridden in subclasses)
        self.online_net = self._build_network()
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.lr)

        replay_cfg = ReplayConfig(
            capacity   = cfg.replay_capacity,
            batch_size = cfg.batch_size,
            min_size   = cfg.replay_min_size,
        )
        self.replay = ReplayBuffer(replay_cfg, state_dim, self.device)

    @abstractmethod
    def _build_network(self) -> nn.Module: ...

    @abstractmethod
    def _compute_loss(self, batch: dict) -> torch.Tensor: ...

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        epsilon = self._get_epsilon() if not eval_mode else 0.0
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        state_t  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self._get_q_values(state_t)     # (1, A)
        return int(q_values.argmax(dim=-1).item())

    @torch.no_grad()
    def _get_q_values(self, state_t: torch.Tensor) -> torch.Tensor:
        """Q-values for action selection. Overridden in QR-DQN."""
        return self.online_net(state_t)

    def store(self, state, action, reward, next_state, done) -> None:
        self.replay.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        if not self.replay.ready:
            return None
        batch = self.replay.sample()
        loss  = self._compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(),
                                 self.cfg.grad_clip_norm)
        self.optimizer.step()
        self._step += 1
        if self._step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss.item()

    def _get_epsilon(self) -> float:
        progress = min(self._step / self.cfg.epsilon_decay_steps, 1.0)
        return (self.cfg.epsilon_start
                + progress * (self.cfg.epsilon_end - self.cfg.epsilon_start))

    def _auto_device(self) -> torch.device:
        if torch.backends.mps.is_available(): return torch.device('mps')
        if torch.cuda.is_available():         return torch.device('cuda')
        return torch.device('cpu')

    def save(self, path: str) -> None:
        torch.save({'online_net': self.online_net.state_dict(),
                    'target_net': self.target_net.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'step'      : self._step}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self._step = ckpt['step']

    @property
    def epsilon(self) -> float:
        return self._get_epsilon()


# ============================================================================
# 3. DQN Agent (Mnih et al. 2015)
# ============================================================================

class DQNAgent(_DeepRLBase):
    """
    Deep Q-Network.

    Loss (Mnih et al. 2015 Eq. 2):
        L(θ) = E[(y - Q(s,a;θ))²]
        y    = r + γ · max_{a'} Q(s', a'; θ⁻)   ← target network

    Known limitation:
        max_{a'} Q(s', a'; θ⁻) is a biased estimator.
        Target Q-values are computed with the same network used for
        evaluation — creating a positive feedback loop that inflates
        Q-values (overestimation bias). DDQN fixes this.

    In paper Table 3, DQN is shown to be outperformed by DDQN due to
    overestimation bias in the volatile execution setting.

    Reference:
        Mnih, V. et al. (2015). "Human-level control through deep
        reinforcement learning." Nature, 518, 529–533.
    """

    def _build_network(self) -> nn.Module:
        return _QMLP(self.state_dim, self.n_actions,
                     self.cfg.hidden_dim, self.cfg.n_hidden_layers
                     ).to(self.device)

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Standard DQN loss: MSE between Q(s,a) and Bellman target.

        Target: y = r + γ(1-d) · max_{a'} Q_{target}(s', a')
                                  ↑ target net, max over actions
        """
        states      = batch['states']       # (B, state_dim)
        actions     = batch['actions']      # (B,)
        rewards     = batch['rewards']      # (B,)
        next_states = batch['next_states']  # (B, state_dim)
        dones       = batch['dones']        # (B,)
        B = states.shape[0]

        # Current Q-values: Q(s, a; θ)
        q_curr = self.online_net(states)                        # (B, A)
        q_sa   = q_curr.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Target Q-values: max_{a'} Q(s', a'; θ⁻)
        with torch.no_grad():
            q_next  = self.target_net(next_states)              # (B, A)
            v_next  = q_next.max(dim=1).values                  # (B,)   max over a'
            targets = rewards + self.cfg.gamma * (1.0 - dones) * v_next  # (B,)

        # MSE loss
        loss = nn.functional.mse_loss(q_sa, targets)
        return loss

    @property
    def name(self) -> str:
        return 'DQN'


# ============================================================================
# 4. DDQN Agent (Van Hasselt et al. 2016)
# ============================================================================

class DDQNAgent(_DeepRLBase):
    """
    Double Deep Q-Network.

    Fixes DQN's overestimation bias by decoupling action selection
    from action evaluation:
        - Online network selects the greedy action a*
        - Target network evaluates the value of a*

    Loss (Van Hasselt et al. 2016 Eq. 5):
        y    = r + γ · Q(s', a*; θ⁻)
        a*   = argmax_{a'} Q(s', a'; θ)     ← online network selects
        L(θ) = E[(y - Q(s,a;θ))²]

    This is the direct predecessor of IQN in the execution literature.
    Ning et al. (2021) use DDQN as their main method.
    Outperforms DQN because it avoids inflating Q-values in volatile
    financial environments where Q-function is hard to estimate accurately.

    The IQN paper also uses Double DQN-style target computation (same
    decoupling), so the improvement from DDQN → IQN is purely due to
    distributional learning, not the Double trick.

    References:
        Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep
        reinforcement learning with double Q-learning." AAAI-16.
        Ning, B., Lin, F., & Jaimungal, S. (2021). "Double Deep
        Q-Learning for Optimal Execution." arXiv:1812.10490.
    """

    def _build_network(self) -> nn.Module:
        return _QMLP(self.state_dim, self.n_actions,
                     self.cfg.hidden_dim, self.cfg.n_hidden_layers
                     ).to(self.device)

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Double DQN loss: decouple action selection from evaluation.

        Target: a* = argmax_{a'} Q(s', a'; θ)    ← online net
                y  = r + γ(1-d) · Q(s', a*; θ⁻)  ← target net evaluates
        """
        states      = batch['states']       # (B, state_dim)
        actions     = batch['actions']      # (B,)
        rewards     = batch['rewards']      # (B,)
        next_states = batch['next_states']  # (B, state_dim)
        dones       = batch['dones']        # (B,)

        # Current Q-values
        q_curr = self.online_net(states)                            # (B, A)
        q_sa   = q_curr.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            # Double DQN: online net selects action, target net evaluates
            q_next_online = self.online_net(next_states)            # (B, A)
            a_star        = q_next_online.argmax(dim=1)             # (B,)

            q_next_target = self.target_net(next_states)            # (B, A)
            v_next = q_next_target.gather(1, a_star.unsqueeze(1)).squeeze(1)  # (B,)

            targets = rewards + self.cfg.gamma * (1.0 - dones) * v_next  # (B,)

        loss = nn.functional.mse_loss(q_sa, targets)
        return loss

    @property
    def name(self) -> str:
        return 'DDQN'


# ============================================================================
# 5. QR-DQN Agent (Dabney et al. 2017)
# ============================================================================

class QRDQNAgent(_DeepRLBase):
    """
    Quantile Regression DQN.

    The critical ablation step between DDQN and IQN.
    Both learn distributional Q-functions using quantile regression.
    The ONLY difference is how quantile levels are handled:

        QR-DQN : Fixed discrete quantile levels τ_i = (2i-1)/(2N)
                 Network directly outputs quantile VALUES
                 Action selection: argmax_a E[Z(s,a)] = argmax_a mean(Z_i(s,a))

        IQN    : Continuous τ sampled from U[0,1] (or U[0,α] for CVaR)
                 Network takes τ as INPUT and outputs conditional quantile value
                 Action selection: argmax_a E_{τ~U[0,α]}[z(τ,s,a)]

    QR-DQN establishes that distributional learning helps even with fixed
    quantiles. If IQN-neutral matches QR-DQN, the gain from IQN comes
    purely from the continuous τ enabling risk-sensitive policies.

    Loss (Dabney et al. 2017 Eq. 10):
        For fixed quantile τ_i and its estimate Z_i(s,a):
            L(θ) = Σ_i Σ_j ρ_{τ_i}(Z_j(s',a*) - Z_i(s,a)) / N'

        where ρ_τ(u) = |τ - 𝟙[u<0]| · L_κ(u) / κ

    Note on τ convention (Dabney 2017 vs Dabney 2018):
        QR-DQN (2017): τ_i = (2i-1)/(2N)    midpoint rule
        IQN    (2018): τ  ~ U[0,1]            continuous

    Reference:
        Dabney, W., Rowland, M., Bellemare, M. G., & Munos, R. (2017).
        "Distributional reinforcement learning with quantile regression."
        AAAI 2018, 2892–2901.
    """

    def _build_network(self) -> nn.Module:
        return _QRNetwork(
            state_dim   = self.state_dim,
            n_actions   = self.n_actions,
            hidden_dim  = self.cfg.hidden_dim,
            n_layers    = self.cfg.n_hidden_layers,
            n_quantiles = self.cfg.n_quantiles,
        ).to(self.device)

    @torch.no_grad()
    def _get_q_values(self, state_t: torch.Tensor) -> torch.Tensor:
        """Override: mean over quantile dimension for action selection."""
        return self.online_net.q_values(state_t)   # (B, A)

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """
        QR-DQN quantile Huber loss.

        Step-by-step with shapes (B=batch, N=n_quantiles, A=n_actions):

        1. z_curr[i]   = Z_{θ}(τ_i, s, a)    via online net → (B, N)
        2. z_tgt[j]    = Z_{θ̄}(τ_j, s', a*)   via target net → (B, N)
        3. u_{ij}      = z_tgt[j] - z_curr[i]  pairwise TD errors → (B, N, N)
        4. ρ_{τ_i}(u)  = |τ_i - 𝟙[u<0]| · L_κ(u) / κ → (B, N, N)
        5. L           = mean over all (i, j, batch)

        Note: u_{ij} = target_j - current_i
        The outer axis (dim=1) indexes τ_i (current quantile).
        The inner axis (dim=2) indexes τ_j (target quantile).
        """
        states      = batch['states']       # (B, state_dim)
        actions     = batch['actions']      # (B,)
        rewards     = batch['rewards']      # (B,)
        next_states = batch['next_states']  # (B, state_dim)
        dones       = batch['dones']        # (B,)
        B = states.shape[0]
        N = self.cfg.n_quantiles

        # ── Current quantiles Z_θ(τ_i, s, a) ────────────────────────
        all_z_curr = self.online_net(states)    # (B, N, A)
        # Select z for chosen action
        act_idx = actions.view(B, 1, 1).expand(B, N, 1)
        z_curr  = all_z_curr.gather(2, act_idx).squeeze(2)  # (B, N)

        with torch.no_grad():
            # ── Target: a* via online net (Double DQN style) ─────────
            # Use mean quantile value for greedy action selection
            q_next_online = self.online_net.q_values(next_states)  # (B, A)
            a_star        = q_next_online.argmax(dim=1)            # (B,)

            # ── Target quantiles Z_{θ̄}(τ_j, s', a*) ─────────────────
            all_z_tgt  = self.target_net(next_states)   # (B, N, A)
            tgt_idx    = a_star.view(B, 1, 1).expand(B, N, 1)
            z_next     = all_z_tgt.gather(2, tgt_idx).squeeze(2)  # (B, N)

            # ── TD targets: r + γ(1-d) · z_next_j ───────────────────
            # rewards: (B,) → (B, 1), dones: (B,) → (B, 1)
            z_tgt = (rewards.unsqueeze(1)
                     + self.cfg.gamma * (1.0 - dones.unsqueeze(1)) * z_next
                     )  # (B, N)

        # ── Pairwise TD errors u_{ij} = z_tgt_j - z_curr_i ──────────
        # z_curr: (B, N, 1), z_tgt: (B, 1, N) → broadcast → (B, N, N)
        u = z_tgt.unsqueeze(1) - z_curr.unsqueeze(2)  # (B, N, N)
        #   z_tgt.unsqueeze(1):  (B, 1, N)
        #   z_curr.unsqueeze(2): (B, N, 1)
        #   u[b, i, j] = z_tgt[b,j] - z_curr[b,i]

        # ── Quantile Huber loss ρ_{τ_i}(u_{ij}) ─────────────────────
        loss = self._quantile_huber_loss(u)
        return loss

    def _quantile_huber_loss(self, u: torch.Tensor) -> torch.Tensor:
        """
        Quantile Huber loss with FIXED τ levels from network buffer.

        u     : (B, N, N)   TD errors  u[b,i,j] = target_j - current_i
        taus  : (N,)        fixed quantile levels for current estimates

        ρ_{τ_i}(u) = |τ_i - 𝟙[u < 0]| · L_κ(u) / κ

        τ_i expands to (1, N, 1) to broadcast over all (b, j) pairs.
        """
        kappa = self.cfg.huber_kappa

        # Huber loss: (B, N, N)
        huber = torch.where(
            u.abs() <= kappa,
            0.5 * u.pow(2),
            kappa * (u.abs() - 0.5 * kappa),
        ) / kappa

        # τ_i for current quantiles: (N,) → (1, N, 1)
        taus = self.online_net.taus.view(1, self.cfg.n_quantiles, 1)  # (1, N, 1)

        indicator = (u < 0).float()                     # (B, N, N)
        weight    = (taus - indicator).abs()             # (B, N, N)

        rho  = weight * huber                           # (B, N, N)
        # Sum over target quantile dim j, mean over i and batch
        loss = rho.sum(dim=2).mean(dim=1).mean(dim=0)   # scalar
        return loss

    @property
    def name(self) -> str:
        return f'QR-DQN(N={self.cfg.n_quantiles})'