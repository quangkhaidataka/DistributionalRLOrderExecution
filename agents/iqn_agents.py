"""
agents/iqn_agent.py
--------------------
IQN Agent implementing both IQN-neutral and IQN-CVaR policies.

This is the central contribution of the paper.

Architecture:  IQN(s,τ) = f( ψ(s) ⊙ φ(τ) )
Loss:          Quantile Huber loss (Dabney et al. 2018)
Policy:
    IQN-neutral  →  τ ~ U[0,1]   at inference
    IQN-CVaR_α   →  τ ~ U[0,α]   at inference (SAME weights)

Key design decision — ONE class, TWO policies:
    IQNAgent stores one online network and one target network.
    The risk profile (neutral vs CVaR) is controlled by a
    single parameter self.cvar_alpha passed at construction:
        IQNAgent(cvar_alpha=1.0)  → IQN-neutral
        IQNAgent(cvar_alpha=0.95) → IQN-CVaR_0.95
        IQNAgent(cvar_alpha=0.90) → IQN-CVaR_0.90

    During TRAINING: always τ ~ U[0,1] for both variants
        → same loss, same distributional learning objective
        → this is crucial: we learn the FULL distribution

    During INFERENCE: τ ~ U[0, cvar_alpha]
        → for neutral: U[0,1]  → optimizes E[Z]
        → for CVaR:    U[0,α]  → optimizes CVaR_α[Z]

    This means IQN-neutral and IQN-CVaR are trained identically
    and only differ at inference time. The paper's key claim is
    that this zero-cost policy transformation produces superior
    tail-risk management without sacrificing mean performance.

Loss derivation (from paper Section 3):
    Given transition (s, a, r, s') and samples τ, τ':
        TD target:  ŷ = r + γ · z_{θ̄}(τ', s', a*)
                    a* = argmax_a' E_{τ''}[z_θ(τ'', s', a'')]
        TD error:   u = ŷ - z_θ(τ, s, a)

        Huber loss:  L_κ(u) = { ½u²         if |u| ≤ κ
                               { κ(|u| - κ/2) if |u| > κ

        Quantile Huber loss:
            ρ_τ(u) = |τ - 𝟙[u<0]| · L_κ(u) / κ

        Total loss:
            L(θ) = E_{τ,τ'} [ ρ_τ( ŷ - z_θ(τ,s,a) ) ]

Tensor shape guide (used consistently in comments below):
    B  = batch_size (e.g. 64)
    N  = n_tau_samples during training (e.g. 8)
    N' = n_tau_targets during training (e.g. 8, can differ)
    A  = n_actions (5)
    d  = hidden_dim (128)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.iqn_network import IQNNetwork, NetworkConfig
from training.replay_buffer import ReplayBuffer, ReplayConfig


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """
    All hyperparameters for IQN agent training.
    Defaults tuned for our execution MDP (small state space,
    short episodes, financial reward scale).
    """
    # --- Risk profile ---
    # cvar_alpha = 1.0  → IQN-neutral  (τ ~ U[0,1] at inference)
    # cvar_alpha = 0.95 → IQN-CVaR_0.95 (τ ~ U[0,0.95] at inference)
    cvar_alpha       : float = 1.0

    # --- Training ---
    lr               : float = 1e-4    # Adam learning rate
    gamma            : float = 0.99    # discount factor
    batch_size       : int   = 64      # minibatch size
    target_update_freq: int  = 100     # hard target update every N steps
    grad_clip_norm   : float = 10.0    # gradient clipping (stability)

    # --- Quantile sampling ---
    n_tau_samples    : int   = 8       # N  taus sampled per update (train)
    n_tau_targets    : int   = 8       # N' taus for TD targets
    n_tau_policy     : int   = 32      # taus for greedy action selection
    huber_kappa      : float = 1.0     # κ in Huber loss (paper default)

    # --- Exploration ---
    epsilon_start    : float = 1.0
    epsilon_end      : float = 0.01
    epsilon_decay_steps: int = 10_000  # linear decay over this many steps

    # --- Replay buffer ---
    replay_capacity  : int   = 50_000
    replay_min_size  : int   = 1_000

    # --- Network ---
    hidden_dim       : int   = 128
    cos_embedding_dim: int   = 64
    n_hidden_layers  : int   = 2

    @property
    def is_cvar(self) -> bool:
        return self.cvar_alpha < 1.0

    @property
    def name(self) -> str:
        if self.is_cvar:
            return f'IQN-CVaR_{self.cvar_alpha:.2f}'
        return 'IQN-neutral'


# ---------------------------------------------------------------------------
# IQN Agent
# ---------------------------------------------------------------------------

class IQNAgent:
    """
    IQN Agent for optimal execution.

    Implements both IQN-neutral and IQN-CVaR_α with a single class.
    The only behavioral difference between the two is the τ sampling
    range used during action selection (inference).

    Usage:
        # IQN-neutral
        agent = IQNAgent(AgentConfig(cvar_alpha=1.0), state_dim=6, n_actions=5)

        # IQN-CVaR_0.95
        agent = IQNAgent(AgentConfig(cvar_alpha=0.95), state_dim=6, n_actions=5)

        # Training loop
        state = env.reset()
        for step in range(total_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.update()
            state = env.reset() if done else next_state
    """

    def __init__(
        self,
        cfg       : AgentConfig,
        state_dim : int,
        n_actions : int,
        device    : Optional[torch.device] = None,
        seed      : int = 42,
    ):
        self.cfg       = cfg
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device    = device or self._auto_device()
        self._step     = 0       # global step counter

        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── Networks ──────────────────────────────────────────────────
        net_cfg = NetworkConfig(
            state_dim         = state_dim,
            n_actions         = n_actions,
            hidden_dim        = cfg.hidden_dim,
            cos_embedding_dim = cfg.cos_embedding_dim,
            n_hidden_layers   = cfg.n_hidden_layers,
            n_tau_samples     = cfg.n_tau_samples,
            n_tau_policy      = cfg.n_tau_policy,
        )

        # Online network: updated every step
        self.online_net = IQNNetwork(net_cfg).to(self.device)

        # Target network: hard-updated every target_update_freq steps
        # Initialized as deep copy — same weights, no gradient flow
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()   # target network never in training mode
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        # ── Optimizer ────────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=cfg.lr,
        )

        # ── Replay buffer ────────────────────────────────────────────
        replay_cfg = ReplayConfig(
            capacity   = cfg.replay_capacity,
            batch_size = cfg.batch_size,
            min_size   = cfg.replay_min_size,
        )
        self.replay = ReplayBuffer(replay_cfg, state_dim, self.device)

        print(f'[IQNAgent] {cfg.name} | device={self.device} | '
              f'params={self._count_params():,}')

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        ε-greedy action selection.

        During training: ε-greedy with decaying ε
        During evaluation: greedy (ε=0)

        Greedy policy:
            IQN-neutral:  a* = argmax_a E_{τ~U[0,1]}  [z_θ(τ,s,a)]
            IQN-CVaR_α:   a* = argmax_a E_{τ~U[0,α]}  [z_θ(τ,s,a)]
                          = argmax_a CVaR_α[Z(s,a)]

        Args:
            state    : (state_dim,) numpy array
            eval_mode: if True, always greedy (no exploration)

        Returns:
            action: int in {0, 1, 2, 3, 4}
        """
        epsilon = self._get_epsilon() if not eval_mode else 0.0

        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)

        # Greedy: use CVaR_alpha range for action selection
        state_t = torch.FloatTensor(state).to(self.device)
        q_values = self.online_net.get_action_values(
            state    = state_t,
            n_tau    = self.cfg.n_tau_policy,
            tau_low  = 0.0,
            tau_high = self.cfg.cvar_alpha,   # KEY: 1.0 for neutral, α for CVaR
        )                                      # q_values: (1, A)

        return int(q_values.argmax(dim=-1).item())

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(
        self,
        state      : np.ndarray,
        action     : int,
        reward     : float,
        next_state : np.ndarray,
        done       : bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update(self) -> Optional[float]:
        """
        Perform one gradient update step if buffer is ready.

        Returns:
            loss value (float) if update was performed, else None
        """
        if not self.replay.ready:
            return None

        batch = self.replay.sample()
        loss  = self._compute_loss(batch)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents exploding gradients with
        # financial reward noise
        nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            self.cfg.grad_clip_norm,
        )
        self.optimizer.step()

        # Hard target update
        self._step += 1
        if self._step % self.cfg.target_update_freq == 0:
            self._update_target()

        return loss.item()

    # ------------------------------------------------------------------
    # Loss computation  (core of the IQN algorithm)
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Quantile Huber loss from Dabney et al. (2018).

        Step-by-step with tensor shapes annotated:

        1. Sample τ  for current-state quantiles    → (B*N,  1)
        2. Sample τ' for next-state TD targets       → (B*N', 1)
        3. Compute z_θ(τ,  s,  a)  via online net   → (B, N,  A) → select a → (B, N)
        4. Compute z_{θ̄}(τ', s', a*) via target net → (B, N', A) → select a*→ (B, N')
        5. Build TD targets:  ŷ = r + γ(1-d)·z_{θ̄}  → (B, 1, N')
        6. Compute TD errors: u = ŷ - z_θ(τ,s,a)    → (B, N, N')
        7. Apply quantile Huber loss                  → scalar
        """
        states      = batch['states']       # (B, state_dim)
        actions     = batch['actions']      # (B,)
        rewards     = batch['rewards']      # (B,)
        next_states = batch['next_states']  # (B, state_dim)
        dones       = batch['dones']        # (B,)
        B = states.shape[0]

        # ── Step 3: Current quantiles z_θ(τ, s, a) ───────────────────
        # Always train with τ ~ U[0,1] regardless of cvar_alpha
        # (we learn the FULL distribution during training)
        current_quantiles, tau = self.online_net(
            state    = states,
            n_tau    = self.cfg.n_tau_samples,
            tau_low  = 0.0,
            tau_high = 1.0,
        )
        # current_quantiles: (B, N, A)
        # tau:               (B*N, 1)

        # Select quantiles for chosen actions
        # actions: (B,) → expand to (B, N, 1) for gather
        action_idx = actions.view(B, 1, 1).expand(B, self.cfg.n_tau_samples, 1)
        current_quantiles = current_quantiles.gather(dim=2, index=action_idx)
        current_quantiles = current_quantiles.squeeze(2)   # (B, N)

        # ── Step 4: TD targets via target network ─────────────────────
        with torch.no_grad():
            # 4a. Greedy action a* = argmax_a' E[z_θ(τ'', s', a')]
            #     Use online network for action selection (Double DQN style)
            #     This reduces overestimation bias (Van Hasselt et al. 2016)
            q_next_online = self.online_net.get_action_values(
                state    = next_states,
                n_tau    = self.cfg.n_tau_policy,
                tau_low  = 0.0,
                tau_high = 1.0,
            )                              # (B, A) — mean over τ''
            a_star = q_next_online.argmax(dim=1)  # (B,)

            # 4b. Evaluate a* with target network
            target_quantiles, _ = self.target_net(
                state    = next_states,
                n_tau    = self.cfg.n_tau_targets,
                tau_low  = 0.0,
                tau_high = 1.0,
            )
            # target_quantiles: (B, N', A)

            # Select quantiles for greedy action a*
            a_star_idx = a_star.view(B, 1, 1).expand(B, self.cfg.n_tau_targets, 1)
            target_quantiles = target_quantiles.gather(dim=2, index=a_star_idx)
            target_quantiles = target_quantiles.squeeze(2)   # (B, N')

            # ── Step 5: Build TD targets ŷ = r + γ(1-d)·z_{θ̄} ────────
            # rewards: (B,)   → (B, 1)
            # dones:   (B,)   → (B, 1)
            # target_quantiles: (B, N') → (B, 1, N')
            rewards_exp = rewards.unsqueeze(1)               # (B, 1)
            dones_exp   = dones.unsqueeze(1)                 # (B, 1)
            td_targets  = (
                rewards_exp
                + self.cfg.gamma * (1.0 - dones_exp) * target_quantiles
            )                                                # (B, N')
            td_targets  = td_targets.unsqueeze(1)            # (B, 1, N')

        # ── Step 6: TD errors u = ŷ - z_θ(τ,s,a) ────────────────────
        # current_quantiles: (B, N,  1)
        # td_targets:        (B, 1,  N')
        # u:                 (B, N,  N')   broadcasting
        current_quantiles = current_quantiles.unsqueeze(2)   # (B, N, 1)
        u = td_targets - current_quantiles                   # (B, N, N')

        # ── Step 7: Quantile Huber loss ───────────────────────────────
        # Reshape tau from (B*N, 1) to (B, N, 1) for broadcasting
        tau_rs = tau.view(B, self.cfg.n_tau_samples, 1)      # (B, N, 1)

        loss = self._quantile_huber_loss(u, tau_rs)          # scalar

        return loss

    def _quantile_huber_loss(
        self,
        u  : torch.Tensor,   # (B, N, N')  TD errors
        tau: torch.Tensor,   # (B, N, 1)   quantile levels
    ) -> torch.Tensor:
        """
        Quantile Huber loss from Dabney et al. (2018) Eq. (10):

            ρ_τ(u) = |τ - 𝟙[u < 0]| · L_κ(u) / κ

        where L_κ is the Huber loss with threshold κ:
            L_κ(u) = { ½u²           if |u| ≤ κ
                     { κ(|u| - κ/2)  if |u| > κ

        Dividing by κ converts Huber loss to asymptotic
        unit-slope, making the scale invariant to κ choice.

        Args:
            u  : (B, N, N')  TD errors  (target - current)
            tau: (B, N, 1)   quantile levels τ

        Returns:
            loss: scalar mean loss
        """
        kappa = self.cfg.huber_kappa

        # Huber loss element-wise: (B, N, N')
        huber = torch.where(
            u.abs() <= kappa,
            0.5 * u.pow(2),
            kappa * (u.abs() - 0.5 * kappa),
        )
        # Normalize by kappa → asymptotic linear slope = 1
        huber = huber / kappa   # (B, N, N')

        # Asymmetric quantile weighting: |τ - 𝟙[u < 0]|
        # 𝟙[u < 0]: indicator that TD error is negative
        # tau broadcasts from (B, N, 1) to (B, N, N')
        indicator      = (u < 0).float()                     # (B, N, N')
        quantile_weight = (tau - indicator).abs()            # (B, N, N')

        # Element-wise product: quantile-weighted Huber loss
        rho = quantile_weight * huber                        # (B, N, N')

        # Mean over N' target samples (sum is also valid per paper)
        # Then mean over batch B and current samples N
        loss = rho.mean(dim=2).mean(dim=1).mean(dim=0)       # scalar

        return loss

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def _update_target(self) -> None:
        """Hard copy online → target network weights."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    def _get_epsilon(self) -> float:
        """
        Linear epsilon decay from epsilon_start to epsilon_end
        over epsilon_decay_steps steps.
        """
        progress = min(self._step / self.cfg.epsilon_decay_steps, 1.0)
        return (self.cfg.epsilon_start
                + progress * (self.cfg.epsilon_end - self.cfg.epsilon_start))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _auto_device(self) -> torch.device:
        """Select best available device (MPS for Apple Silicon, then CUDA, then CPU)."""
        if torch.backends.mps.is_available():
            return torch.device('mps')    # MacBook Air M1/M2 GPU
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.online_net.parameters())

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'online_net'  : self.online_net.state_dict(),
            'target_net'  : self.target_net.state_dict(),
            'optimizer'   : self.optimizer.state_dict(),
            'step'        : self._step,
            'cfg'         : self.cfg,
        }, path)
        print(f'[IQNAgent] Checkpoint saved → {path}')

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self._step = ckpt['step']
        print(f'[IQNAgent] Checkpoint loaded ← {path} (step={self._step})')

    @property
    def epsilon(self) -> float:
        return self._get_epsilon()

    @property
    def name(self) -> str:
        return self.cfg.name

    def __repr__(self) -> str:
        return (f"IQNAgent(name={self.name}, "
                f"step={self._step}, "
                f"epsilon={self.epsilon:.3f}, "
                f"buffer={self.replay.size})")