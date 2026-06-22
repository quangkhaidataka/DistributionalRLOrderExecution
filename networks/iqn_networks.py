"""
networks/iqn_network.py
-----------------------
IQN neural network architecture.

Exactly follows Dabney et al. (2018) Section 3:

    IQN(s, τ) = f( ψ(s) ⊙ φ(τ) )

    where:
        ψ(s)  = state encoder MLP         → R^d
        φ(τ)  = cosine quantile embedding → R^d
        ⊙     = element-wise product
        f(·)  = output MLP                → R^|A|

Architecture rationale:
    - State encoder and quantile embedding share the same
      embedding dimension d, so element-wise product is valid
    - Cosine embedding from the paper:
          φ_j(τ) = ReLU( Σ_{i=1}^{n} cos(π·i·τ) · w_{ij} + b_j )
      where n = cos_embedding_dim (paper uses 64)
    - Element-wise product forces τ-conditioning throughout
      the entire value prediction — not just at the output layer
    - This is the key architectural difference from QR-DQN,
      which learns fixed quantile values with no τ input

Tensor shape conventions (documented at each step):
    B  = batch size
    A  = number of actions  (5 in our problem)
    N  = number of tau samples per forward pass
    d  = embedding dimension (hidden_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Network configuration
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """
    IQN network hyperparameters.

    Defaults follow Dabney et al. (2018) adapted for our
    small state space (dim=6) instead of Atari image inputs.
    We use a smaller network since our state is 6-dim,
    not 84×84 pixels — a large Atari-style network would
    massively overfit on tabular financial features.
    """
    state_dim        : int   = 5      # our MDP state dimension
    n_actions        : int   = 5      # {0%, 25%, 50%, 75%, 100%}
    hidden_dim       : int   = 128    # embedding dimension d
    cos_embedding_dim: int   = 64     # n in cosine embedding (paper uses 64)
    n_hidden_layers  : int   = 2      # depth of state encoder + output MLP
    n_tau_samples    : int   = 8      # N taus sampled per forward pass (train)
    n_tau_policy     : int   = 32     # N taus for greedy action selection


# ---------------------------------------------------------------------------
# Cosine quantile embedding  φ(τ)
# ---------------------------------------------------------------------------

class CosineQuantileEmbedding(nn.Module):
    """
    Embeds scalar quantile levels τ ∈ [0,1] into R^d.

    From Dabney et al. (2018) Eq. (4):
        φ_j(τ) = ReLU( Σ_{i=1}^{n} cos(π·i·τ) · w_{ij} + b_j )

    Why cosine features?
        - Smooth in τ: nearby quantiles get similar embeddings
        - Periodic basis: captures the U-shape of quantile functions
        - Fixed basis (no learning needed for basis functions):
          only the linear projection w_{ij} is learned
        - Better than raw τ scalar: gives the network rich
          τ-information to condition on

    Input:  tau  shape (B*N, 1)  — flattened batch × tau samples
    Output: phi  shape (B*N, d)  — quantile embedding vectors
    """

    def __init__(self, cos_embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.cos_embedding_dim = cos_embedding_dim   # n
        self.hidden_dim        = hidden_dim           # d

        # Learnable linear projection: R^n → R^d
        # Maps cosine features to embedding space
        self.linear = nn.Linear(cos_embedding_dim, hidden_dim)

        # Precompute i = [1, 2, ..., n] — used in cos(π·i·τ)
        # Register as buffer so it moves to GPU automatically
        i_vals = torch.arange(0, cos_embedding_dim, dtype=torch.float32)
        self.register_buffer('i_vals', i_vals)  # shape: (n,)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau: shape (B*N, 1)  quantile levels in [0, 1]

        Returns:
            phi: shape (B*N, d)  quantile embeddings
        """
        # tau: (B*N, 1) → broadcast to (B*N, n)
        # cos_features[k, i] = cos(π · (i+1) · τ_k)
        cos_features = torch.cos(
            torch.pi * self.i_vals.unsqueeze(0) * tau  # (B*N, n)
        )                                               # shape: (B*N, n)

        # Linear projection + ReLU: (B*N, n) → (B*N, d)
        phi = F.relu(self.linear(cos_features))         # shape: (B*N, d)
        return phi


# ---------------------------------------------------------------------------
# State encoder  ψ(s)
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """
    Encodes state s ∈ R^6 into embedding space R^d.

    Simple MLP: input_dim → hidden_dim → hidden_dim
    Uses LayerNorm for training stability with financial data
    (returns/prices have different scales across features).

    Input:  s    shape (B, state_dim)
    Output: psi  shape (B, d)
    """

    def __init__(self, state_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ]
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: shape (B, state_dim)

        Returns:
            psi: shape (B, d)
        """
        return self.net(state)   # (B, d)


# ---------------------------------------------------------------------------
# Output MLP  f(·)
# ---------------------------------------------------------------------------

class OutputMLP(nn.Module):
    """
    Maps combined embedding ψ(s) ⊙ φ(τ) to Q-quantile values.

    Input:  combined  shape (B*N, d)
    Output: q_vals    shape (B*N, A)
    """

    def __init__(self, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        """
        Args:
            combined: shape (B*N, d)

        Returns:
            q_vals: shape (B*N, A)
        """
        return self.net(combined)   # (B*N, A)


# ---------------------------------------------------------------------------
# Full IQN Network
# ---------------------------------------------------------------------------

class IQNNetwork(nn.Module):
    """
    Full IQN network: IQN(s, τ) = f( ψ(s) ⊙ φ(τ) )

    Two usage modes (controlled by tau_low/tau_high at call time):

    1. Risk-neutral (IQN-neutral):
           tau ~ U[0, 1]  →  action = argmax_a E[Z(s,a)]

    2. Risk-sensitive (IQN-CVaR_α):
           tau ~ U[0, α]  →  action = argmax_a CVaR_α[Z(s,a)]
           (same trained weights, just different τ sampling)

    This single network parameterizes the ENTIRE family of
    risk-sensitive policies — no retraining needed to change α.

    Forward pass shapes (B=batch, N=n_tau, A=n_actions, d=hidden):
        state        → (B, state_dim)
        tau          → (B*N, 1)
        psi          → (B, d)       state embedding
        psi_expand   → (B*N, d)     repeated N times for each tau
        phi          → (B*N, d)     quantile embedding
        combined     → (B*N, d)     element-wise product ψ ⊙ φ
        quantiles    → (B*N, A)     quantile values per action
        quantiles_rs → (B, N, A)    reshaped for downstream use
    """

    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg

        self.state_encoder = StateEncoder(
            state_dim  = cfg.state_dim,
            hidden_dim = cfg.hidden_dim,
            n_layers   = cfg.n_hidden_layers,
        )
        self.quantile_embed = CosineQuantileEmbedding(
            cos_embedding_dim = cfg.cos_embedding_dim,
            hidden_dim        = cfg.hidden_dim,
        )
        self.output_mlp = OutputMLP(
            hidden_dim = cfg.hidden_dim,
            n_actions  = cfg.n_actions,
        )

        # Weight initialization: Xavier uniform (standard for RL)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        state   : torch.Tensor,
        n_tau   : int,
        tau_low : float = 0.0,
        tau_high: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute quantile values for sampled τ levels.

        Args:
            state    : (B, state_dim)   normalized state vectors
            n_tau    : int              number of τ samples per state
            tau_low  : float            lower bound of τ sampling range
                                        0.0 for risk-neutral
                                        0.0 for CVaR (upper = alpha)
            tau_high : float            upper bound of τ sampling range
                                        1.0 for risk-neutral
                                        alpha for CVaR_alpha

        Returns:
            quantiles: (B, N, A)   quantile values Z(s, τ_i, a)
            tau      : (B*N, 1)    the sampled τ values (needed for loss)
        """
        B = state.shape[0]
        N = n_tau
        device = state.device

        # ── Step 1: Encode state ──────────────────────────────────────
        # psi: (B, d)
        psi = self.state_encoder(state)

        # ── Step 2: Sample τ values ───────────────────────────────────
        # tau: (B*N, 1)  all τ values for this batch
        tau = torch.rand(B * N, 1, device=device) * (tau_high - tau_low) + tau_low

        # ── Step 3: Compute cosine quantile embedding ─────────────────
        # phi: (B*N, d)
        phi = self.quantile_embed(tau)

        # ── Step 4: Expand ψ(s) to match τ dimension ─────────────────
        # psi:        (B, d)
        # psi_expand: (B, 1, d) → repeat N times → (B, N, d)
        #           → reshape to (B*N, d)
        psi_expand = psi.unsqueeze(1).repeat(1, N, 1)  # (B, N, d)
        psi_expand = psi_expand.view(B * N, -1)         # (B*N, d)

        # ── Step 5: Element-wise product ψ(s) ⊙ φ(τ) ─────────────────
        # This is the key IQN operation — τ-conditioning via Hadamard product
        # combined: (B*N, d)
        combined = psi_expand * phi  # (B*N, d)

        # ── Step 6: Output MLP → quantile values ─────────────────────
        # q_flat:   (B*N, A)
        q_flat = self.output_mlp(combined)

        # ── Step 7: Reshape to (B, N, A) ─────────────────────────────
        quantiles = q_flat.view(B, N, self.cfg.n_actions)  # (B, N, A)

        return quantiles, tau

    @torch.no_grad()
    def get_action_values(
        self,
        state   : torch.Tensor,
        n_tau   : int,
        tau_low : float = 0.0,
        tau_high: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute expected Q-values for action selection.
        No gradient tracking needed — inference only.

        Q(s, a) = E_{τ ~ U[tau_low, tau_high]}[Z(s, τ, a)]
                ≈ (1/N) Σ_i z(s, τ_i, a)

        For IQN-neutral:  tau_low=0.0, tau_high=1.0  → E[Z]
        For IQN-CVaR_α:   tau_low=0.0, tau_high=α    → CVaR_α

        Args:
            state   : (B, state_dim) or (state_dim,) for single state
            n_tau   : number of τ samples
            tau_low : lower τ bound
            tau_high: upper τ bound

        Returns:
            q_values: (B, A)  mean quantile values per action
        """
        # Handle single state input (no batch dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)   # (1, state_dim)

        quantiles, _ = self.forward(state, n_tau, tau_low, tau_high)
        # quantiles: (B, N, A)
        # Mean over N tau samples → Q(s,a)
        q_values = quantiles.mean(dim=1)  # (B, A)
        return q_values