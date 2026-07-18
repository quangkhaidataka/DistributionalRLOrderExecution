"""
base_env.py
-----------
Abstract base class for all execution environments.

Design philosophy:
    - Defines the contract every environment must fulfill
    - Gym-style interface (reset / step) so training loop
      never needs to know which environment it is running
    - All environments share the same state/action/reward
      semantics defined in the paper's MDP formulation
    - Subclasses only need to implement price dynamics;
      inventory logic, IS calculation, and action mapping
      live here once and are inherited everywhere
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """
    Single configuration object passed to every environment.
    Using a dataclass (instead of raw dict) gives us:
        - Type hints → catches bugs early
        - Auto-generated __repr__ → easy logging
        - IDE autocomplete
    """
    # Execution horizon
    N: int   = 5       # number of decision periods
    T: float = 60.0     # total horizon in minutes

    # Inventory
    q0: int  = 100_000  # initial shares to liquidate
    p0: float = 100.0   # arrival (benchmark) price

    # Market impact (used by simulated envs)
    eta: float   = 2.5e-6   # temporary impact coefficient
    gamma: float = 2.5e-7   # permanent impact coefficient
    a: float     = 0.001      # quadratic penalty coefficient
    penalty_type: str = 'quadratic' 
    # Volatility
    sigma: float = 0.00095  # per-period volatility

    # Discount factor (close to 1 for short horizons)
    discount: float = 0.99

    # --- Action representation (Design-v2 B1; DEFAULTS = legacy behaviour) ---
    # action_basis:
    #   'remaining' (default, legacy) → executes ACTION_FRACS[a] * q_t
    #                                   (a fraction of REMAINING inventory)
    #   'q0'        (Design-v2)       → executes min(ACTION_FRACS[a] * q0, q_t)
    #                                   (a fraction of the ORIGINAL block, clipped)
    # action_fracs: None → use the module-level ACTION_FRACS (legacy 6-level grid).
    #   In 'q0' mode the max frac is the per-step *cap* (e.g. 0.25).
    action_basis : str                     = 'remaining'
    action_fracs : Optional[Sequence[float]] = None

    # --- State feature toggle (Design-v2 B1) ---
    # use_rv_feature=False (default, legacy) → 5-D state; True → append σ̂_t (6-D).
    use_rv_feature : bool = False
    rv_window      : int  = 5     # realized-vol window (for state feature σ̂_t)

    @property
    def dt(self) -> float:
        """Length of each decision period in minutes."""
        return self.T / self.N


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

# Global action set: fractions of remaining inventory to execute.
# Index 0 = wait, Index 5 = liquidate everything now.
# ACTION_FRACS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
ACTION_FRACS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)


N_ACTIONS    = len(ACTION_FRACS)


# ---------------------------------------------------------------------------
# Feasible-action mask  (Design-v2 B1 — the SINGLE source of truth for masking)
# ---------------------------------------------------------------------------

def feasible_action_mask(
    q_norm      : "float | np.ndarray",
    action_fracs: Sequence[float],
    action_basis: str = 'q0',
) -> np.ndarray:
    """Boolean feasibility mask over the discrete action grid.

    This is the ONE function every masking site shares — the env's oversell
    protection, the agents' ε-greedy sampling / greedy argmax, and the Bellman
    target-max all mask through here. Keeping a single implementation is what
    makes the T3 "both sites share one mask function" invariant hold.

    Args:
        q_norm       : q_t / q0, the remaining fraction of the ORIGINAL block.
                       Scalar → returns (A,); array (B,) → returns (B, A).
        action_fracs : the action grid (fractions), assumed sorted ascending.
        action_basis : 'remaining' → every action a·q_t ≤ q_t (a ≤ 1) is always
                          feasible, so the mask is all-True (legacy: no masking).
                       'q0'        → feasible = {a: frac ≤ q_norm}
                          ∪ {smallest frac > q_norm — the remainder-seller reached
                             via the clip x = min(frac·q0, q_t)} ∪ {action 0}.

    Returns:
        bool ndarray of shape (A,) for scalar q_norm, else (B, A).
    """
    fracs  = np.asarray(action_fracs, dtype=np.float64)
    A      = fracs.shape[0]
    scalar = np.ndim(q_norm) == 0
    q      = np.atleast_1d(np.asarray(q_norm, dtype=np.float64))   # (B,)

    if action_basis == 'remaining':
        mask = np.ones((q.shape[0], A), dtype=bool)
    else:
        eps = 1e-9
        le  = fracs[None, :] <= q[:, None] + eps      # frac·q0 ≤ q_t  (includes a=0)
        gt  = fracs[None, :] >  q[:, None] + eps       # frac·q0 > q_t
        mask = le.copy()
        # Smallest strictly-greater frac per row (fracs sorted ascending → first
        # True of `gt`). That action clips to sell the exact remainder q_t.
        has_gt   = gt.any(axis=1)
        first_gt = np.argmax(gt, axis=1)               # 0 where has_gt is False
        rows     = np.where(has_gt)[0]
        mask[rows, first_gt[rows]] = True
        mask[:, 0] = True                              # action 0 (wait) always feasible

    return mask[0] if scalar else mask


# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------

class BaseExecutionEnv(ABC):
    """
    Abstract execution environment.

    State space  (dim = 5):
        s_t = [t*, q_t*, Δp_t*, spread_t*, imb_t*]
        (the realized-vol feature σ̂_t is currently disabled — see
        _build_state; the state is 5-D, matching the paper's R^5 encoder.)
        All features normalized to a stable range for neural nets.

    Action space (discrete, N_ACTIONS = 6):
        a_t ∈ {0,1,2,3,4,5}  →  φ(a) ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}
        x_t = φ(a_t) * q_t  shares executed this period

    Reward:
        r_t = x_t * (p_exec_t - p0) / (p0 * q0)
            = −normalized IS contribution for this period

    Subclasses must implement:
        _init_price()      → float
        _evolve_price()    → float
        _get_lob_features()→ Tuple[float, float]   (spread, imbalance)
    """

    STATE_DIM  = None
    N_ACTIONS  = N_ACTIONS

    def __init__(self, config: EnvConfig):
        self.cfg  = config
        self._rng = np.random.default_rng()  # reproducible via seed()

        # Per-instance action grid (Design-v2 B1). Default = module ACTION_FRACS,
        # so legacy configs are byte-identical. The grid is the single source for
        # both the env's execution and the agents' feasibility masks.
        fracs = config.action_fracs if config.action_fracs is not None else ACTION_FRACS
        self._action_fracs : np.ndarray = np.asarray(fracs, dtype=np.float32)

        # Per-instance state dimension (Design-v2 B1). Was a cached class attr,
        # which broke when 5-D and 6-D envs coexisted in one process. Base state
        # is 5 features; the σ̂_t realized-vol feature adds one when enabled.
        self._state_dim : int = 5 + (1 if config.use_rv_feature else 0)

        # Episode state (set properly in reset())
        self.t              : int   = 0
        self.q              : float = 0.0
        self.p              : float = 0.0
        self._price_history : list  = []
        self._total_revenue : float = 0.0

    # ------------------------------------------------------------------
    # Public interface (used by training loop and evaluation)
    # ------------------------------------------------------------------

    def seed(self, seed: int) -> None:
        """Set RNG seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        """
        Reset environment to start of a new episode.
        Returns initial state vector s_0.
        """
        self.t              = 0
        self.q              = float(self.cfg.q0)
        self.p              = self._init_price()
        self._price_history = [self.p]
        self._total_revenue = 0.0
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one decision period.

        Args:
            action: integer index in {0, 1, 2, 3, 4}

        Returns:
            next_state : np.ndarray  shape (STATE_DIM,)
            reward     : float       normalized IS contribution
            done       : bool        True if episode is complete
            info       : dict        diagnostic quantities
        """
        fracs    = self._action_fracs
        terminal = (self.t == self.cfg.N - 1)

        # 1. Determine shares to execute.
        #    Force full liquidation on the last period (cap-exempt in both bases).
        if self.cfg.action_basis == 'remaining':
            # ── Legacy path: fraction of REMAINING inventory (byte-identical) ──
            # Keep `fracs[action] * self.q` (numpy float32 × Python float → float32
            # under NEP-50 weak promotion) exactly as the pre-refactor code did;
            # widening to a Python float here silently upgrades the whole IS
            # computation to float64 and breaks the T1 regression gate.
            assert 0 <= action < len(fracs), f"Invalid action {action}"
            x_t  = self.q if terminal else fracs[action] * self.q
            frac = float(fracs[action])   # for the at_cap / info fields only
        else:
            # ── Design-v2 'q0' path: fraction of the ORIGINAL block, clipped ──
            # `action` may be a grid index (learned/discrete agents) or a raw
            # continuous fraction of q0 (rule-based AC/TWAP executing exact
            # sinh/uniform amounts, not grid-projected).
            if isinstance(action, (int, np.integer)):
                assert 0 <= action < len(fracs), f"Invalid action {action}"
                frac = float(fracs[action])
            else:
                frac = float(action)
            x_t = self.q if terminal else min(frac * self.cfg.q0, self.q)

        # 2. Compute execution price and reward
        p_exec  = self._execution_price(x_t)
        reward  = self._compute_reward(x_t, p_exec)
        self._total_revenue += x_t * p_exec

        # 3. Update inventory
        self.q -= x_t
        self.q  = max(self.q, 0.0)   # numerical safety
        assert self.q >= -1e-6, f"Oversell: q={self.q}"

        # 4. Evolve price to next period
        self.p = self._evolve_price(x_t)
        self._price_history.append(self.p)
        self.t += 1

        # 5. Check termination
        done = (self.t >= self.cfg.N) or (self.q <= 1e-8)

        # 6. Compute IS at end of episode for logging.
        #    `at_cap`: traded exactly at the per-step cap (max grid frac) on a
        #    non-terminal step — the input to the cap_frac diagnostic (metrics.py).
        at_cap = (not terminal) and bool(np.isclose(frac, float(fracs[-1])))
        info = {
            'x_t'            : x_t,
            'p_exec'         : p_exec,
            'q_remaining'    : self.q,
            'frac'           : frac,
            'at_cap'         : at_cap,
            'implementation_shortfall': self._compute_is() if done else None,
        }

        return self._build_state(), float(reward), bool(done), info

    def compute_twap_is(self) -> float:
        """
        Reference IS for TWAP policy (equal shares each period).
        Used as benchmark in evaluation metrics.
        Computed analytically given current price path.
        """
        x_twap = self.cfg.q0 / self.cfg.N
        # TWAP revenue = sum of (x_twap * p_t) for each period
        # Under pure Brownian motion, E[TWAP IS] = 0 by definition
        # In practice we compare realized IS vs realized TWAP IS
        return 0.0   # overridden in evaluation with realized comparison

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execution_price(self, x_t: float) -> float:
        """
        Price at which x_t shares are executed.
        Default: mid-price minus temporary impact.
        Subclasses can override for more realistic fill models.
        """
        return self.p - self.cfg.eta * (x_t / self.cfg.dt)

    def _compute_reward(self, x_t: float, p_exec: float) -> float:
        """
        r_t = x_t*(p_exec - p0)/(p0*q0) - penalty
        penalty = a*(x_t/q0)^2  if quadratic
        penalty = a*(x_t/q0)    if linear
        """
        is_contribution = (x_t * (p_exec - self.cfg.p0)) / (self.cfg.p0 * self.cfg.q0)
        frac = x_t / self.cfg.q0
        if self.cfg.penalty_type == 'linear':
            trade_penalty = self.cfg.a * frac
        else:
            trade_penalty = self.cfg.a * frac ** 2
        return is_contribution - trade_penalty


    def _compute_is(self) -> float:
        """
        Total implementation shortfall at episode end.
        IS = (p0 * q0 - total_revenue) / (p0 * q0)
        Positive IS = we received less than arrival benchmark.
        """
        benchmark_revenue = self.cfg.p0 * self.cfg.q0
        return (benchmark_revenue - self._total_revenue) / benchmark_revenue

    def _realized_vol(self) -> float:
        """
        Rolling realized volatility over last rv_window periods.
        Normalized by long-run sigma so feature is O(1).
        This is the agent's signal about the hidden volatility regime.
        (Only appended to the state when cfg.use_rv_feature is True.)
        """
        w = self.cfg.rv_window
        if len(self._price_history) < 2:
            return 1.0   # prior: assume normal regime
        recent  = self._price_history[-w:]
        returns = np.diff(recent) / (recent[0] + 1e-8)
        rv      = float(np.std(returns))
        # Normalize by long-run sigma
        return rv / (self.cfg.sigma + 1e-8)

    def _build_state(self) -> np.ndarray:
        """
        Construct the normalized state vector:
            s_t = [t*, q_t*, Δp_t*, spread_t*, imb_t*]   (5-D, default)
        and, when cfg.use_rv_feature is True, append the realized-vol feature:
            s_t = [t*, q_t*, Δp_t*, spread_t*, imb_t*, σ̂_t*]   (6-D)
        """
        spread, imbalance = self._get_lob_features()

        t_norm    = self.t / self.cfg.N
        q_norm    = self.q / self.cfg.q0
        dp_norm   = (self.p - self.cfg.p0) / (self.cfg.p0 + 1e-8)
        spd_norm  = spread / (self.cfg.p0 + 1e-8)
        imb_norm  = float(np.clip(imbalance, -1.0, 1.0))

        feats = [t_norm, q_norm, dp_norm, spd_norm, imb_norm]
        if self.cfg.use_rv_feature:
            feats.append(float(np.clip(self._realized_vol(), 0.0, 5.0)))

        return np.array(feats, dtype=np.float32)

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_price(self) -> float:
        """Return initial price p_0 at episode start."""
        ...

    @abstractmethod
    def _evolve_price(self, x_t: float) -> float:
        """
        Advance price from p_t to p_{t+1}.
        x_t is passed so subclasses can model permanent impact.
        """
        ...

    @abstractmethod
    def _get_lob_features(self) -> Tuple[float, float]:
        """
        Return (spread, order_book_imbalance) for current period.
        Simulated envs generate these synthetically.
        Real env reads them from LOBSTER data.
        """
        ...

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def n_actions(self) -> int:
        return len(self._action_fracs)

    @property
    def action_fracs(self) -> np.ndarray:
        """The per-instance action grid (fractions)."""
        return self._action_fracs

    @property
    def action_basis(self) -> str:
        return self.cfg.action_basis

    def action_mask(self, state: np.ndarray) -> np.ndarray:
        """Feasibility mask for the (possibly batched) normalized state.

        Reads q* = state[..., 1] (the remaining fraction of the original block)
        and delegates to the module-level ``feasible_action_mask`` — so the env
        and the agents mask through the exact same code path.
        """
        q_norm = np.asarray(state)[..., 1]
        return feasible_action_mask(q_norm, self._action_fracs, self.cfg.action_basis)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"N={self.cfg.N}, q0={self.cfg.q0}, "
                f"T={self.cfg.T}min)")