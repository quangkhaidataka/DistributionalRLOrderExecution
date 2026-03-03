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
from dataclasses import dataclass
from typing import Tuple, Dict, Any

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
    N: int   = 10       # number of decision periods
    T: float = 60.0     # total horizon in minutes

    # Inventory
    q0: int  = 100_000  # initial shares to liquidate
    p0: float = 100.0   # arrival (benchmark) price

    # Market impact (used by simulated envs)
    eta: float   = 2.5e-6   # temporary impact coefficient
    gamma: float = 2.5e-7   # permanent impact coefficient
    a: float     = 0.1      # quadratic penalty coefficient

    # Volatility
    sigma: float = 0.00095  # per-period volatility

    # Discount factor (close to 1 for short horizons)
    discount: float = 0.99

    # Realized vol window (for state feature σ̂_t)
    rv_window: int = 5

    @property
    def dt(self) -> float:
        """Length of each decision period in minutes."""
        return self.T / self.N


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

# Global action set: fractions of remaining inventory to execute.
# Index 0 = wait, Index 4 = liquidate everything now.
ACTION_FRACS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
N_ACTIONS    = len(ACTION_FRACS)


# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------

class BaseExecutionEnv(ABC):
    """
    Abstract execution environment.

    State space  (dim = 6):
        s_t = [t*, q_t*, Δp_t*, spread_t*, imb_t*, σ̂_t*]
        All features normalized to a stable range for neural nets.

    Action space (discrete, N_ACTIONS = 5):
        a_t ∈ {0,1,2,3,4}  →  φ(a) ∈ {0, 0.25, 0.50, 0.75, 1.0}
        x_t = φ(a_t) * q_t  shares executed this period

    Reward:
        r_t = x_t * (p_exec_t - p0) / (p0 * q0)
            = −normalized IS contribution for this period

    Subclasses must implement:
        _init_price()      → float
        _evolve_price()    → float
        _get_lob_features()→ Tuple[float, float]   (spread, imbalance)
    """

    STATE_DIM  = 6
    N_ACTIONS  = N_ACTIONS

    def __init__(self, config: EnvConfig):
        self.cfg  = config
        self._rng = np.random.default_rng()  # reproducible via seed()

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
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"

        # 1. Determine shares to execute
        #    Force full liquidation on the last period
        if self.t == self.cfg.N - 1:
            x_t = self.q
        else:
            x_t = ACTION_FRACS[action] * self.q

        # 2. Compute execution price and reward
        p_exec  = self._execution_price(x_t)
        reward  = self._compute_reward(x_t, p_exec)
        self._total_revenue += x_t * p_exec

        # 3. Update inventory
        self.q -= x_t
        self.q  = max(self.q, 0.0)   # numerical safety

        # 4. Evolve price to next period
        self.p = self._evolve_price(x_t)
        self._price_history.append(self.p)
        self.t += 1

        # 5. Check termination
        done = (self.t >= self.cfg.N) or (self.q <= 1e-8)

        # 6. Compute IS at end of episode for logging
        info = {
            'x_t'            : x_t,
            'p_exec'         : p_exec,
            'q_remaining'    : self.q,
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
        Normalized IS contribution for this period.
        r_t = x_t * (p_exec - p0) / (p0 * q0)
        Maximizing reward ≡ minimizing IS.
        """
        return (x_t * (p_exec - self.cfg.p0)) / (self.cfg.p0 * self.cfg.q0)

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
        Construct the 6-dimensional normalized state vector:
            s_t = [t*, q_t*, Δp_t*, spread_t*, imb_t*, σ̂_t*]
        """
        spread, imbalance = self._get_lob_features()

        t_norm    = self.t / self.cfg.N
        q_norm    = self.q / self.cfg.q0
        dp_norm   = (self.p - self.cfg.p0) / (self.cfg.p0 + 1e-8)
        spd_norm  = spread / (self.cfg.p0 + 1e-8)
        imb_norm  = float(np.clip(imbalance, -1.0, 1.0))
        rv_norm   = float(np.clip(self._realized_vol(), 0.0, 5.0))

        return np.array(
            [t_norm, q_norm, dp_norm, spd_norm, imb_norm, rv_norm],
            dtype=np.float32,
        )

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
        return self.STATE_DIM

    @property
    def n_actions(self) -> int:
        return self.N_ACTIONS

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"N={self.cfg.N}, q0={self.cfg.q0}, "
                f"T={self.cfg.T}min)")