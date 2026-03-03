"""
simulated_env.py
----------------
Two simulation models for Phase 1 validation.

Model 1 — AlmgrenChrissEnv:
    Standard Almgren-Chriss with Gaussian IS distribution.
    Purpose: verify IQN recovers the known AC analytical optimum.
    Expected result: CVaR advantage minimal (Gaussian tails).

Model 2 — RegimeSwitchingEnv:
    AC dynamics with hidden two-state Markov volatility.
    Purpose: first demonstration of IQN-CVaR advantage.
    Expected result: clear CVaR reduction vs DDQN/IQN-neutral
    because IS distribution is a fat-tailed Gaussian mixture.

Design note:
    Both classes inherit BaseExecutionEnv and only override
    the three abstract methods: _init_price, _evolve_price,
    _get_lob_features. All reward/state/IS logic is inherited.
    This ensures simulation and real env are evaluated identically.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from envs.base_env import BaseExecutionEnv, EnvConfig


# ---------------------------------------------------------------------------
# Extended config for simulated environments
# ---------------------------------------------------------------------------

@dataclass
class SimConfig(EnvConfig):
    """
    Extends EnvConfig with simulation-specific parameters.
    Parameters calibrated to match a liquid NASDAQ large-cap
    (AAPL-like) stock following Almgren et al. (2005).
    """
    # --- Regime-switching parameters (Model 2 only) ---
    sigma_low  : float = 0.0005   # normal regime vol  (~0.8x baseline)
    sigma_high : float = 0.003    # stress regime vol  (~3x baseline)
    p_01       : float = 0.05     # P(normal → stress) per period
    p_10       : float = 0.30     # P(stress → normal) per period
    # Expected regime durations:
    #   normal: 1/p_01  = 20 periods
    #   stress: 1/p_10  = 3.3 periods

    # --- LOB feature simulation ---
    # Spread modeled as sigma-proportional with noise
    spread_base  : float = 0.02   # base spread as fraction of price
    spread_noise : float = 0.3    # spread noise level
    # Imbalance modeled as mean-reverting OU process
    imb_mean_rev : float = 0.5    # imbalance mean-reversion speed


# ---------------------------------------------------------------------------
# Model 1: Standard Almgren-Chriss
# ---------------------------------------------------------------------------

class AlmgrenChrissEnv(BaseExecutionEnv):
    """
    Almgren-Chriss execution environment.

    Price dynamics:
        p_{t+1} = p_t - γ·x_t + σ·√dt·ξ_t,   ξ_t ~ N(0,1)

    Execution price (temporary impact):
        p_exec_t = p_t - η·(x_t / dt)

    IS distribution: Gaussian by construction.

    Use this environment to:
        1. Verify your reward/IS calculation is correct
        2. Confirm DDQN ≈ AC analytical solution
        3. Confirm IQN-neutral ≈ DDQN (same objective)
        4. Confirm IQN-CVaR ≈ IQN-neutral (no tail benefit
           in Gaussian world — this is expected and correct)
    """

    def __init__(self, config: SimConfig):
        super().__init__(config)
        self.cfg: SimConfig = config   # typed for IDE autocomplete
        # Imbalance state (OU process)
        self._imbalance: float = 0.0

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _init_price(self) -> float:
        """Start each episode at the configured arrival price."""
        self._imbalance = 0.0
        return self.cfg.p0

    def _evolve_price(self, x_t: float) -> float:
        """
        Arithmetic Brownian motion + permanent impact.
        p_{t+1} = p_t - γ·x_t + σ·√dt·ξ
        """
        diffusion        = self.cfg.sigma * np.sqrt(self.cfg.dt)
        permanent_impact = self.cfg.gamma * x_t
        noise            = self._rng.standard_normal()
        return self.p - permanent_impact + diffusion * noise

    def _get_lob_features(self) -> Tuple[float, float]:
        """
        Synthetic LOB features consistent with price level.

        Spread: proportional to sigma (wider in volatile markets)
        Imbalance: mean-reverting OU process in [-1, 1]
        """
        # Spread ~ sigma-proportional with multiplicative noise
        spread_level = self.cfg.spread_base * self.cfg.p0
        spread_noise = self.cfg.spread_noise * spread_level
        spread       = max(0.01, spread_level + self._rng.normal(0, spread_noise))

        # Imbalance: OU process  dI = -κ·I·dt + dW
        kappa           = self.cfg.imb_mean_rev
        imb_noise       = self._rng.normal(0, 0.1)
        self._imbalance = (self._imbalance * np.exp(-kappa * self.cfg.dt)
                           + imb_noise)
        imbalance = float(np.clip(self._imbalance, -1.0, 1.0))

        return spread, imbalance


# ---------------------------------------------------------------------------
# Model 2: Regime-Switching (Hidden Markov Volatility)
# ---------------------------------------------------------------------------

class RegimeSwitchingEnv(AlmgrenChrissEnv):
    """
    Almgren-Chriss with two-state hidden Markov volatility.

    Regime dynamics:
        R_t ∈ {0=normal, 1=stress}
        P(R_{t+1}=1 | R_t=0) = p_01  (enter stress)
        P(R_{t+1}=0 | R_t=1) = p_10  (exit stress)

    Volatility:
        σ_t = σ_low  if R_t = 0
        σ_t = σ_high if R_t = 1

    Key property:
        Agent does NOT observe R_t directly.
        It must infer regime from σ̂_t (realized vol in state).
        IQN learns wider return distribution in stress regimes,
        allowing CVaR-optimal policy to be more conservative
        when vol is high — something DDQN cannot express.

    IS distribution: Gaussian mixture → fat tails.
    This is where IQN-CVaR should first show clear advantage.
    """

    def __init__(self, config: SimConfig):
        super().__init__(config)
        self._regime : int   = 0      # 0=normal, 1=stress
        self._sigma  : float = config.sigma_low

    # ------------------------------------------------------------------
    # Override abstract methods
    # ------------------------------------------------------------------

    def _init_price(self) -> float:
        """
        Reset regime: start in normal with 80% probability.
        Stationary distribution: π_1 = p_01/(p_01+p_10) ≈ 0.14
        We start mostly in normal to match stationary dist.
        """
        self._imbalance = 0.0
        pi_stress       = self.cfg.p_01 / (self.cfg.p_01 + self.cfg.p_10)
        self._regime    = int(self._rng.random() < pi_stress)
        self._sigma     = (self.cfg.sigma_low if self._regime == 0
                           else self.cfg.sigma_high)
        return self.cfg.p0

    def _evolve_price(self, x_t: float) -> float:
        """
        1. Transition hidden regime via Markov chain
        2. Update volatility to match new regime
        3. Evolve price with current-regime volatility
        """
        # Markov regime transition
        if self._regime == 0:
            self._regime = int(self._rng.random() < self.cfg.p_01)
        else:
            self._regime = int(self._rng.random() > self.cfg.p_10)

        # Volatility follows regime
        self._sigma = (self.cfg.sigma_low if self._regime == 0
                       else self.cfg.sigma_high)

        # Price evolution with regime-dependent vol
        diffusion        = self._sigma * np.sqrt(self.cfg.dt)
        permanent_impact = self.cfg.gamma * x_t
        noise            = self._rng.standard_normal()
        return self.p - permanent_impact + diffusion * noise

    def _get_lob_features(self) -> Tuple[float, float]:
        """
        Spread widens in stress regime (illiquidity in crises).
        Imbalance becomes more extreme in stress regime.
        """
        # Regime-dependent spread multiplier
        stress_multiplier = 1.0 if self._regime == 0 else 3.0
        spread_level      = self.cfg.spread_base * self.cfg.p0 * stress_multiplier
        spread_noise      = self.cfg.spread_noise * spread_level
        spread            = max(0.01, spread_level + self._rng.normal(0, spread_noise))

        # Imbalance: more extreme in stress (one-sided flow)
        imb_vol         = 0.1 if self._regime == 0 else 0.25
        kappa           = self.cfg.imb_mean_rev
        imb_noise       = self._rng.normal(0, imb_vol)
        self._imbalance = (self._imbalance * np.exp(-kappa * self.cfg.dt)
                           + imb_noise)
        imbalance = float(np.clip(self._imbalance, -1.0, 1.0))

        return spread, imbalance

    @property
    def current_regime(self) -> str:
        """Expose regime for analysis/debugging (not available to agent)."""
        return 'stress' if self._regime == 1 else 'normal'