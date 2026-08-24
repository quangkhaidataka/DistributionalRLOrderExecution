"""
envs/regime_jump_env.py
-----------------------
Design-v2 B3 — RegimeJumpEnv: a hidden two-state Markov volatility environment
with compound-Poisson jumps that fire ONLY in the stress regime.

This is the environment where IQN-CVaR's tail control is expected to pay off:
most of the time the market is calm (thin Gaussian), but a minority of stress
episodes carry BOTH elevated volatility AND fat-tailed jumps — a fat-tailed
Gaussian mixture the risk-neutral policy cannot hedge as well as CVaR.

Dynamics (per period t):
    1. Hidden regime R_t ∈ {0=calm, 1=stress} transitions via a 2-state chain:
           P(calm → stress) = p₀₁   (from config: cfg.p_01)
           P(stress → calm) = p₁₀   (FIXED = 0.40 → mean stress ≈ 2.5 periods)
    2. Volatility follows the regime:
           σ = σ_low   (FIXED = 0.0005)     in calm
           σ = σ_high  (from config)         in stress
    3. Price: p_{t+1} = p_t − γ·x_t + σ·√dt·ξ_t  (+ jumps in stress)
    4. Jumps (compound Poisson) ONLY in stress:
           N_t ~ Poisson(λ_J)  (per-period rate, from cfg.jump_intensity, e.g. 0.1)
           J_k ~ N(μ_J, σ_J²)  (cfg.jump_mean=0 symmetric, cfg.jump_std=0.16 $)
    5. Spread ×3 in stress (illiquidity), imbalance more extreme in stress.

FIXED constants (σ_low, p₁₀) are class attributes, NOT read from config, so the
calibration scan can only vary σ_high and p₀₁ (cfg.sigma_low / cfg.p_10 are
intentionally ignored). Jump params come from the config (a regime runner sets
jump_intensity=0.1, jump_std=0.16, jump_mean=0.0).

Existing env classes (AlmgrenChriss / MeanReverting / JumpDiffusion /
RegimeSwitching) are untouched — this is a NEW subclass of AlmgrenChrissEnv, so
it inherits all the B1 execution/masking/state machinery (works with the q0
action basis and use_rv_feature on/off).

Per-episode diagnostics exposed in step()'s info dict:
    'regime'     : int 0/1 (current regime after this period's transition)
    'stress'     : bool (regime == 1)
    'n_jumps'    : int  (jumps drawn THIS period; > 0 ⇒ stress)
    'spread'     : float (current spread level)
    'stress_hit' : bool (≥ 1 stress period so far this episode) — for the
                   stress-hit vs calm conditional breakdown (T-RG-3).
"""

from typing import Tuple

import numpy as np

from envs.simulated_env import AlmgrenChrissEnv, SimConfig


class RegimeJumpEnv(AlmgrenChrissEnv):
    """AC dynamics + hidden 2-state Markov vol + stress-only compound-Poisson jumps."""

    # ── FIXED regime constants (NOT config-driven) ────────────────────────
    SIGMA_LOW = 0.0005   # calm-regime volatility (fixed)
    P_10      = 0.40     # P(stress → calm) per period (fixed; mean stress 2.5 periods)

    def __init__(self, config: SimConfig):
        super().__init__(config)
        self.cfg: SimConfig = config
        # Episode regime state (properly (re)set in _init_price via reset()).
        self._regime       : int   = 0
        self._sigma        : float = self.SIGMA_LOW
        self._stress_hit   : bool  = False
        self._last_n_jumps : int   = 0
        self._last_spread  : float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _regime_sigma(self) -> float:
        return self.SIGMA_LOW if self._regime == 0 else self.cfg.sigma_high

    @property
    def pi_stress(self) -> float:
        """Stationary stress fraction π₁ = p₀₁ / (p₀₁ + p₁₀)."""
        return self.cfg.p_01 / (self.cfg.p_01 + self.P_10)

    @property
    def current_regime(self) -> str:
        return 'stress' if self._regime == 1 else 'calm'

    # ------------------------------------------------------------------
    # Abstract-method overrides
    # ------------------------------------------------------------------

    def _init_price(self) -> float:
        """Start each episode in the STATIONARY regime distribution.

        The initial regime R₀ only SEEDS the Markov chain (and sets the initial
        spread): _evolve_price transitions R₀→R₁ *before* the first price move,
        so R₀ never drives price/jumps. `stress_hit` therefore counts only the
        N price-evolution regimes R₁…R_N (each set in _evolve_price) — it starts
        False here and is consistent with the per-step info['stress'] flags.
        """
        self._imbalance    = 0.0
        self._regime       = int(self._rng.random() < self.pi_stress)
        self._sigma        = self._regime_sigma()
        self._stress_hit   = False
        self._last_n_jumps = 0
        self._last_spread  = 0.0
        return self.cfg.p0

    def _evolve_price(self, x_t: float) -> float:
        """Transition the hidden regime, then evolve price with regime vol + (stress) jumps."""
        # 1. Markov regime transition
        if self._regime == 0:
            self._regime = int(self._rng.random() < self.cfg.p_01)         # calm → stress
        else:
            self._regime = int(self._rng.random() > self.P_10)             # stay stress w.p. 1−p₁₀
        self._sigma = self._regime_sigma()
        if self._regime == 1:
            self._stress_hit = True

        # 2. Diffusion + permanent impact
        diffusion        = self._sigma * np.sqrt(self.cfg.dt)
        permanent_impact = self.cfg.gamma * x_t
        noise            = self._rng.standard_normal()

        # 3. Compound-Poisson jumps ONLY in stress (per-period Poisson rate)
        n_jumps = 0
        jump_total = 0.0
        if self._regime == 1:
            n_jumps = int(self._rng.poisson(self.cfg.jump_intensity))
            if n_jumps > 0:
                jumps = self._rng.normal(self.cfg.jump_mean, self.cfg.jump_std,
                                         size=n_jumps)
                jump_total = float(np.sum(jumps))
        self._last_n_jumps = n_jumps

        return self.p - permanent_impact + diffusion * noise + jump_total

    def _get_lob_features(self) -> Tuple[float, float]:
        """Spread ×3 in stress; imbalance more extreme in stress (illiquidity)."""
        stress_mult  = 1.0 if self._regime == 0 else 3.0
        spread_level = self.cfg.spread_base * self.cfg.p0 * stress_mult
        spread_noise = self.cfg.spread_noise * spread_level
        spread       = max(0.01, spread_level + self._rng.normal(0, spread_noise))

        imb_vol         = 0.1 if self._regime == 0 else 0.25
        kappa           = self.cfg.imb_mean_rev
        imb_noise       = self._rng.normal(0, imb_vol)
        self._imbalance = (self._imbalance * np.exp(-kappa * self.cfg.dt) + imb_noise)
        imbalance       = float(np.clip(self._imbalance, -1.0, 1.0))

        self._last_spread = spread
        return spread, imbalance

    # ------------------------------------------------------------------
    # Diagnostics in info
    # ------------------------------------------------------------------

    def step(self, action):
        state, reward, done, info = super().step(action)
        info['regime']     = int(self._regime)
        info['stress']     = bool(self._regime == 1)
        info['n_jumps']    = int(self._last_n_jumps)
        info['spread']     = float(self._last_spread)
        info['stress_hit'] = bool(self._stress_hit)
        return state, reward, done, info
