"""
envs/taq_env.py
---------------
Real-data execution environment backed by NYSE TAQ data.

Follows Ning et al. (2021) methodology:
    - Mid-price from NBBO quotes (no LOB walk)
    - Execution price = mid_price - eta * (volume / avg_volume)
    - Quadratic penalty for market impact
    - State: (t*, q*, Δp*, spread*, imb*)

Data: preprocessed parquet from extract_taq.py containing
1-minute bars with mid_price, spread, imbalance, volatility,
buy/sell volumes from TAQ NBBO + trades.

Citation:
    Ning, B., Lin, F.H.T. & Jaimungal, S. (2021).
    "Double Deep Q-Learning for Optimal Execution."
    Applied Mathematical Finance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from envs.base_env import EnvConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TAQConfig(EnvConfig):
    """Extends EnvConfig for TAQ data.

    (action_basis / action_fracs / use_rv_feature / rv_window are inherited from
    EnvConfig — B1 — so the v2 q0-grid + σ̂ feature are available here too.)
    """
    data_dir    : str   = 'data/processed'
    stock       : str   = 'AAPL'
    year        : int   = 2014
    lob_levels  : int   = 1       # Level 1 only (NBBO)

    # --- Design-v2 B4 (defaults = legacy 1-min bars, N=5, remaining basis) ---
    # bar_source: parquet filename override; None → legacy '{stock}_{year}.parquet'.
    #   v2 uses '{stock}_{year}_3min_adj.parquet' (built by data/build_taq_3min.py).
    # bar_minutes: resolution of the source bars, used to derive the step stride
    #   so a decision = one bar in v2 (3-min bars, N=20 → stride 1).
    bar_source  : str   = None
    bar_minutes : int   = 1

    # Execution horizon
    # N=10 periods over T=30 minutes → 3-min steps
    # Gives enough price movement between steps

    # Volume to liquidate (in shares)
    # AAPL 2014: avg 1-min volume ≈ 82,000 shares
    # q0 = 5000 shares ≈ 6% of 1-min volume (reasonable)

    # Impact model: p_exec = mid - eta * (x_t / avg_vol)
    # eta calibrated from data spread


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TAQEnv:
    """
    Execution environment replaying real NYSE TAQ data.

    Each episode = N consecutive bars starting from a random point.
    Execution price uses mid-price + linear temporary impact model.

    State: s_t = (t*, q*, Δp*, spread*, imb*) ∈ ℝ⁵
    """

    STATE_DIM = 5
    ACTION_FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def __init__(self, cfg: TAQConfig, dates: List[str]):
        """
        Args:
            cfg: TAQConfig with all parameters
            dates: list of date strings to use (for train/val/test split)
        """
        self.cfg = cfg

        # Load parquet — bar_source (v2, e.g. the 3-min split-adjusted file) or
        # the legacy 1-min file. Only the source parquet name changes; all the
        # replay logic below is shared.
        parquet_name = cfg.bar_source or f'{cfg.stock}_{cfg.year}.parquet'
        parquet_path = Path(cfg.data_dir) / parquet_name
        if not parquet_path.exists():
            raise FileNotFoundError(
                f'{parquet_path} not found. '
                f'Run extract_taq.py on WRDS Cloud first '
                f'(or data/build_taq_3min.py for the v2 3-min file).'
            )

        full_df = pd.read_parquet(parquet_path)

        # Filter to specified dates
        self.data = full_df[full_df['date'].isin(dates)].reset_index(drop=True)
        self.dates = sorted(dates)

        # Convert nullable types to float for numpy compatibility
        for col in ['mid_price', 'spread', 'best_bid', 'best_ask',
                     'buy_vol', 'sell_vol', 'imbalance', 'volatility']:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').astype(float)

        for col in ['best_bidsizeshares', 'best_asksizeshares', 'total_vol', 'trade_count']:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0).astype(float)

        # Drop rows with NaN mid_price
        self.data = self.data.dropna(subset=['mid_price']).reset_index(drop=True)

        # Compute average volume for impact scaling
        self.avg_vol = self.data['total_vol'].mean()
        if self.avg_vol < 1:
            self.avg_vol = 1.0

        # Action grid / basis (Design-v2 B4). Defaults = legacy 6-level grid,
        # 'remaining' basis → byte-identical. v2 passes the q0-grid + 'q0' basis.
        self._fracs  = list(cfg.action_fracs) if cfg.action_fracs is not None \
                       else list(self.ACTION_FRACS)
        self._basis  = getattr(cfg, 'action_basis', 'remaining')
        self._use_rv = getattr(cfg, 'use_rv_feature', False)

        # Step stride: how many source rows per decision step.
        #   legacy (1-min bars): T/N minutes per step → stride = int(T/N).
        #   v2 (bar_minutes-resolution bars): stride = round((T/N) / bar_minutes),
        #   so a decision = one bar when the bars are already at the step size.
        if cfg.bar_source is None:
            self.step_stride = max(1, int(cfg.T / cfg.N))           # legacy, unchanged
        else:
            self.step_stride = max(1, int(round(cfg.T / cfg.N / cfg.bar_minutes)))

        # Precompute valid starting indices
        needed_rows = cfg.N * self.step_stride
        self._valid_starts = []
        for date in self.dates:
            day_idx = self.data.index[self.data['date'] == date].tolist()
            if len(day_idx) >= needed_rows + 1:
                for i in range(len(day_idx) - needed_rows):
                    self._valid_starts.append(day_idx[i])

        if len(self._valid_starts) == 0:
            raise RuntimeError(
                f'No valid episodes for {cfg.stock} with N={cfg.N}, '
                f'stride={self.step_stride}. '
                f'Available days: {len(self.dates)}, rows: {len(self.data)}'
            )

        self.n_episodes = len(self.dates)

        # Environment interface (state_dim/n_actions config-driven; the σ̂ feature
        # adds one dim when use_rv_feature is on).
        self.state_dim = self.STATE_DIM + (1 if self._use_rv else 0)
        self.n_actions = len(self._fracs)

        # Episode state
        self._rng = np.random.RandomState(42)
        self._ep_start = 0
        self._t = 0
        self._q = 0.0
        self._p0 = 0.0
        self._rewards = []

        print(f'  [TAQEnv {cfg.stock}] {len(self.dates)} days, '
              f'{len(self._valid_starts)} valid starts, '
              f'stride={self.step_stride}, avg_vol={self.avg_vol:.0f}')

    def seed(self, s: int):
        self._rng = np.random.RandomState(s)

    @property
    def action_fracs(self) -> np.ndarray:
        return np.asarray(self._fracs, dtype=np.float64)

    @property
    def action_basis(self) -> str:
        return self._basis

    def reset(self) -> np.ndarray:
        cfg = self.cfg

        # Pick random starting point
        idx = self._rng.randint(len(self._valid_starts))
        self._ep_start = self._valid_starts[idx]

        # Arrival price
        row0 = self.data.iloc[self._ep_start]
        self._p0 = float(row0['mid_price'])
        self._q = float(cfg.q0)
        self._t = 0
        self._rewards = []

        return self._get_state()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        cfg = self.cfg
        # `action` is a grid index (learned/discrete agents) or a raw continuous
        # fraction of q0 (rule-based TWAP/AC executing exact amounts in q0 mode).
        if isinstance(action, (int, np.integer)):
            frac = self._fracs[action]
        else:
            frac = float(action)
        terminal = (self._t == cfg.N - 1)

        # Volume to execute. Terminal force-liquidates (cap-exempt, both bases).
        #   'remaining' (legacy): x_t = frac · q_remaining  (byte-identical).
        #   'q0'        (v2):     x_t = min(frac · q0, q_remaining)  (oversell clip).
        if self._basis == 'remaining':
            x_t = self._q if terminal else frac * self._q
        else:
            x_t = self._q if terminal else min(float(frac) * float(cfg.q0), self._q)
        x_t = max(x_t, 0.0)

        # Get current market snapshot
        row_idx = self._ep_start + self._t * self.step_stride
        row = self.data.iloc[row_idx]

        mid = float(row['mid_price'])

        # Fill model: execute at the mid-price minus a linear temporary impact
        # (p_exec = mid - eta * x_t). (N9: the half-spread "Ning et al." variant
        # that preceded this was dead code — it was always overwritten by this
        # block — so it has been removed; runtime behavior is unchanged.)
        if x_t > 1e-8:
            impact = cfg.eta * x_t
            p_exec = mid - impact
        else:
            p_exec = mid

        # Reward: r_t = x_t * (p_exec - p0) / (p0 * q0) - a * (x_t / q0)^2
        q0 = float(cfg.q0)
        is_contrib = x_t * (p_exec - self._p0) / (self._p0 * q0 + 1e-12)
        penalty = cfg.a * (x_t / (q0 + 1e-12)) ** 2
        reward = is_contrib - penalty

        self._rewards.append(reward)

        # Update inventory
        self._q -= x_t
        self._t += 1

        done = (self._t >= cfg.N) or (self._q <= 1e-8)

        # `at_cap`: traded exactly at the per-step cap on a non-terminal step —
        # the cap_frac diagnostic (metrics.py). Additive; does not affect IS.
        at_cap = (not terminal) and bool(np.isclose(float(frac), float(self._fracs[-1])))
        info = {'x_t': x_t, 'frac': float(frac), 'at_cap': at_cap}
        if done:
            info['implementation_shortfall'] = -sum(self._rewards)

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """Build state: (t*, q*, Δp*, spread*, imb*)"""
        cfg = self.cfg
        row_idx = self._ep_start + min(self._t, cfg.N - 1) * self.step_stride
        row = self.data.iloc[row_idx]

        t_star = self._t / cfg.N
        q_star = self._q / (cfg.q0 + 1e-12)

        mid = float(row['mid_price'])
        dp_star = (mid - self._p0) / (self._p0 + 1e-8)

        spread_star = float(row['spread']) / (mid + 1e-8)
        imb = float(row['imbalance'])

        feats = [t_star, q_star, dp_star, spread_star, imb]
        if self._use_rv:
            # σ̂_t: normalized 3-min realized vol (clip [0,5]), same shape as the
            # sim envs' rv feature. Off by default → legacy 5-D state unchanged.
            vol = float(row['volatility']) if 'volatility' in row else 0.0
            feats.append(float(np.clip(vol / (cfg.sigma + 1e-8), 0.0, 5.0)))

        return np.array(feats, dtype=np.float32)