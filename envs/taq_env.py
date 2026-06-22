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
    """Extends EnvConfig for TAQ data."""
    data_dir    : str   = 'data/processed'
    stock       : str   = 'AAPL'
    year        : int   = 2014
    lob_levels  : int   = 1       # Level 1 only (NBBO)

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

        # Load parquet
        parquet_path = Path(cfg.data_dir) / f'{cfg.stock}_{cfg.year}.parquet'
        if not parquet_path.exists():
            raise FileNotFoundError(
                f'{parquet_path} not found. '
                f'Run extract_taq.py on WRDS Cloud first.'
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

        # Step stride: how many 1-min rows per decision step
        # T minutes / N steps = minutes per step
        self.step_stride = max(1, int(cfg.T / cfg.N))

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

        # Environment interface
        self.state_dim = self.STATE_DIM
        self.n_actions = len(self.ACTION_FRACS)

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        cfg = self.cfg
        frac = self.ACTION_FRACS[action]

        # Volume to execute
        if self._t == cfg.N - 1:
            x_t = self._q  # force liquidate remaining
        else:
            x_t = frac * self._q
        x_t = max(x_t, 0.0)

        # Get current market snapshot
        row_idx = self._ep_start + self._t * self.step_stride
        row = self.data.iloc[row_idx]

        mid = float(row['mid_price'])
        spread = float(row['spread'])

        # Execution price: Ning et al. (2021) approach
        # Sell at mid - half_spread - temporary_impact
        # temporary_impact = eta * (x_t / avg_vol)
        if x_t > 1e-8:
            half_spread = spread / 2.0
            impact = cfg.eta * (x_t / (self.avg_vol + 1e-8))
            p_exec = mid - half_spread - impact
        else:
            p_exec = mid

        # Reward: IS contribution - quadratic penalty
        q0 = float(cfg.q0)
        is_contrib = x_t * (p_exec - self._p0) / (self._p0 * q0 + 1e-12)
        penalty = cfg.a * (x_t / (q0 + 1e-12)) ** 2
        reward = is_contrib - penalty

        # With:
        # Execution price: mid-price with temporary impact
        if x_t > 1e-8:
            impact = cfg.eta * x_t
            p_exec = mid - impact
        else:
            p_exec = mid

        # Reward: IS-based (same as simulation setting)
        # r_t = x_t * (p_exec - p0) / (p0 * q0) - a * (x_t / q0)^2
        q0 = float(cfg.q0)
        is_contrib = x_t * (p_exec - self._p0) / (self._p0 * q0 + 1e-12)
        penalty = cfg.a * (x_t / (q0 + 1e-12)) ** 2
        reward = is_contrib - penalty

        self._rewards.append(reward)

        # Update inventory
        self._q -= x_t
        self._t += 1

        done = (self._t >= cfg.N) or (self._q <= 1e-8)

        info = {}
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

        return np.array([t_star, q_star, dp_star, spread_star, imb],
                        dtype=np.float32)