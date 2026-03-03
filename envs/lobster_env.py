"""
lobster_env.py
--------------
Real-data execution environment backed by LOBSTER LOB data.

How it works:
    - Each episode = one trading day for one stock
    - Environment replays historical LOB snapshots at 1-min intervals
    - Agent's market orders are assumed small enough not to move
      the historical price path (standard academic assumption,
      consistent with Ning et al. 2021)
    - Execution price = mid-price minus temporary impact model
      (same as simulated env — ensures fair comparison)

LOBSTER file format:
    message file:   time, type, order_id, size, price, direction
    orderbook file: ask1, askvol1, bid1, bidvol1, ... (10 levels)

Memory strategy for MacBook Air (8GB RAM):
    - Raw CSVs live on external SSD / data/raw/
    - Preprocessed parquet lives in data/processed/
    - LOBSTERLoader converts raw → parquet once
    - LobsterEnv loads one day at a time (~0.5MB per episode)
    - Training loop never holds more than one episode in memory
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from envs.base_env import BaseExecutionEnv, EnvConfig


# ---------------------------------------------------------------------------
# LOBSTER-specific config
# ---------------------------------------------------------------------------

@dataclass
class LobsterConfig(EnvConfig):
    """
    Extends EnvConfig with LOBSTER-specific parameters.
    """
    data_dir    : str  = 'data/processed'
    stock       : str  = 'AAPL'
    train_years : list = None   # e.g. [2019, 2020, 2021]
    test_years  : list = None   # e.g. [2022]
    lob_levels  : int  = 5      # how many LOB levels to use (max 10)
    resample_freq: str = '1min' # LOB snapshot frequency

    def __post_init__(self):
        if self.train_years is None:
            self.train_years = [2019, 2020, 2021]
        if self.test_years is None:
            self.test_years  = [2022]


# ---------------------------------------------------------------------------
# Data loader — run once to preprocess raw LOBSTER files
# ---------------------------------------------------------------------------

class LOBSTERLoader:
    """
    Converts raw LOBSTER CSV files to memory-efficient parquet.

    Usage (run once before training):
        loader = LOBSTERLoader('data/', 'AAPL')
        loader.preprocess(year=2019)

    Output:
        data/processed/AAPL_2019.parquet
        Columns: date, time, mid_price, spread, imbalance,
                 ask1..ask5, bid1..bid5, askvol1..bidvol5
        Size: ~5MB per stock per year (vs ~15GB raw)
    """

    # Columns we keep from the 10-level orderbook
    # (5 levels sufficient; deeper levels add noise for 1min bars)
    KEEP_LEVELS = 5

    def __init__(self, data_dir: str, stock: str):
        self.data_dir = Path(data_dir)
        self.stock    = stock
        self.raw_dir  = self.data_dir / 'raw'  / stock
        self.out_dir  = self.data_dir / 'processed'
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, year: int) -> Path:
        """
        Process all days for one stock-year into a single parquet file.
        Safe to call multiple times — skips if output exists.
        """
        out_path = self.out_dir / f'{self.stock}_{year}.parquet'
        if out_path.exists():
            print(f'[LOBSTERLoader] {out_path} already exists, skipping.')
            return out_path

        daily_frames = []
        msg_files    = sorted(self.raw_dir.glob(f'*{year}*message*.csv'))

        if not msg_files:
            raise FileNotFoundError(
                f'No LOBSTER message files found in {self.raw_dir} '
                f'for year {year}. '
                f'Download from https://lobsterdata.com and place in '
                f'data/raw/{self.stock}/'
            )

        for msg_path in msg_files:
            date_str  = msg_path.stem.split('_')[0]
            lob_path  = Path(str(msg_path).replace('message', 'orderbook'))

            if not lob_path.exists():
                print(f'[LOBSTERLoader] Warning: no orderbook for {date_str}, skipping.')
                continue

            try:
                df = self._process_day(msg_path, lob_path, date_str)
                if df is not None and len(df) >= self.KEEP_LEVELS:
                    daily_frames.append(df)
            except Exception as e:
                print(f'[LOBSTERLoader] Error processing {date_str}: {e}')
                continue

        if not daily_frames:
            raise RuntimeError(f'No valid trading days found for {self.stock} {year}.')

        combined = pd.concat(daily_frames, ignore_index=True)
        combined.to_parquet(out_path, compression='snappy', index=False)

        size_mb = out_path.stat().st_size / 1e6
        print(f'[LOBSTERLoader] Saved {out_path} '
              f'({len(daily_frames)} days, {size_mb:.1f} MB)')
        return out_path

    def _process_day(self,
                     msg_path : Path,
                     lob_path : Path,
                     date_str : str) -> Optional[pd.DataFrame]:
        """
        Process one trading day:
            1. Load raw message + orderbook CSV files
            2. Compute mid-price, spread, imbalance
            3. Resample to 1-minute bars (reduces size ~100x)
            4. Filter to core trading hours (10:00–15:30)
               to avoid open/close auction noise
        """
        # --- Load message file ---
        msg_cols = ['time', 'type', 'order_id', 'size', 'price', 'direction']
        msg      = pd.read_csv(msg_path, header=None, names=msg_cols)

        # --- Load orderbook file ---
        # LOBSTER orderbook: ask1, askvol1, bid1, bidvol1, ... (10 levels × 4)
        n_cols   = self.KEEP_LEVELS * 4
        lob_cols = []
        for lvl in range(1, self.KEEP_LEVELS + 1):
            lob_cols += [f'ask{lvl}', f'askvol{lvl}', f'bid{lvl}', f'bidvol{lvl}']

        # Read only the columns we need
        lob = pd.read_csv(lob_path, header=None,
                          usecols=range(n_cols),
                          names=lob_cols)

        if len(msg) != len(lob):
            return None   # corrupted file

        # --- Align on time index ---
        base_date = pd.Timestamp(date_str)
        df        = pd.concat([msg[['time', 'price', 'size']], lob], axis=1)
        df['datetime'] = base_date + pd.to_timedelta(df['time'], unit='s')
        df = df.set_index('datetime').sort_index()

        # --- Compute derived features ---
        # LOBSTER prices are in cents → convert to dollars
        price_cols = [c for c in lob_cols if 'vol' not in c]
        df[price_cols] = df[price_cols] / 10_000

        df['mid_price']  = (df['ask1'] + df['bid1']) / 2
        df['spread']     = (df['ask1'] - df['bid1']).clip(lower=0)
        df['imbalance']  = ((df['bid_vol1'] - df['askvol1']) /
                            (df['bid_vol1'] + df['askvol1'] + 1e-8))

        # --- Resample to 1-minute bars (last observation per bar) ---
        feature_cols = ['mid_price', 'spread', 'imbalance'] + lob_cols
        df           = df[feature_cols].resample('1min').last().dropna()

        # --- Filter to core trading hours: 10:00 – 15:30 ---
        df = df.between_time('10:00', '15:30')

        if len(df) < 10:   # skip days with too few bars
            return None

        df['date']  = date_str
        df['stock'] = self.stock
        df          = df.reset_index()
        df.rename(columns={'datetime': 'time'}, inplace=True)

        return df


# ---------------------------------------------------------------------------
# LOBSTER execution environment
# ---------------------------------------------------------------------------

class LobsterEnv(BaseExecutionEnv):
    """
    Execution environment backed by real LOBSTER LOB data.

    Episode structure:
        - One episode = one trading day
        - Each decision period = (total_minutes / N) minutes
          of real LOB data, sampled at 1-min resolution
        - The agent executes x_t shares at each decision period
        - Execution price = mid_price_t - temporary impact

    Identical reward/state/IS logic as SimulatedEnv (inherited).
    Only _init_price, _evolve_price, _get_lob_features differ.
    This guarantees the evaluation protocol is identical across
    simulated and real environments — a key methodological point.
    """

    def __init__(self, config: LobsterConfig, mode: str = 'train'):
        """
        Args:
            config: LobsterConfig with data paths and parameters
            mode:   'train' or 'test' — selects year split
        """
        super().__init__(config)
        self.cfg  : LobsterConfig = config
        self.mode : str           = mode

        # Load all episodes into memory as list of DataFrames
        # Each DataFrame = one trading day
        years         = config.train_years if mode == 'train' else config.test_years
        self._episodes: List[pd.DataFrame] = self._load_episodes(years)
        self._ep_idx  : int                = 0
        self._current_day: Optional[pd.DataFrame] = None
        self._step_idx: int                = 0

        # Shuffle training episodes for better learning
        if mode == 'train':
            self._rng.shuffle(self._episodes)

        print(f'[LobsterEnv] Loaded {len(self._episodes)} episodes '
              f'for {config.stock} ({mode})')

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _init_price(self) -> float:
        """
        Load next episode (trading day) and return arrival price.
        Arrival price = mid-price at start of execution window.
        """
        # Cycle through episodes (wrap around at end)
        self._current_day = self._episodes[self._ep_idx % len(self._episodes)]
        self._ep_idx     += 1
        self._step_idx    = 0

        # Sample a random start point within the day
        # (agent doesn't always start at 10:00)
        max_start = max(0, len(self._current_day) - self.cfg.N - 1)
        self._start_idx = int(self._rng.integers(0, max_start + 1))
        self._step_idx  = self._start_idx

        arrival_price = float(
            self._current_day.iloc[self._step_idx]['mid_price']
        )
        # Update config arrival price for IS calculation
        self.cfg.p0 = arrival_price
        return arrival_price

    def _evolve_price(self, x_t: float) -> float:
        """
        Advance to next LOB snapshot.
        Price is historical mid-price — we do not model our own
        permanent impact on historical data (standard assumption).
        """
        self._step_idx += 1
        # Guard against end of day
        if self._step_idx >= len(self._current_day):
            self._step_idx = len(self._current_day) - 1

        return float(self._current_day.iloc[self._step_idx]['mid_price'])

    def _get_lob_features(self) -> Tuple[float, float]:
        """
        Read real spread and imbalance from current LOB snapshot.
        """
        if self._current_day is None or self._step_idx >= len(self._current_day):
            return 0.01, 0.0   # fallback

        row       = self._current_day.iloc[self._step_idx]
        spread    = float(row.get('spread',    0.01))
        imbalance = float(row.get('imbalance', 0.0))
        return spread, float(np.clip(imbalance, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_episodes(self, years: list) -> List[pd.DataFrame]:
        """
        Load preprocessed parquet files for the given years.
        Returns list of per-day DataFrames.

        Memory: ~0.5 MB per day × ~250 trading days × 3 years
               ≈ 375 MB total for one stock — fine for 8GB RAM.
        """
        data_dir = Path(self.cfg.data_dir)
        episodes = []

        for year in years:
            path = data_dir / f'{self.cfg.stock}_{year}.parquet'
            if not path.exists():
                raise FileNotFoundError(
                    f'Preprocessed data not found: {path}\n'
                    f'Run LOBSTERLoader.preprocess({year}) first.'
                )
            df = pd.read_parquet(path)

            # Split by date into per-day episodes
            for date, day_df in df.groupby('date'):
                day_df = day_df.reset_index(drop=True)
                if len(day_df) >= self.cfg.N:
                    episodes.append(day_df)

        if not episodes:
            raise RuntimeError(
                f'No valid episodes found for {self.cfg.stock} '
                f'in years {years}.'
            )
        return episodes

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    def set_mode(self, mode: str) -> None:
        """Switch between train and test splits."""
        self.mode    = mode
        years        = (self.cfg.train_years if mode == 'train'
                        else self.cfg.test_years)
        self._episodes = self._load_episodes(years)
        self._ep_idx   = 0
        if mode == 'train':
            self._rng.shuffle(self._episodes)