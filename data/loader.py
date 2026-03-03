"""
data/loader.py
==============
LOBSTER raw data → preprocessed parquet pipeline.

Quick-start
-----------
Run ONCE before training:

    from data.loader import LOBSTERLoader, load_episodes

    loader = LOBSTERLoader(raw_dir='data/raw', processed_dir='data/processed')
    loader.preprocess('AAPL', 2019)   # repeat for each stock / year
    loader.preprocess('AAPL', 2020)
    ...

    # LobsterEnv calls this to get its episode pool
    episodes = load_episodes(
        processed_dir = 'data/processed',
        stocks        = ['AAPL', 'MSFT'],
        years         = [2019, 2020, 2021],
    )
    # episodes : List[pd.DataFrame], one DataFrame per trading day

Verify a processed file:

    from data.loader import verify_parquet
    verify_parquet('data/processed', 'AAPL', 2019)

─────────────────────────────────────────────────────────────────────────────
LOBSTER raw file format  (official spec → lobsterdata.com/info/DataStructure)
─────────────────────────────────────────────────────────────────────────────
Filename convention (single-day download):
    <TICKER>_<DATE>_<DATE>_<LEVELS>_message_<N>.csv
    <TICKER>_<DATE>_<DATE>_<LEVELS>_orderbook_<N>.csv
    e.g.  AAPL_2019-01-02_2019-01-02_10_message_1.csv

Message file — N × 6 matrix, NO header:
    Col 0  time        float64   seconds after midnight, nanosecond precision
    Col 1  event_type  int8      1=new limit  2=cancel partial  3=delete
                                 4=execute visible  5=execute hidden
                                 6=cross trade  7=trading halt
    Col 2  order_id    int64     unique order reference
    Col 3  size        int32     shares
    Col 4  price       int64     dollar price × 10,000
                                 e.g. $118.60 → 1,186,000
    Col 5  direction   int8      +1=buy limit  −1=sell limit

Orderbook file — N × (4 × NumLevels) matrix, NO header.
Columns INTERLEAVED per level:
    AskPrice1, AskSize1, BidPrice1, BidSize1,
    AskPrice2, AskSize2, BidPrice2, BidSize2,  ...
    Prices: same integer × 10,000 encoding.
    Unoccupied ask levels: sentinel +9,999,999,999
    Unoccupied bid levels: sentinel −9,999,999,999

Row k in the message file caused the state change to orderbook row k.
Both files always have the same number of rows N.

─────────────────────────────────────────────────────────────────────────────
Output parquet schema  (contract with LobsterEnv — do not change names)
─────────────────────────────────────────────────────────────────────────────
File     : data/processed/<TICKER>_<YEAR>.parquet
Encoding : snappy-compressed, no index

Columns:
    date        str        'YYYY-MM-DD'
    time        datetime64 bar timestamp (1-min bars, last tick in bar)
    mid_price   float64    (ask1 + bid1) / 2                   [dollars]
    spread      float64    ask1 - bid1, clipped to ≥ 0          [dollars]
    imbalance   float64    (bidvol1 - askvol1)/(bidvol1+askvol1)  [-1,+1]
    ask1..ask5  float64    best 5 ask prices                    [dollars]
    bid1..bid5  float64    best 5 bid prices                    [dollars]
    askvol1..5  float64    best 5 ask volumes                   [shares]
    bidvol1..5  float64    best 5 bid volumes                   [shares]

Filter:  10:00–15:30 ET   (drops opening auction + MOC noise)
Resample : last tick per 1-minute bar; forward-fill gaps ≤ 1 bar
Size     : ~5 MB per stock-year vs ~15 GB raw CSV
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants  (single source of truth shared with tests and LobsterEnv)
# ─────────────────────────────────────────────────────────────────────────────

PRICE_SCALE     : int = 10_000          # raw integer → dollars
N_LEVELS        : int = 5               # LOB depth levels to keep
TRADING_START   : str = '10:00'         # core session start (ET)
TRADING_END     : str = '15:30'         # core session end (ET)
MIN_BARS_PER_DAY: int = 30              # minimum 1-min bars per episode

# Raw integer sentinel values for unoccupied LOB levels
_ASK_SENTINEL: int =  9_999_999_999
_BID_SENTINEL: int = -9_999_999_999

# Event types to DROP before feature computation
# 6 = cross trade (off-book price)   7 = trading halt (duplicated rows)
_DROP_EVENTS: frozenset = frozenset({6, 7})


# ─────────────────────────────────────────────────────────────────────────────
# Column name schema
#
# The names below ARE the interface contract between loader.py and lobster_env.py.
# LobsterEnv reads: mid_price, spread, imbalance, ask1..5, bid1..5, askvol1..5, bidvol1..5.
# ─────────────────────────────────────────────────────────────────────────────

# Raw orderbook CSV column order — interleaved per level (LOBSTER spec):
#   ask1, askvol1, bid1, bidvol1,  ask2, askvol2, bid2, bidvol2,  ...
_RAW_LOB_COLS: List[str] = []
for _l in range(1, N_LEVELS + 1):
    _RAW_LOB_COLS += [f'ask{_l}', f'askvol{_l}', f'bid{_l}', f'bidvol{_l}']

# Parquet output column schema (price cols grouped, then volume cols)
LOB_COLS    : List[str] = (
    [f'ask{l}'    for l in range(1, N_LEVELS + 1)] +
    [f'bid{l}'    for l in range(1, N_LEVELS + 1)] +
    [f'askvol{l}' for l in range(1, N_LEVELS + 1)] +
    [f'bidvol{l}' for l in range(1, N_LEVELS + 1)]
)
PARQUET_COLS: List[str] = (
    ['date', 'time', 'mid_price', 'spread', 'imbalance'] + LOB_COLS
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function processing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _read_message_file(path: Path) -> Optional[pd.DataFrame]:
    """
    Read a LOBSTER message CSV file.

    Official column order (no header):
        0  time        float64   seconds after midnight
        1  event_type  int8      event code 1-7
        2  order_id    int64     unique reference (not used downstream)
        3  size        int32     shares
        4  price       int64     dollar price × 10,000
        5  direction   int8      +1 / -1

    Returns None on any read failure (caller skips the day).
    """
    try:
        return pd.read_csv(
            path,
            header = None,
            names  = ['time', 'event_type', 'order_id', 'size', 'price', 'direction'],
            dtype  = {
                'time'       : np.float64,
                'event_type' : np.int8,
                'order_id'   : np.int64,
                'size'       : np.int32,
                'price'      : np.int64,
                'direction'  : np.int8,
            },
            engine = 'c',   # fastest CSV parser in pandas
        )
    except Exception as exc:
        log.error('Cannot read message file %s: %s', path.name, exc)
        return None


def _read_orderbook_file(path: Path) -> Optional[pd.DataFrame]:
    """
    Read a LOBSTER orderbook CSV file.

    Official column order (no header) — interleaved per level:
        AskPrice1, AskSize1, BidPrice1, BidSize1,
        AskPrice2, AskSize2, BidPrice2, BidSize2,  ...

    We read only the first N_LEVELS × 4 columns and assign _RAW_LOB_COLS
    so downstream code can reference ask1, askvol1, bid1, bidvol1, etc.

    Prices are int64 because sentinels (±9,999,999,999) exceed int32 range.
    Volumes are int32 (share counts never reach 2 billion).

    Returns None on any read failure.
    """
    n_cols = N_LEVELS * 4
    dtype  = {}
    for l in range(1, N_LEVELS + 1):
        dtype[f'ask{l}']    = np.int64   # may hold sentinel ±9_999_999_999
        dtype[f'askvol{l}'] = np.int32
        dtype[f'bid{l}']    = np.int64
        dtype[f'bidvol{l}'] = np.int32

    try:
        return pd.read_csv(
            path,
            header  = None,
            usecols = list(range(n_cols)),  # ignore levels beyond N_LEVELS
            names   = _RAW_LOB_COLS,
            dtype   = dtype,
            engine  = 'c',
        )
    except Exception as exc:
        log.error('Cannot read orderbook file %s: %s', path.name, exc)
        return None


def _process_day(
    msg_path : Path,
    lob_path : Path,
    date_str : str,
) -> Optional[pd.DataFrame]:
    """
    Full single-day processing pipeline.

    Steps
    -----
    1.  Read raw CSV files with exact LOBSTER dtypes.
    2.  Validate row counts match.
    3.  Drop event types 6 (cross trade) and 7 (halt).
    4.  Build absolute datetime index from 'seconds after midnight'.
    5.  Convert raw integer prices → dollars (÷ 10,000).
    6.  Filter rows where best bid/ask are sentinel dummy values.
    7.  Compute mid_price, spread, order-book imbalance.
    8.  Resample to 1-minute bars (LAST tick in each bar).
    9.  Forward-fill gaps of exactly 1 bar (handles quiet minutes).
    10. Filter to core trading hours [10:00, 15:30] ET.
    11. Validate minimum bar count.

    Returns
    -------
    DataFrame with columns == PARQUET_COLS, or None if day unusable.
    """

    # ── Step 1 ──────────────────────────────────────────────────────
    msg = _read_message_file(msg_path)
    lob = _read_orderbook_file(lob_path)
    if msg is None or lob is None:
        return None

    # ── Step 2 ──────────────────────────────────────────────────────
    if len(msg) != len(lob):
        log.warning('%s: row count mismatch msg=%d lob=%d — skipped',
                    date_str, len(msg), len(lob))
        return None

    # ── Step 3: drop halts + cross trades ────────────────────────────
    # Do this BEFORE building the datetime index so indices stay aligned.
    keep = ~msg['event_type'].isin(_DROP_EVENTS)
    msg  = msg.loc[keep].reset_index(drop=True)
    lob  = lob.loc[keep].reset_index(drop=True)
    if len(msg) == 0:
        return None

    # ── Step 4: build datetime index ─────────────────────────────────
    # LOBSTER 'time' = seconds after midnight ET.
    # Converting to absolute datetime lets pandas resample correctly.
    base     = pd.Timestamp(date_str)
    lob['dt'] = base + pd.to_timedelta(msg['time'].values, unit='s')

    # ── Step 5: convert prices integer → dollars ─────────────────────
    # Official encoding: stored as integer × 10,000.
    # $118.60 → 1,186,000 raw → 118.60 after ÷ 10,000.
    for l in range(1, N_LEVELS + 1):
        lob[f'ask{l}'] = lob[f'ask{l}'] / PRICE_SCALE
        lob[f'bid{l}'] = lob[f'bid{l}'] / PRICE_SCALE
    # Volumes stay as-is (already in shares).

    # ── Step 6: filter sentinel dummy prices ─────────────────────────
    # Unoccupied ask sentinel after conversion: +999,999.9999
    # Unoccupied bid sentinel after conversion: −999,999.9999
    # Real stock prices never exceed $10,000, so 100,000 is a safe threshold.
    ask_sentinel_d = _ASK_SENTINEL / PRICE_SCALE   #  999_999.9999
    bid_sentinel_d = _BID_SENTINEL / PRICE_SCALE   # -999_999.9999
    _THRESHOLD = 100_000.0                         # > any real stock price

    valid = (
        (lob['ask1'] < _THRESHOLD)  &
        (lob['bid1'] > -_THRESHOLD) &
        (lob['ask1'] > 0)           &
        (lob['bid1'] > 0)
    )
    lob = lob.loc[valid].reset_index(drop=True)
    if len(lob) == 0:
        return None

    # ── Step 7: derived features ──────────────────────────────────────
    # mid_price: arithmetic mean of best ask and best bid
    lob['mid_price'] = (lob['ask1'] + lob['bid1']) * 0.5

    # spread: best ask − best bid, clipped to ≥ 0
    # Crossed books (ask < bid) occasionally appear due to data issues.
    lob['spread'] = (lob['ask1'] - lob['bid1']).clip(lower=0.0)

    # order-book imbalance at level 1:
    #   imbalance = (bidvol1 − askvol1) / (bidvol1 + askvol1)
    #   +1 = all depth on bid side (bullish pressure)
    #   −1 = all depth on ask side (bearish / selling pressure)
    bv    = lob['bidvol1'].astype(np.float64)
    av    = lob['askvol1'].astype(np.float64)
    total = bv + av
    lob['imbalance'] = np.where(total > 0, (bv - av) / total, 0.0)

    # ── Step 8: resample to 1-minute bars ─────────────────────────────
    # Take the LAST tick in each 1-minute window.
    # This represents the LOB state the agent observes at decision time.
    feature_cols = ['mid_price', 'spread', 'imbalance'] + LOB_COLS
    bars = (
        lob.set_index('dt')[feature_cols]
           .resample('1min')
           .last()
    )

    # ── Step 9: forward-fill short gaps ──────────────────────────────
    # A quiet minute with zero trades leaves NaN after resample.
    # Fill at most 1 consecutive bar (1 minute) — protects against stale data.
    bars = bars.ffill(limit=1).dropna()

    # ── Step 10: trading hours filter ────────────────────────────────
    # 10:00–15:30 ET.  Reasons:
    #   09:30–10:00 : opening auction — abnormal spreads & volumes
    #   15:30–16:00 : MOC imbalance prints — distort IS estimates
    bars = bars.between_time(TRADING_START, TRADING_END)

    # ── Step 11: minimum bar count ────────────────────────────────────
    if len(bars) < MIN_BARS_PER_DAY:
        return None

    # ── Assemble output ───────────────────────────────────────────────
    # Replace sentinel values at deeper levels (lvl 2-5) with NaN.
    # Level 1 (BBO) is always real at this point.
    for l in range(2, N_LEVELS + 1):
        bars[f'ask{l}'] = bars[f'ask{l}'].where(
            bars[f'ask{l}'] < _THRESHOLD, np.nan)
        bars[f'bid{l}'] = bars[f'bid{l}'].where(
            bars[f'bid{l}'] > -_THRESHOLD, np.nan)

    bars        = bars.reset_index().rename(columns={'dt': 'time'})
    bars['date'] = date_str
    return bars.reindex(columns=PARQUET_COLS)


def _extract_date(msg_path: Path) -> str:
    """
    Extract 'YYYY-MM-DD' from a LOBSTER message filename.

    LOBSTER filename format:
        <TICKER>_<DATE>_<DATE>_<LEVELS>_message_<N>.csv
    The date is always the first 10-character underscore-separated part
    that parses as a valid date (parts[1] for standard files).

    Falls back to file modification time if parsing fails.
    """
    for part in msg_path.stem.split('_'):
        if len(part) == 10 and part[4] == '-' and part[7] == '-':
            try:
                pd.Timestamp(part)
                return part
            except Exception:
                continue
    # Fallback — should never be reached with real LOBSTER files
    import datetime
    return datetime.datetime.fromtimestamp(
        msg_path.stat().st_mtime).strftime('%Y-%m-%d')


def _find_orderbook(msg_path: Path) -> Optional[Path]:
    """
    Locate the matching LOBSTER orderbook file for a message file.

    Strategy 1 (primary): replace 'message' with 'orderbook' in the filename.
        AAPL_2019-01-02_..._message_1.csv  →  AAPL_2019-01-02_..._orderbook_1.csv

    Strategy 2 (fallback): glob for any file containing the date string
    and 'orderbook' in the same directory.

    Returns None if the message file does not exist or no orderbook is found.
    """
    if not msg_path.exists():
        return None

    candidate = Path(str(msg_path).replace('message', 'orderbook'))
    if candidate.exists():
        return candidate

    date_str   = _extract_date(msg_path)
    candidates = sorted(msg_path.parent.glob(f'*{date_str}*orderbook*.csv'))
    if candidates:
        return candidates[0]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# LOBSTERLoader  — orchestrates preprocess() over a stock-year directory
# ─────────────────────────────────────────────────────────────────────────────

class LOBSTERLoader:
    """
    Converts raw LOBSTER CSV files into compressed parquet files.

    Expected directory layout:

        data/raw/
            AAPL/
                AAPL_2019-01-02_2019-01-02_10_message_1.csv
                AAPL_2019-01-02_2019-01-02_10_orderbook_1.csv
                AAPL_2019-01-03_2019-01-03_10_message_1.csv
                ...
            MSFT/
                ...

    Output: data/processed/AAPL_2019.parquet  (~5 MB)
    """

    def __init__(
        self,
        raw_dir      : str = 'data/raw',
        processed_dir: str = 'data/processed',
    ):
        self.raw_dir       = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, stock: str, year: int, force: bool = False) -> Path:
        """
        Preprocess one stock-year of LOBSTER data into a parquet file.

        Idempotent: skips if output exists unless force=True.

        Args:
            stock : ticker, e.g. 'AAPL'
            year  : calendar year, e.g. 2019
            force : reprocess even if parquet already exists

        Returns:
            Path to the output parquet file.

        Raises:
            FileNotFoundError : raw directory or message files not found
            RuntimeError      : zero valid trading days after processing
        """
        out_path = self.processed_dir / f'{stock}_{year}.parquet'

        if out_path.exists() and not force:
            print(f'[Loader] {out_path.name} already exists — skipping '
                  f'(pass force=True to reprocess)')
            return out_path

        # ── locate raw files ─────────────────────────────────────────
        stock_dir = self.raw_dir / stock
        if not stock_dir.exists():
            raise FileNotFoundError(
                f'Raw data directory not found: {stock_dir}\n'
                f'Expected layout: {self.raw_dir}/<TICKER>/<files>.csv\n'
                f'Download from https://lobsterdata.com'
            )

        msg_files = sorted(stock_dir.glob(f'{stock}_{year}-*_message_*.csv'))
        if not msg_files:                             # fallback glob
            msg_files = sorted(stock_dir.glob(f'*{year}*message*.csv'))
        if not msg_files:
            raise FileNotFoundError(
                f'No message files for {stock} {year} in {stock_dir}\n'
                f'Expected: {stock}_{year}-MM-DD_{year}-MM-DD_10_message_1.csv'
            )

        print(f'[Loader] {stock} {year}: {len(msg_files)} days found, processing ...')

        # ── process each day ─────────────────────────────────────────
        frames : List[pd.DataFrame] = []
        n_skip  = 0

        for msg_path in msg_files:
            date_str = _extract_date(msg_path)
            lob_path = _find_orderbook(msg_path)

            if lob_path is None:
                log.warning('No orderbook file for %s — skipped', date_str)
                n_skip += 1
                continue

            try:
                df = _process_day(msg_path, lob_path, date_str)
            except Exception as exc:
                log.warning('Error on %s: %s', date_str, exc)
                n_skip += 1
                continue

            if df is None or len(df) < MIN_BARS_PER_DAY:
                n_skip += 1
                continue

            frames.append(df)

        if not frames:
            raise RuntimeError(
                f'No valid days for {stock} {year} after processing '
                f'({n_skip} skipped). Check raw files in {stock_dir}.'
            )

        # ── concatenate and save ──────────────────────────────────────
        combined = (pd.concat(frames, ignore_index=True)
                      .reindex(columns=PARQUET_COLS))
        combined.to_parquet(out_path, compression='snappy', index=False)

        mb = out_path.stat().st_size / 1e6
        print(f'[Loader] Saved {out_path.name}  '
              f'({len(frames)} days, {n_skip} skipped, {mb:.1f} MB)')
        return out_path

    def preprocess_all(
        self,
        stocks: List[str],
        years : List[int],
        force : bool = False,
    ) -> dict:
        """
        Preprocess every (stock, year) pair. Continues on individual failures.
        Returns dict: '{STOCK}_{YEAR}' → Path.
        """
        results = {}
        for stock in stocks:
            for year in years:
                key = f'{stock}_{year}'
                try:
                    results[key] = self.preprocess(stock, year, force=force)
                except Exception as exc:
                    print(f'[Loader] ERROR {key}: {exc}')
        return results

    def describe(self, stock: str, year: int) -> None:
        """Print summary statistics for a processed parquet file."""
        path = self.processed_dir / f'{stock}_{year}.parquet'
        if not path.exists():
            print(f'[Loader] {path.name} not found. Run preprocess() first.')
            return
        df     = pd.read_parquet(path)
        n_days = df['date'].nunique()
        n_bars = len(df)
        print(f'\n{"─"*56}')
        print(f'  {stock} {year}  |  {n_days} trading days  |  {n_bars:,} bars')
        print(f'{"─"*56}')
        print(f'  mid_price  : ${df.mid_price.mean():.2f} ± '
              f'${df.mid_price.std():.2f}'
              f'  [{df.mid_price.min():.2f}, {df.mid_price.max():.2f}]')
        print(f'  spread     : {df.spread.mean()*100:.3f}¢ avg')
        print(f'  imbalance  : {df.imbalance.mean():.4f} ± '
              f'{df.imbalance.std():.4f}')
        print(f'  bars/day   : {n_bars/n_days:.0f}')
        nans = df[['mid_price', 'spread', 'imbalance']].isna().sum()
        if nans.any():
            print(f'  WARNING NaN: {nans[nans > 0].to_dict()}')
        else:
            print(f'  No NaN in critical columns  ✓')
        print(f'{"─"*56}\n')


# ─────────────────────────────────────────────────────────────────────────────
# load_episodes  —  consumed directly by LobsterEnv
# ─────────────────────────────────────────────────────────────────────────────

def load_episodes(
    processed_dir : str,
    stocks        : List[str],
    years         : List[int],
    min_bars      : int  = MIN_BARS_PER_DAY,
    shuffle       : bool = False,
    seed          : int  = 42,
) -> List[pd.DataFrame]:
    """
    Load preprocessed parquet files → list of per-day DataFrames.

    This is the function LobsterEnv calls to populate its episode pool.
    Each returned DataFrame is one trading day with columns = PARQUET_COLS.

    Args:
        processed_dir : path containing {STOCK}_{YEAR}.parquet files
        stocks        : e.g. ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
        years         : e.g. [2019, 2020, 2021]
        min_bars      : days with fewer 1-min bars are excluded
        shuffle       : randomise episode order (True for training split)
        seed          : RNG seed for shuffle

    Returns:
        List[pd.DataFrame], ~len(stocks) × len(years) × 250 elements

    Raises:
        FileNotFoundError : any parquet file is missing
        RuntimeError      : no episodes survive the min_bars filter

    Memory budget (MacBook Air 8GB):
        5 stocks × 3 years × 250 days × 331 bars × 25 cols × 8 bytes ≈ 1.2 GB
        Single stock / 3 years ≈ 240 MB — well within 8 GB budget.
    """
    proc_dir = Path(processed_dir)
    missing  : List[str]          = []
    episodes : List[pd.DataFrame] = []

    for stock in stocks:
        for year in years:
            path = proc_dir / f'{stock}_{year}.parquet'
            if not path.exists():
                missing.append(str(path))
                continue

            df = pd.read_parquet(path)

            for date, day_df in df.groupby('date', sort=True):
                day_df = day_df.reset_index(drop=True)
                if len(day_df) >= min_bars:
                    episodes.append(day_df)

    if missing:
        raise FileNotFoundError(
            'Preprocessed files not found '
            '(run LOBSTERLoader.preprocess() first):\n'
            + '\n'.join(f'  {p}' for p in missing)
        )

    if not episodes:
        raise RuntimeError(
            f'No valid episodes for stocks={stocks} years={years}. '
            f'All days had fewer than {min_bars} bars.'
        )

    if shuffle:
        np.random.default_rng(seed).shuffle(episodes)

    print(f'[load_episodes] {len(episodes)} episodes  '
          f'({len(stocks)} stocks, {len(years)} years)')
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# verify_parquet  —  run after preprocess() to check data quality
# ─────────────────────────────────────────────────────────────────────────────

def verify_parquet(processed_dir: str, stock: str, year: int) -> bool:
    """
    Run a suite of sanity checks on a preprocessed parquet file.

    Checks:
        1.  File exists
        2.  All PARQUET_COLS present
        3.  No NaN in mid_price, spread, imbalance
        4.  mid_price ∈ (0, 10,000)
        5.  spread ≥ 0
        6.  imbalance ∈ [−1, +1]
        7.  ask1 ≥ bid1 (no inverted BBO)
        8.  ≥ 100 trading days
        9.  Average bars/day ≥ MIN_BARS_PER_DAY

    Prints ✓/✗ for each check.  Returns True iff all pass.
    """
    path = Path(processed_dir) / f'{stock}_{year}.parquet'
    ok   = True

    def _ok  (msg):       print(f'  ✓  {msg}')
    def _fail(msg): nonlocal ok; ok = False; print(f'  ✗  {msg}')

    print(f'\nVerifying {stock}_{year}.parquet ...')

    if not path.exists():
        _fail(f'File not found: {path}')
        return False

    df = pd.read_parquet(path)

    # 1 — schema
    missing = [c for c in PARQUET_COLS if c not in df.columns]
    if missing: _fail(f'Missing columns: {missing}')
    else:        _ok(f'All {len(PARQUET_COLS)} required columns present')

    # 2 — no NaN in critical columns
    nan_cnts = df[['mid_price', 'spread', 'imbalance']].isna().sum()
    if nan_cnts.any(): _fail(f'NaN in critical columns: {nan_cnts[nan_cnts>0].to_dict()}')
    else:               _ok('No NaN in mid_price / spread / imbalance')

    # 3 — mid_price range
    if not ((df['mid_price'] > 0) & (df['mid_price'] < 10_000)).all():
        _fail(f'mid_price outside (0, 10000): '
              f'[{df.mid_price.min():.2f}, {df.mid_price.max():.2f}]')
    else:
        _ok(f'mid_price ∈ (${df.mid_price.min():.2f}, ${df.mid_price.max():.2f})')

    # 4 — spread non-negative
    if not (df['spread'] >= 0).all():
        _fail('Negative spread values found')
    else:
        _ok(f'spread ≥ 0  (avg {df.spread.mean()*100:.3f}¢)')

    # 5 — imbalance bounds
    if not ((df['imbalance'] >= -1.0) & (df['imbalance'] <= 1.0)).all():
        _fail('imbalance outside [−1, +1]')
    else:
        _ok('imbalance ∈ [−1, +1]')

    # 6 — non-inverted BBO
    if not (df['ask1'] >= df['bid1']).all():
        _fail('Inverted BBO: ask1 < bid1 in some rows')
    else:
        _ok('ask1 ≥ bid1 (no inverted BBO)')

    # 7 — day count
    n_days = df['date'].nunique()
    if n_days < 100:
        _fail(f'Only {n_days} trading days (expected ≥ 100)')
    else:
        _ok(f'{n_days} trading days')

    # 8 — bars per day
    avg_bars = len(df) / max(n_days, 1)
    if avg_bars < MIN_BARS_PER_DAY:
        _fail(f'{avg_bars:.0f} avg bars/day (minimum {MIN_BARS_PER_DAY})')
    else:
        _ok(f'{avg_bars:.0f} avg bars/day')

    print(f'\n  {"PASS ✓" if ok else "FAIL ✗"}\n')
    return ok