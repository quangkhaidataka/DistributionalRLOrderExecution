"""
data/build_taq_3min.py
----------------------
Design-v2 B4 — rebuild 3-minute AAPL bars from the 1-minute TAQ parquet and
split-adjust the 7:1 stock split (2014-06-09), for the N=20 / dt=3′ v2 study.

What it does:
  1. Loads data/processed/AAPL_2014.parquet (1-min bars). The real trading date
     is in the `date` column (str); the `time` column carries the time-of-day
     (its date part is a placeholder).
  2. Resamples to 3-minute bars WITHIN each day (never straddling days):
       mid_price/best_bid/best_ask = last (bar close), spread/imbalance = mean,
       volumes/trade_count = sum; ret & volatility recomputed on the 3-min series.
  3. Split-adjusts: divides all PRICE columns (mid_price, best_bid, best_ask,
     spread) by 7 for dates BEFORE 2014-06-09, so the series is continuous across
     the split (the raw data already has the split IN it: 06-06 ≈ $645, 06-09 ≈ $92).
  4. Writes data/processed/AAPL_2014_3min_adj.parquet — NEVER overwriting the
     1-minute source.
  5. Emits a validation report (split-date continuity, bars/day, NaN scan,
     monotone timestamps, post-adjust summary stats) → feeds thesis Table 3.1.

Deterministic data prep (no training) — safe to run in B4.

Usage:
    python data/build_taq_3min.py            # build + print the report
    python data/build_taq_3min.py --report-only   # rebuild report from existing parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PARQUET  = PROJECT_ROOT / 'data' / 'processed' / 'AAPL_2014.parquet'
OUT_PARQUET  = PROJECT_ROOT / 'data' / 'processed' / 'AAPL_2014_3min_adj.parquet'

SPLIT_DATE   = '2014-06-09'   # AAPL 7:1 split effective date
SPLIT_RATIO  = 7.0
BAR          = '3min'
PRICE_COLS   = ['mid_price', 'best_bid', 'best_ask', 'spread']   # scaled by 1/7 pre-split
RV_WINDOW    = 5             # bars, for the recomputed 3-min realized-vol column


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _time_of_day(s: pd.Series) -> pd.Series:
    """Extract the HH:MM:SS time-of-day (the `time` column's date part is bogus)."""
    t = pd.to_datetime(s)
    return t - t.dt.normalize()


def build() -> pd.DataFrame:
    if not SRC_PARQUET.exists():
        raise FileNotFoundError(f'{SRC_PARQUET} not found (1-min TAQ source).')
    raw = pd.read_parquet(SRC_PARQUET)
    raw['date'] = raw['date'].astype(str)

    # Real per-bar timestamp = real date + intraday time-of-day.
    tod = _time_of_day(raw['time'])
    raw['ts'] = pd.to_datetime(raw['date']) + tod

    # Numeric coercion (source uses pandas nullable dtypes).
    num_cols = ['mid_price', 'spread', 'best_bid', 'best_ask', 'buy_vol',
                'sell_vol', 'total_vol', 'trade_count', 'imbalance',
                'best_bidsizeshares', 'best_asksizeshares']
    for c in num_cols:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce').astype(float)

    agg = {
        'mid_price': 'last', 'best_bid': 'last', 'best_ask': 'last',
        'spread': 'mean', 'imbalance': 'mean',
        'buy_vol': 'sum', 'sell_vol': 'sum', 'total_vol': 'sum',
        'trade_count': 'sum',
        'best_bidsizeshares': 'mean', 'best_asksizeshares': 'mean',
    }
    agg = {k: v for k, v in agg.items() if k in raw.columns}

    # Resample to 3-min bars PER DAY (so a bar never straddles two sessions).
    bars = []
    for date, g in raw.sort_values('ts').groupby('date'):
        r = (g.set_index('ts')
               .resample(BAR, label='left', closed='left', origin='start_day')
               .agg(agg))
        r = r.dropna(subset=['mid_price'])          # drop empty 3-min bins
        if r.empty:
            continue
        r['date']  = date
        r['stock'] = 'AAPL'
        r = r.reset_index().rename(columns={'ts': 'time'})
        bars.append(r)

    df = pd.concat(bars, ignore_index=True).sort_values(['date', 'time']).reset_index(drop=True)

    # 7:1 split adjustment: divide pre-split PRICES by 7 (volumes/imbalance intact).
    pre = df['date'] < SPLIT_DATE
    for c in PRICE_COLS:
        if c in df.columns:
            df.loc[pre, c] = df.loc[pre, c] / SPLIT_RATIO

    # Recompute ret & realized-vol on the adjusted 3-min series, per day.
    df['ret'] = df.groupby('date')['mid_price'].pct_change()
    df['volatility'] = (df.groupby('date')['ret']
                          .transform(lambda s: s.rolling(RV_WINDOW, min_periods=1).std())
                          .fillna(0.0))
    df['ret'] = df['ret'].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def report(df: pd.DataFrame) -> str:
    L = []
    L.append('=' * 74)
    L.append('  TAQ 3-min build — validation report (AAPL 2014, split-adjusted)')
    L.append('=' * 74)

    days = df['date'].nunique()
    L.append(f'  bars total       : {len(df):,}   trading days: {days}')

    # 1. Split-date continuity: adjusted 06-06 close vs 06-09 open.
    d_before = df[df['date'] == '2014-06-06']
    d_after  = df[df['date'] == SPLIT_DATE]
    if len(d_before) and len(d_after):
        close_before = float(d_before['mid_price'].iloc[-1])
        open_after   = float(d_after['mid_price'].iloc[0])
        gap = (open_after - close_before) / close_before * 100.0
        L.append(f'  split continuity : 06-06 adj close ${close_before:.2f} -> '
                 f'06-09 open ${open_after:.2f}  (overnight gap {gap:+.2f}%)')
    else:
        L.append('  split continuity : [06-06 or 06-09 missing]')

    # 2. Bars/day.
    bpd = df.groupby('date').size()
    L.append(f'  bars/day         : median {int(bpd.median())}  '
             f'[min {bpd.min()}, max {bpd.max()}]  (full 6.5h day -> ~130)')

    # 3. NaN scan.
    key = [c for c in ['mid_price', 'spread', 'imbalance', 'best_bid', 'best_ask',
                       'ret', 'volatility'] if c in df.columns]
    nans = {c: int(df[c].isna().sum()) for c in key}
    L.append(f'  NaN scan         : {nans}  '
             f'({"clean" if sum(nans.values()) == 0 else "HAS NaN"})')

    # 4. Monotone timestamps within each day.
    def _mono(g):
        return g['time'].is_monotonic_increasing
    non_mono = [d for d, g in df.groupby('date') if not _mono(g)]
    L.append(f'  monotone ts/day  : {"all monotone" if not non_mono else f"NON-MONOTONE: {non_mono[:5]}"}')

    # 5. Never-straddle sanity: every 3-min bar's time-of-day within [09:30,16:03].
    tod = (df['time'] - df['time'].dt.normalize())
    lo, hi = tod.min(), tod.max()
    L.append(f'  intraday range   : {str(lo)} .. {str(hi)}  (within a single session)')

    # 6. Post-adjust price summary (pre-split vs post-split, both adjusted).
    pre  = df[df['date'] < SPLIT_DATE]['mid_price']
    post = df[df['date'] >= SPLIT_DATE]['mid_price']
    L.append(f'  adj mid_price    : pre-split  mean ${pre.mean():.2f}  [{pre.min():.2f}, {pre.max():.2f}]')
    L.append(f'                     post-split mean ${post.mean():.2f}  [{post.min():.2f}, {post.max():.2f}]')
    L.append(f'  adj spread(bps)  : mean {(df["spread"]/df["mid_price"]*1e4).mean():.2f}  '
             f'median {(df["spread"]/df["mid_price"]*1e4).median():.2f}')
    L.append('=' * 74)
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description='Build 3-min split-adjusted AAPL bars')
    ap.add_argument('--report-only', action='store_true',
                    help='Rebuild the report from the existing output parquet')
    args = ap.parse_args()

    if args.report_only:
        if not OUT_PARQUET.exists():
            raise SystemExit(f'{OUT_PARQUET} does not exist; run without --report-only first.')
        df = pd.read_parquet(OUT_PARQUET)
    else:
        df = build()
        if OUT_PARQUET.exists():
            print(f'  NOTE: {OUT_PARQUET.name} exists — overwriting the DERIVED 3-min parquet '
                  f'(source {SRC_PARQUET.name} is never touched).')
        df.to_parquet(OUT_PARQUET, index=False)
        print(f'  wrote {OUT_PARQUET}  ({len(df):,} bars)')

    print(report(df))


if __name__ == '__main__':
    main()
