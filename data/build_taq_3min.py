"""
data/build_taq_3min.py
----------------------
Design-v2 B4 — rebuild 3-minute bars from the 1-minute TAQ parquet and
split-adjust, for the N=20 / dt=3′ v2 study. Generalised across tickers (B4-XS):
``--ticker T`` sources ``data/processed/{T}_2014.parquet`` → writes
``data/processed/{T}_2014_3min_adj.parquet``. Default ticker AAPL is byte-identical
to the original single-ticker build.

Split handling is DATA-DRIVEN (not hard-coded 7:1):
  * Scan every trading day's open-vs-prior-close gap; a |gap| > 20% flags a split.
  * For a ticker in ``KNOWN_SPLITS`` (AAPL 2014-06-09 7:1) the documented exact ratio
    is used (reproducibility — the detected ~6.96 only confirms the date); the
    detected date must match.
  * For any other ticker the MEASURED ratio (prev_close/open) is used and
    sanity-checked to sit near a clean split (nearest small integer within 5%).
  * No >20% gap ⇒ no adjustment ("no split").
  * An ambiguous gap (not near a clean ratio) raises — caller decides, never guesses.

What it does:
  1. Loads the 1-min source; real date is the `date` column, `time` carries the
     intraday time-of-day (its date part is a placeholder).
  2. Resamples to 3-min bars WITHIN each day (never straddling sessions).
  3. Split-adjusts pre-event PRICE columns by the ratio so the series is continuous.
  4. Writes {T}_2014_3min_adj.parquet — NEVER overwriting the 1-min source.
  5. Emits the T-TAQ-0 validation report (split continuity, bars/day, NaN scan,
     monotone timestamps, intraday range, pre/post price summary).

Deterministic data prep (no training).

Usage:
    python data/build_taq_3min.py                       # AAPL (byte-identical default)
    python data/build_taq_3min.py --ticker GOOG         # build + report
    python data/build_taq_3min.py --ticker MSFT --report-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / 'data' / 'processed'

# Documented clean splits (exact ratio for reproducibility). Detection still runs
# and must agree on the DATE; the exact ratio keeps the committed parquet stable.
KNOWN_SPLITS = {'AAPL': ('2014-06-09', 7.0)}

BAR          = '3min'
PRICE_COLS   = ['mid_price', 'best_bid', 'best_ask', 'spread']   # scaled by 1/ratio pre-split
RV_WINDOW    = 5             # bars, for the recomputed 3-min realized-vol column
GAP_THRESH   = 0.20         # |open/prev_close − 1| above this ⇒ candidate split


def _src_parquet(ticker: str) -> Path:
    return PROCESSED / f'{ticker}_2014.parquet'


def _out_parquet(ticker: str) -> Path:
    return PROCESSED / f'{ticker}_2014_3min_adj.parquet'


# ---------------------------------------------------------------------------
# Split detection (data-driven)
# ---------------------------------------------------------------------------

def detect_split(daily: pd.DataFrame):
    """Return (date_str, measured_ratio) for the largest >20% daily gap, else (None, None).

    ``daily`` has index = date str, columns 'first' (day open) and 'last' (day close).
    """
    prev_close = daily['last'].shift(1)
    gap = daily['first'] / prev_close - 1.0
    flagged = gap[gap.abs() > GAP_THRESH].dropna()
    if flagged.empty:
        return None, None
    # largest-magnitude gap (a real split dwarfs any normal move)
    date = flagged.abs().idxmax()
    ratio = float(prev_close.loc[date] / daily['first'].loc[date])
    return str(date), ratio


def _resolve_split(ticker: str, daily: pd.DataFrame):
    """Decide (split_date, adjust_ratio, note) for a ticker from detection + known table."""
    det_date, det_ratio = detect_split(daily)
    if ticker in KNOWN_SPLITS:
        known_date, known_ratio = KNOWN_SPLITS[ticker]
        if det_date != known_date:
            raise SystemExit(
                f'{ticker}: known split {known_date} not detected (detected {det_date}). '
                f'Refusing to guess.')
        note = (f'known {known_ratio:g}:1 split {known_date} '
                f'(detected ratio {det_ratio:.3f} confirms); using exact {known_ratio:g}')
        return known_date, known_ratio, note
    if det_date is None:
        return None, None, 'no split detected (no >20% daily gap)'
    # data-driven: sanity-check the measured ratio sits near a clean split
    nearest = round(det_ratio)
    if nearest < 2 or abs(det_ratio - nearest) / nearest > 0.05:
        raise SystemExit(
            f'{ticker}: detected {det_date} gap ratio {det_ratio:.3f} is NOT near a clean '
            f'split (nearest {nearest}). Ambiguous — STOP and inspect, do not guess.')
    note = (f'data-driven split {det_date}, MEASURED ratio {det_ratio:.3f} '
            f'(≈{nearest}:1, within 5%); using measured {det_ratio:.3f}')
    return det_date, det_ratio, note


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _time_of_day(s: pd.Series) -> pd.Series:
    """Extract the HH:MM:SS time-of-day (the `time` column's date part is bogus)."""
    t = pd.to_datetime(s)
    return t - t.dt.normalize()


def build(ticker: str = 'AAPL'):
    """Build the 3-min split-adjusted bars for ``ticker``. Returns (df, meta)."""
    src = _src_parquet(ticker)
    if not src.exists():
        raise SystemExit(f'{src} not found (1-min TAQ source for {ticker}).')
    raw = pd.read_parquet(src)
    raw['date'] = raw['date'].astype(str)

    tod = _time_of_day(raw['time'])
    raw['ts'] = pd.to_datetime(raw['date']) + tod

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

    bars = []
    for date, g in raw.sort_values('ts').groupby('date'):
        r = (g.set_index('ts')
               .resample(BAR, label='left', closed='left', origin='start_day')
               .agg(agg))
        r = r.dropna(subset=['mid_price'])
        if r.empty:
            continue
        r['date']  = date
        r['stock'] = ticker
        r = r.reset_index().rename(columns={'ts': 'time'})
        bars.append(r)

    df = pd.concat(bars, ignore_index=True).sort_values(['date', 'time']).reset_index(drop=True)

    # Resolve + apply the split adjustment (data-driven per ticker).
    daily = df.groupby('date')['mid_price'].agg(['first', 'last'])
    split_date, ratio, note = _resolve_split(ticker, daily)
    if split_date is not None:
        pre = df['date'] < split_date
        for c in PRICE_COLS:
            if c in df.columns:
                df.loc[pre, c] = df.loc[pre, c] / ratio

    # Recompute ret & realized-vol on the adjusted 3-min series, per day.
    df['ret'] = df.groupby('date')['mid_price'].pct_change()
    df['volatility'] = (df.groupby('date')['ret']
                          .transform(lambda s: s.rolling(RV_WINDOW, min_periods=1).std())
                          .fillna(0.0))
    df['ret'] = df['ret'].fillna(0.0)

    meta = {'ticker': ticker, 'split_date': split_date, 'ratio': ratio, 'note': note}
    return df, meta


# ---------------------------------------------------------------------------
# Validation report (T-TAQ-0)
# ---------------------------------------------------------------------------

def report(df: pd.DataFrame, meta: dict) -> str:
    ticker = meta['ticker']
    split_date = meta['split_date']
    L = []
    L.append('=' * 74)
    L.append(f'  TAQ 3-min build — T-TAQ-0 validation report ({ticker} 2014)')
    L.append('=' * 74)

    days = df['date'].nunique()
    L.append(f'  bars total       : {len(df):,}   trading days: {days}')
    L.append(f'  split handling   : {meta["note"]}')

    # 1. Continuity across the split date (adjusted): last close before vs first open on split.
    if split_date is not None:
        all_dates = sorted(df['date'].unique())
        if split_date in all_dates:
            idx = all_dates.index(split_date)
            prev_date = all_dates[idx - 1] if idx > 0 else None
            d_after = df[df['date'] == split_date]
            if prev_date is not None:
                d_before = df[df['date'] == prev_date]
                close_before = float(d_before['mid_price'].iloc[-1])
                open_after   = float(d_after['mid_price'].iloc[0])
                gap = (open_after - close_before) / close_before * 100.0
                L.append(f'  split continuity : {prev_date} adj close ${close_before:.2f} -> '
                         f'{split_date} open ${open_after:.2f}  (overnight gap {gap:+.2f}%)')
            else:
                L.append('  split continuity : [split date is first day — no prior close]')
        else:
            L.append(f'  split continuity : [{split_date} not in bars]')
    else:
        L.append('  split continuity : n/a (no split)')

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
    non_mono = [d for d, g in df.groupby('date') if not g['time'].is_monotonic_increasing]
    L.append(f'  monotone ts/day  : {"all monotone" if not non_mono else f"NON-MONOTONE: {non_mono[:5]}"}')

    # 5. Never-straddle sanity: intraday time-of-day range.
    tod = (df['time'] - df['time'].dt.normalize())
    L.append(f'  intraday range   : {str(tod.min())} .. {str(tod.max())}  (within a single session)')

    # 6. Pre/post price summary (both adjusted; if no split, whole-series summary).
    if split_date is not None:
        pre  = df[df['date'] < split_date]['mid_price']
        post = df[df['date'] >= split_date]['mid_price']
        L.append(f'  adj mid_price    : pre-split  mean ${pre.mean():.2f}  [{pre.min():.2f}, {pre.max():.2f}]')
        L.append(f'                     post-split mean ${post.mean():.2f}  [{post.min():.2f}, {post.max():.2f}]')
    else:
        mp = df['mid_price']
        L.append(f'  mid_price        : mean ${mp.mean():.2f}  [{mp.min():.2f}, {mp.max():.2f}]  (no split)')
    L.append(f'  adj spread(bps)  : mean {(df["spread"]/df["mid_price"]*1e4).mean():.2f}  '
             f'median {(df["spread"]/df["mid_price"]*1e4).median():.2f}')
    L.append('=' * 74)
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description='Build 3-min split-adjusted bars (per ticker)')
    ap.add_argument('--ticker', default='AAPL', help='ticker (default AAPL, byte-identical)')
    ap.add_argument('--report-only', action='store_true',
                    help='Rebuild the report from the existing output parquet')
    ap.add_argument('--out-dir', default=None,
                    help='override output dir (for byte-identity verification; default data/processed)')
    args = ap.parse_args()

    ticker = args.ticker
    out = _out_parquet(ticker)
    if args.out_dir:
        out = Path(args.out_dir) / out.name

    if args.report_only:
        if not out.exists():
            raise SystemExit(f'{out} does not exist; run without --report-only first.')
        df = pd.read_parquet(out)
        # reconstruct minimal meta for the report from the data
        daily = df.groupby('date')['mid_price'].agg(['first', 'last'])
        sd, ratio, note = _resolve_split(ticker, daily) if ticker in KNOWN_SPLITS else (None, None, 'report-only')
        meta = {'ticker': ticker, 'split_date': sd, 'ratio': ratio, 'note': note}
    else:
        df, meta = build(ticker)
        if out.exists():
            print(f'  NOTE: {out.name} exists — overwriting the DERIVED 3-min parquet '
                  f'(source {_src_parquet(ticker).name} is never touched).')
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f'  wrote {out}  ({len(df):,} bars)')

    print(report(df, meta))


if __name__ == '__main__':
    main()
