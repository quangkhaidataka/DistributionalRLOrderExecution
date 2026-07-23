"""
scripts/eta_scale_check.py
--------------------------
Design-v2 B4 — η-scale gate for the TAQ study.

Computes DETERMINISTIC TWAP and MaxSpeed execution costs (in bps) on a sample of
real test days at the v2 resolution (3-min bars, N=20), to sanity-check the impact
coefficient η BEFORE any training. It answers: is fast execution meaningfully more
expensive than slow (TWAP) execution — but not insanely so?

  - impact too CHEAP  (MaxSpeed ≈ TWAP)  → every agent pins the cap (corner problem).
  - impact too COSTLY (MaxSpeed ≫ plausible) → agents just learn "never trade fast".

The costs are deterministic given the day's real prices and the schedule: TWAP
sells q0/N each bar; MaxSpeed sells at the cap (0.25·q0) until exhausted. We roll
BOTH from every valid start in each day and average — no RNG, no training.

Running this in B4 is allowed & required (deterministic, no training). It writes
results/_v2_taq/eta_scale_report.{json,txt}. The FORMAL go/no-go stays at B5;
this is the early-warning read.

Usage:
    python scripts/eta_scale_check.py                # default: η=1e-5, 15 test days
    python scripts/eta_scale_check.py --eta 2e-5 --n-days 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from envs.taq_env import TAQEnv, TAQConfig
from agents.baselines import TWAPAgent, MaxSpeedAgent

V2_GRID    = [round(0.025 * i, 3) for i in range(11)]
OUT_DIR    = PROJECT_ROOT / 'results' / '_v2_taq'


def _bar_source(ticker: str) -> str:
    return f'{ticker}_2014_3min_adj.parquet'


def _report_stem(ticker: str) -> str:
    # AAPL keeps the legacy unprefixed name (do not overwrite); others ticker-prefixed.
    return 'eta_scale_report' if ticker == 'AAPL' else f'{ticker}_eta_scale_report'

# Sanity band for the TWAP mean cost (bps): v1 TAQ TWAP mean ≈ 1.6 bps; a v2
# 3-min AAPL day should land in the same order of magnitude.
TWAP_SANE_LO, TWAP_SANE_HI = 0.1, 50.0
MAXSPEED_MIN_RATIO = 1.10      # MaxSpeed must cost ≥1.1× TWAP (else impact too cheap)


def build_cfg(eta: float, ticker: str = 'AAPL', q0: int = 5000) -> TAQConfig:
    return TAQConfig(
        N=20, T=60.0, q0=q0, p0=100.0, eta=eta, a=1e-4,
        bar_source=_bar_source(ticker), bar_minutes=3,
        action_basis='q0', action_fracs=V2_GRID,
        data_dir=str(PROJECT_ROOT / 'data' / 'processed'),
    )


def _roll_from(env, agent, start):
    """Deterministically roll `agent` from a fixed episode start.

    Returns (realized_IS_bps, impact_only_bps). The realized IS mixes the day's
    price drift with the impact cost; the impact-only component
    η·Σx_t²/(p0·q0) is the DRIFT-FREE, deterministic η-scale signal.
    """
    cfg = env.cfg
    env._ep_start = start
    env._p0 = float(env.data.iloc[start]['mid_price'])
    env._q  = float(cfg.q0)
    env._t  = 0
    env._rewards = []
    if hasattr(agent, 'reset'):
        agent.reset()
    state = env._get_state()
    done, info = False, {}
    sum_x2 = 0.0
    while not done:
        a = agent.select_action(state, eval_mode=True)
        state, r, done, info = env.step(a)
        sum_x2 += float(info['x_t']) ** 2
    is_bps     = float(info['implementation_shortfall']) * 1e4
    impact_bps = cfg.eta * sum_x2 / (env._p0 * float(cfg.q0)) * 1e4
    return is_bps, impact_bps


def day_costs(cfg: TAQConfig, date: str):
    """Mean TWAP / MaxSpeed realized-IS and impact-only (bps) over one day's starts."""
    env = TAQEnv(cfg, [date])
    twap, mspd = TWAPAgent(cfg), MaxSpeedAgent(cfg)
    tw, ms, tw_i, ms_i = [], [], [], []
    for start in env._valid_starts:
        a, ai = _roll_from(env, twap, start); tw.append(a); tw_i.append(ai)
        b, bi = _roll_from(env, mspd, start); ms.append(b); ms_i.append(bi)
    return (float(np.mean(tw)), float(np.mean(ms)),
            float(np.mean(tw_i)), float(np.mean(ms_i)), len(env._valid_starts))


def main():
    ap = argparse.ArgumentParser(description='TAQ η-scale gate (deterministic, no training)')
    ap.add_argument('--ticker', default='AAPL', help='ticker (default AAPL; unprefixed report)')
    ap.add_argument('--eta', type=float, default=1e-5)
    ap.add_argument('--q0', type=int, default=5000)
    ap.add_argument('--n-days', type=int, default=15)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / _bar_source(args.ticker))
    df['month'] = pd.to_datetime(df['date']).dt.month
    test_days = sorted(df[df['month'].isin([10, 11, 12])]['date'].unique().tolist())
    if len(test_days) < 10:
        raise SystemExit(f'Only {len(test_days)} Oct–Dec test days found (<10).')
    sample = test_days[:args.n_days]

    cfg = build_cfg(args.eta, args.ticker, args.q0)
    rows = []
    for d in sample:
        tw, ms, twi, msi, n = day_costs(cfg, d)
        rows.append({'date': d, 'twap_bps': tw, 'maxspeed_bps': ms,
                     'twap_impact_bps': twi, 'maxspeed_impact_bps': msi, 'n_starts': n})

    twap_mean = float(np.mean([r['twap_bps'] for r in rows]))
    mspd_mean = float(np.mean([r['maxspeed_bps'] for r in rows]))
    twap_imp  = float(np.mean([r['twap_impact_bps'] for r in rows]))
    mspd_imp  = float(np.mean([r['maxspeed_impact_bps'] for r in rows]))
    imp_ratio = mspd_imp / twap_imp if twap_imp else float('inf')

    # Primary gate = the DRIFT-FREE impact-only cost (the true η-scale signal):
    # MaxSpeed must cost meaningfully more impact than TWAP, at a sane magnitude
    # (not corner-cheap ≈0, not insanely large).
    speed_sep = imp_ratio >= MAXSPEED_MIN_RATIO
    imp_sane  = TWAP_SANE_LO <= mspd_imp <= TWAP_SANE_HI
    gate_ok   = bool(speed_sep and imp_sane)

    L = ['=' * 78,
         f'  TAQ η-scale gate — {args.ticker} — TWAP vs MaxSpeed (3-min, N=20), deterministic',
         '=' * 78,
         f'  η = {args.eta:g}   q0 = {cfg.q0}   a = {cfg.a:g}   '
         f'sample = {len(sample)} test days (Oct–Dec)',
         '  realized IS = impact + real price drift; impact-only = η·Σx²/(p0·q0) '
         '(drift-free).',
         '',
         f'  {"date":<12} {"TWAP IS":>9} {"MaxSpd IS":>10} | {"TWAP imp":>9} {"MaxSpd imp":>11} {"imp×":>6}']
    for r in rows:
        ir = r['maxspeed_impact_bps'] / r['twap_impact_bps'] if r['twap_impact_bps'] else float('nan')
        L.append(f'  {r["date"]:<12} {r["twap_bps"]:>9.3f} {r["maxspeed_bps"]:>10.3f} | '
                 f'{r["twap_impact_bps"]:>9.4f} {r["maxspeed_impact_bps"]:>11.4f} {ir:>6.2f}')
    L += ['  ' + '-' * 62,
          f'  {"MEAN":<12} {twap_mean:>9.3f} {mspd_mean:>10.3f} | '
          f'{twap_imp:>9.4f} {mspd_imp:>11.4f} {imp_ratio:>6.2f}',
          '',
          f'  realized IS (drift-contaminated): TWAP {twap_mean:.3f} bps, '
          f'MaxSpeed {mspd_mean:.3f} bps  (per-day IS swings ±10 bps on price drift)',
          f'  impact-only (the η signal)      : TWAP {twap_imp:.4f} bps, '
          f'MaxSpeed {mspd_imp:.4f} bps  (MaxSpeed = {imp_ratio:.1f}× — front-loads Σx²)',
          '',
          f'  MaxSpeed impact ≥ {MAXSPEED_MIN_RATIO}× TWAP impact (not corner-cheap): '
          f'{"YES" if speed_sep else "NO"}',
          f'  MaxSpeed impact in sane band [{TWAP_SANE_LO},{TWAP_SANE_HI}] bps       : '
          f'{"YES" if imp_sane else "NO"}  ({mspd_imp:.3f})',
          f'  GATE (early-warning read) : {"OK" if gate_ok else "SUSPICIOUS"}',
          '  (formal go/no-go is at B5; this is the B4 early warning.)',
          '=' * 78]
    txt = '\n'.join(L)
    print(txt)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = _report_stem(args.ticker)
    (OUT_DIR / f'{stem}.txt').write_text(txt)
    with open(OUT_DIR / f'{stem}.json', 'w') as f:
        json.dump({'ticker': args.ticker, 'eta': args.eta, 'q0': cfg.q0, 'a': cfg.a,
                   'n_days': len(sample), 'per_day': rows, 'twap_mean_bps': twap_mean,
                   'maxspeed_mean_bps': mspd_mean, 'twap_impact_bps': twap_imp,
                   'maxspeed_impact_bps': mspd_imp, 'impact_ratio': imp_ratio,
                   'speed_sep': speed_sep, 'impact_sane': imp_sane,
                   'gate_ok': gate_ok}, f, indent=2)
    print(f'\n  wrote {OUT_DIR/(stem+".json")} (gate_ok={gate_ok})')
    sys.exit(0)


if __name__ == '__main__':
    main()
