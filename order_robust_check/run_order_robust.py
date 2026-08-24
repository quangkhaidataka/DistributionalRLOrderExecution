"""
order_robust_check/run_order_robust.py
======================================
Order-size robustness appendix for the AAPL TAQ execution study: does IQN keep
its tail advantage as the parent order grows and market impact starts to bite?

FULLY SELF-CONTAINED. Every module it imports is a COPY living under
order_robust_check/, every path it writes is under order_robust_check/, and the
AAPL bar file is a copy in order_robust_check/data/processed/. Nothing outside
this folder is opened for writing, and the main study's results/ and
checkpoints/ are never touched.

Design
------
  * AAPL only, temporary impact FIXED at eta = 1e-5 for every level. Impact is
    NOT rescaled with q0 -- the cost per share is eta*x, so a bigger parent order
    mechanically pays more. That is the point of the sweep.
  * q0 in {5,000 / 50,000 / 250,000 / 1,000,000} shares.
  * Agents: TWAP, DQN, DDQN, IQN-neutral (the main study's line-up minus the
    rule-based extras and the CVaR ladder).
  * 3 seeds per level: {42, 123, 7} -- the first three of the main study's five.
  * 12,000 training episodes per agent (the main study uses 50,000). Reduced
    because this is a robustness appendix; stated in every output artefact.

Protocol is otherwise the main study's, so the numbers are comparable:
CRN checkpoint selection on the validation fold at min val-CVaR_0.95, then
10,000 held-out test episodes with the env and global RNG reseeded per agent.

Resumable
---------
Re-running is free: a trained agent is skipped if its final checkpoint exists,
and a (level, seed) cell is skipped entirely if its metrics.json exists.

Layout
------
  checkpoints/q0_<Q>/seed<S>/<AGENT>_ep<E>.pt
  results/q0_<Q>/seed<S>/metrics.json
  results/q0_<Q>/level_summary.{json,txt}
  results/table_order_robust.tex

Usage
-----
  python3 run_order_robust.py --q0 5000 --seed 42     # one cell
  python3 run_order_robust.py --level 5000            # all 3 seeds of a level
  python3 run_order_robust.py --summarize             # tables + SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

import numpy as np
import pandas as pd
import torch

from envs.taq_env import TAQEnv, TAQConfig
import run_v2_taq as TQ
import run_v2_ac as AC
import run_simulation as RS
import selection_lib

# --- study constants -------------------------------------------------------
Q0_LEVELS = [5_000, 50_000, 250_000, 1_000_000]
SEEDS = [42, 123, 7, 2024, 31]   # main study's set; the 5,000 baseline
                                 # keeps only the first three (see SUMMARY)
AGENTS = ['TWAP', 'DQN', 'DDQN', 'IQN-neutral']
TRAINABLE = ['DQN', 'DDQN', 'IQN-neutral']

EPISODES = 12_000          # main study: 50,000 (reduced for this appendix)
CKPT_FREQ = 1_000
N_VAL = 1_200              # same CRN validation budget as the main study
N_TEST = 10_000            # same held-out test budget as the main study
ETA = 1e-5                 # FIXED across every level -- never rescaled with q0

BARS = ROOT / 'data' / 'processed' / 'AAPL_2014_3min_adj.parquet'
CKPT_ROOT = ROOT / 'checkpoints'
RES_ROOT = ROOT / 'results'


def _guard_paths(*paths: Path) -> None:
    """Refuse to write anywhere outside order_robust_check/."""
    for p in paths:
        if ROOT not in Path(p).resolve().parents and Path(p).resolve() != ROOT:
            raise SystemExit(f'ISOLATION VIOLATION: refusing to write {p}')


def build_config(q0: int) -> TAQConfig:
    kw = dict(TQ.TAQ_BASE_CONFIG)
    kw['q0'] = int(q0)
    kw['eta'] = ETA                     # explicit: fixed, never scaled with q0
    return TAQConfig(data_dir=str(ROOT / 'data' / 'processed'), stock='AAPL',
                     year=2014, bar_source='AAPL_2014_3min_adj.parquet', **kw)


def adv_shares() -> float:
    d = pd.read_parquet(BARS, columns=['date', 'total_vol'])
    return float(d.groupby('date')['total_vol'].sum().mean())


# ---------------------------------------------------------------------------
# One (level, seed) cell
# ---------------------------------------------------------------------------

def run_cell(q0: int, seed: int, device: str = 'cpu') -> dict:
    ckpt_dir = CKPT_ROOT / f'q0_{q0}' / f'seed{seed}'
    out_dir = RES_ROOT / f'q0_{q0}' / f'seed{seed}'
    _guard_paths(ckpt_dir, out_dir)
    metrics_path = out_dir / 'metrics.json'
    if metrics_path.exists():
        print(f'  [skip] q0={q0:,} seed={seed} already done -> {metrics_path.name}')
        return json.loads(metrics_path.read_text())

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(q0)
    _all, train_dates, val_dates, test_dates = TQ.load_folds(cfg)
    agents, _ = TQ.build_taq_agents(cfg, train_dates, seed, device)

    # ---- train (resumable per agent) --------------------------------------
    t_train = time.time()
    for name in TRAINABLE:
        final = ckpt_dir / f'{name}_ep{EPISODES}.pt'
        if final.exists():
            print(f'  [skip] {name} already trained ({final.name})')
            continue
        t0 = time.time()
        AC.train_segment(agents[name], TAQEnv(cfg, train_dates), name,
                         0, EPISODES, EPISODES, seed, ckpt_dir, CKPT_FREQ,
                         ckpt_dir / f'{name}_resume.pt')
        print(f'  trained {name:<12s} {EPISODES:,} eps in {(time.time()-t0)/60:.1f} min')
    train_min = (time.time() - t_train) / 60

    # ---- CRN checkpoint selection on the val fold (min val-CVaR_0.95) ------
    val_seed = seed + AC.VAL_SEED_OFFSET
    picked = {}
    for name in TRAINABLE:
        paths = selection_lib.ckpt_paths_for(ckpt_dir, name)
        sel = selection_lib.select_checkpoints(
            agents[name], paths, TAQEnv(cfg, val_dates), val_seed,
            n_val=N_VAL, alpha=0.95)
        agents[name].load(sel['best_cvar'])
        picked[name] = {'checkpoint': Path(sel['best_cvar']).name,
                        'episode': sel['best_cvar_ep'], 'n_ckpts': len(paths)}
        print(f'  selected {name:<12s} {Path(sel["best_cvar"]).name} of {len(paths)}')

    # ---- held-out test evaluation -----------------------------------------
    subset = {n: agents[n] for n in AGENTS}
    test_seed = seed + AC.TEST_SEED_OFFSET
    all_results, _ = RS.evaluate_all(subset, TAQEnv(cfg, test_dates),
                                     n_eval=N_TEST, seed=test_seed)
    per_agent = {r['agent_name']: {k: float(r[k]) for k in
                                   ('mean_IS_bps', 'std_IS_bps', 'CVaR_0.90_bps',
                                    'CVaR_0.95_bps', 'max_IS_bps', 'cap_frac')}
                 for r in all_results}

    payload = {'q0': q0, 'seed': seed, 'eta': ETA, 'episodes': EPISODES,
               'n_val': N_VAL, 'n_test': N_TEST, 'val_seed': val_seed,
               'test_seed': test_seed, 'selected': picked,
               'train_minutes': train_min, 'metrics': per_agent}
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f'  -> {metrics_path}')
    return payload


# ---------------------------------------------------------------------------
# Level aggregation + reporting
# ---------------------------------------------------------------------------

def aggregate_level(q0: int) -> dict | None:
    cells = []
    for s in SEEDS:
        p = RES_ROOT / f'q0_{q0}' / f'seed{s}' / 'metrics.json'
        if p.exists():
            cells.append(json.loads(p.read_text()))
    if not cells:
        return None
    keys = ('mean_IS_bps', 'std_IS_bps', 'CVaR_0.90_bps', 'CVaR_0.95_bps')
    agg = {}
    for a in AGENTS:
        vals = {k: np.array([c['metrics'][a][k] for c in cells]) for k in keys}
        agg[a] = {k: {'mean': float(v.mean()), 'sd': float(v.std(ddof=0))}
                  for k, v in vals.items()}
    twap = agg['TWAP']['CVaR_0.95_bps']['mean']
    for a in AGENTS:
        m = agg[a]['CVaR_0.95_bps']['mean']
        agg[a]['cvar95_cut_vs_twap_pct'] = float(100.0 * (m - twap) / twap)
    return {'q0': q0, 'n_seeds': len(cells), 'seeds': [c['seed'] for c in cells],
            'episodes': EPISODES, 'eta': ETA, 'agents': agg}


def format_level(level: dict, adv: float) -> str:
    q0 = level['q0']
    L = [f'\n{"=" * 92}',
         f'q0 = {q0:,} shares  ({100 * q0 / adv:.3f}% of ADV)   '
         f'eta = {level["eta"]:g} (fixed)   {level["n_seeds"]} seeds '
         f'{level["seeds"]}   {level["episodes"]:,} train eps',
         '=' * 92,
         f'{"Agent":<14s}{"Mean IS":>17s}{"Std IS":>17s}'
         f'{"CVaR_0.90":>17s}{"CVaR_0.95":>17s}{"cut vs TWAP":>12s}',
         '-' * 92]
    for a in AGENTS:
        g = level['agents'][a]
        f = lambda k: f'{g[k]["mean"]:9.2f}+-{g[k]["sd"]:5.2f}'
        cut = g['cvar95_cut_vs_twap_pct']
        L.append(f'{a:<14s}{f("mean_IS_bps"):>17s}{f("std_IS_bps"):>17s}'
                 f'{f("CVaR_0.90_bps"):>17s}{f("CVaR_0.95_bps"):>17s}'
                 + (f'{"--":>12s}' if a == 'TWAP' else f'{cut:>11.1f}%'))
    L.append('all figures in basis points, mean +- sd over seeds; '
             'positive IS = cost')
    return '\n'.join(L)


LATEX_HEAD = r"""% Order-size robustness appendix -- AAPL TAQ execution study.
% Generated by order_robust_check/run_order_robust.py (self-contained rerun).
% eta = 1e-5 FIXED at every level; impact grows through eta*x as q0 grows.
\begin{table}[htbp]
    \centering
    \small
    \begin{tabular}{@{}lrrrrr@{}}
        \toprule
        \textbf{Agent} & \textbf{Mean IS} & \textbf{Std IS} &
        \textbf{CVaR$_{0.90}$} & \textbf{CVaR$_{0.95}$} &
        \textbf{CVaR$_{0.95}$ cut} \\
         & (bps) & (bps) & (bps) & (bps) & \textbf{vs TWAP} \\
        \midrule
"""

LATEX_TAIL = r"""        \bottomrule
    \end{tabular}
    \caption[Order-size robustness of the tail advantage]{%%
        Order-size robustness of the AAPL execution study. The temporary-impact
        coefficient is held \emph{fixed} at $\eta = 10^{-5}$ across all four
        levels, so impact cost grows with the parent order through $\eta x$.
        Each block reports the mean $\pm$ standard deviation over the seeds
        listed in its sub-heading (%s), evaluated on %s held-out test episodes
        from the Oct--Dec fold;
        agents were trained for %s episodes per seed (the main study uses
        50{,}000). Order sizes are quoted as a percentage of AAPL's 2014 average
        daily volume (%s shares).}
    \label{tab:order_robust}
\end{table}
"""


def latex_table(levels: list[dict], adv: float) -> str:
    rows = []
    for i, lv in enumerate(levels):
        q0 = lv['q0']
        if i:
            rows.append('        \\midrule\n')
        q0_tex = f'{q0:,}'.replace(',', '{,}')     # thin-space thousands only
        rows.append(f'        \\multicolumn{{6}}{{@{{}}l}}{{\\textit{{'
                    f'$q_0$ = {q0_tex} shares ({100 * q0 / adv:.3f}\\% ADV, '
                    f'{lv["n_seeds"]} seeds)'
                    f'}}}} \\\\[2pt]\n')
        for a in AGENTS:
            g = lv['agents'][a]
            f = lambda k: f'${g[k]["mean"]:.2f} \\pm {g[k]["sd"]:.2f}$'
            cut = ('--' if a == 'TWAP'
                   else f'${g["cvar95_cut_vs_twap_pct"]:.1f}\\%$')
            nm = a.replace('IQN-neutral', 'IQN-neutral')
            rows.append(f'        {nm} & {f("mean_IS_bps")} & {f("std_IS_bps")} & '
                        f'{f("CVaR_0.90_bps")} & {f("CVaR_0.95_bps")} & {cut} \\\\\n')
    allseeds = ', '.join(str(x) for x in sorted({s for lv in levels
                                                 for s in lv['seeds']}))
    return (LATEX_HEAD + ''.join(rows) +
            LATEX_TAIL % (f'drawn from {allseeds}',
                          f'{N_TEST:,}'.replace(',', '{,}'),
                          f'{EPISODES:,}'.replace(',', '{,}'),
                          f'{adv:,.0f}'.replace(',', '{,}')))


def cmd_summarize() -> None:
    adv = adv_shares()
    levels = [lv for lv in (aggregate_level(q) for q in Q0_LEVELS) if lv]
    if not levels:
        raise SystemExit('nothing to summarise yet')
    for lv in levels:
        print(format_level(lv, adv))
        p = RES_ROOT / f'q0_{lv["q0"]}' / 'level_summary.json'
        _guard_paths(p)
        p.write_text(json.dumps(lv, indent=2))
        (p.parent / 'level_summary.txt').write_text(format_level(lv, adv))
    tex = RES_ROOT / 'table_order_robust.tex'
    _guard_paths(tex)
    tex.write_text(latex_table(levels, adv))
    print(f'\nLaTeX table -> {tex}')
    (RES_ROOT / 'adv.json').write_text(json.dumps(
        {'adv_shares_2014': adv,
         'pct_adv': {str(q): 100 * q / adv for q in Q0_LEVELS}}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q0', type=int)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--level', type=int, help='run all seeds of one q0 level')
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
    a = ap.parse_args()

    if a.summarize:
        cmd_summarize()
    elif a.level:
        for s in SEEDS:
            print(f'\n### q0={a.level:,} seed={s}')
            run_cell(a.level, s, a.device)
    elif a.q0 and a.seed:
        run_cell(a.q0, a.seed, a.device)
    else:
        raise SystemExit('need --q0/--seed, --level, or --summarize')


if __name__ == '__main__':
    main()
