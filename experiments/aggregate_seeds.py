"""
experiments/aggregate_seeds.py
------------------------------
Aggregate per-seed evaluation results (R1) into mean ± std tables, in both
plain text and LaTeX booktabs format.

Consumes the per-seed ``logs/all_results.json`` files written by
``run_seeds.py`` (and by ``run_simulation.run_phase``). Each is a list of
per-agent metric dicts using the ``*_bps`` schema produced by
``EpisodeTracker.compute_metrics`` (mean_IS_bps, std_IS_bps, CVaR_0.90_bps,
CVaR_0.95_bps, max_IS_bps, GL_ratio, agent_name).

Usage:
    python experiments/aggregate_seeds.py --env ac      # results/_seeds/seed*/almgren_chriss
    python experiments/aggregate_seeds.py --env jump    # results/_seeds/seed*/jump_diffusion
    python experiments/aggregate_seeds.py --env taq     # results/taq/AAPL_seed*
    python experiments/aggregate_seeds.py --dirs pathA pathB ...

Writes results/_aggregate/<env>_seed_summary.txt and .tex
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (json key, display label) — the metrics aggregated across seeds.
METRICS = [
    ('mean_IS_bps',  'Mean IS'),
    ('std_IS_bps',   'Std IS'),
    ('CVaR_0.90_bps', 'CVaR90'),
    ('CVaR_0.95_bps', 'CVaR95'),
    ('max_IS_bps',   'Max IS'),
    ('GL_ratio',     'GL'),
]

# Glob patterns for the built-in --env shortcuts (relative to PROJECT_ROOT).
ENV_GLOBS = {
    'ac':   'results/_seeds/seed*/almgren_chriss',
    'ou':   'results/_seeds/seed*/mean_reverting',
    'jump': 'results/_seeds/seed*/jump_diffusion',
    'taq':  'results/taq/AAPL_seed*',
}


# ---------------------------------------------------------------------------
# Pure aggregation (unit-tested directly, no file I/O)
# ---------------------------------------------------------------------------

def aggregate_metrics(per_seed, agents_order=None, metrics=METRICS):
    """Aggregate a list of per-seed results into mean/std per agent×metric.

    Args:
        per_seed: list (one entry per seed) of {agent_name: metric_dict}.
        agents_order: optional explicit agent ordering; defaults to the order
                      seen in the first seed.
        metrics: list of (json_key, label) pairs to aggregate.

    Returns:
        (agents, rows) where rows[agent][json_key] = (mean, std) across seeds
        (seeds where the agent or key is missing are skipped).
    """
    if not per_seed:
        raise ValueError('aggregate_metrics: per_seed is empty')
    agents = agents_order or list(per_seed[0].keys())
    rows = {}
    for a in agents:
        rows[a] = {}
        for key, _label in metrics:
            vals = [ps[a][key] for ps in per_seed
                    if a in ps and key in ps[a] and ps[a][key] is not None]
            if vals:
                rows[a][key] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                rows[a][key] = (float('nan'), float('nan'))
    return agents, rows


def to_plain_text(agents, rows, n_seeds, metrics=METRICS, title=''):
    lines = []
    if title:
        lines.append(title)
    lines.append(f'Aggregated over {n_seeds} seed(s). All values in bps '
                 f'(mean ± std across seeds).')
    header = f'{"Agent":<16s}' + ''.join(f' | {lab:>15s}' for _k, lab in metrics)
    lines.append(header)
    lines.append('-' * len(header))
    for a in agents:
        row = f'{a:<16s}'
        for key, _lab in metrics:
            m, s = rows[a][key]
            row += f' | {m:>7.4f}±{s:<7.4f}'
        lines.append(row)
    return '\n'.join(lines) + '\n'


def to_latex(agents, rows, n_seeds, metrics=METRICS, caption='', label=''):
    """Render a booktabs LaTeX table of mean ± std across seeds."""
    col_spec = 'l' + 'r' * len(metrics)
    out = []
    out.append(r'\begin{table}[H]')
    out.append(r'    \centering')
    if caption:
        out.append(f'    \\caption{{{caption}}}')
    if label:
        out.append(f'    \\label{{{label}}}')
    out.append(f'    \\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}')
    out.append(r'        \toprule')
    head = 'Agent & ' + ' & '.join(lab for _k, lab in metrics) + r' \\'
    out.append('        ' + head)
    out.append(r'        \midrule')
    for a in agents:
        cells = []
        for key, _lab in metrics:
            m, s = rows[a][key]
            cells.append(f'{m:.4f} $\\pm$ {s:.4f}')
        safe = a.replace('_', r'\_')
        out.append('        ' + safe + ' & ' + ' & '.join(cells) + r' \\')
    out.append(r'        \bottomrule')
    out.append(r'    \end{tabular}')
    out.append(r'\end{table}')
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------------------
# File loading + CLI
# ---------------------------------------------------------------------------

def load_seed_results(seed_dirs):
    """Load logs/all_results.json from each seed dir → list of {agent: dict}."""
    per_seed = []
    used = []
    for d in seed_dirs:
        p = Path(d) / 'logs' / 'all_results.json'
        if not p.exists():
            print(f'  (skip) no all_results.json in {d}')
            continue
        with open(p) as f:
            data = json.load(f)
        per_seed.append({r['agent_name']: r for r in data})
        used.append(str(d))
    return per_seed, used


def main():
    ap = argparse.ArgumentParser(description='Aggregate per-seed results (R1)')
    ap.add_argument('--env', choices=list(ENV_GLOBS.keys()), default=None,
                    help='Built-in glob shortcut for the per-seed dirs')
    ap.add_argument('--dirs', nargs='+', default=None,
                    help='Explicit list of per-seed result dirs')
    ap.add_argument('--out-dir', default='results/_aggregate',
                    help='Where to write the summary tables')
    args = ap.parse_args()

    if args.dirs:
        seed_dirs = args.dirs
        name = 'custom'
    elif args.env:
        pattern = str(PROJECT_ROOT / ENV_GLOBS[args.env])
        seed_dirs = sorted(glob.glob(pattern))
        name = args.env
        print(f'  glob {pattern} -> {len(seed_dirs)} dir(s)')
    else:
        ap.error('provide --env or --dirs')

    per_seed, used = load_seed_results(seed_dirs)
    if not per_seed:
        print('No per-seed results found. Nothing to aggregate.')
        return

    agents, rows = aggregate_metrics(per_seed)
    n = len(per_seed)
    title = f'Seed-aggregated results ({name}): {n} seed(s) from {len(used)} dir(s).'
    txt = to_plain_text(agents, rows, n, title=title)
    tex = to_latex(agents, rows, n,
                   caption=f'{name}: execution performance, mean $\\pm$ std over '
                           f'{n} seeds (bps).',
                   label=f'tab:{name}_seed_summary')

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{name}_seed_summary.txt').write_text(txt)
    (out_dir / f'{name}_seed_summary.tex').write_text(tex)
    print(txt)
    print(f'Wrote {out_dir}/{name}_seed_summary.{{txt,tex}}')


if __name__ == '__main__':
    main()
