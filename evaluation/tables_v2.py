"""
evaluation/tables_v2.py
-----------------------
Design-v2 (B2) table generators.

Thin, import-safe formatting layer that REUSES the existing, tested table cores
rather than duplicating them:

  - Seed aggregation (mean +/- std across seeds) comes straight from
    ``experiments/aggregate_seeds.py`` (``load_seed_results``,
    ``aggregate_metrics``, ``to_plain_text``, ``to_latex``, ``METRICS``).
  - The CVaR-alpha ladder evaluation comes straight from
    ``experiments/sweep_cvar_alpha.py`` (``evaluate_at_alpha``), i.e. the
    fixed-weight IQN-neutral checkpoint evaluated at a range of inference-time
    tau-truncation levels alpha.

What is *new* here is the Design-v2 main comparison table, which adds a
``cap-frac`` column (the fraction of decision steps that traded exactly at the
per-step cap, from ``EpisodeTracker.cap_frac``) alongside the usual IS/CVaR
columns.

No side effects at import beyond the standard sys.path bootstrap needed to reach
the ``experiments/`` modules; nothing is trained or written on import.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- sys.path bootstrap so the experiments/ modules import cleanly -----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
for _p in (str(PROJECT_ROOT), str(EXPERIMENTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# aggregate_seeds is lightweight (numpy only) — safe to import at module load.
# sweep_cvar_alpha pulls in torch/agents, so it is imported lazily inside
# run_alpha_ladder to keep `import evaluation.tables_v2` cheap.
import aggregate_seeds  # noqa: E402

METRICS = aggregate_seeds.METRICS  # (json_key, label) pairs for seed aggregation


# ---------------------------------------------------------------------------
# Column specs
# ---------------------------------------------------------------------------

# Main Design-v2 comparison table columns: the usual IS/CVaR block plus the new
# cap-frac diagnostic. (json key, display label).
COMPARISON_COLUMNS = [('mean_IS_bps', 'Mean'), ('std_IS_bps', 'Std'),
                      ('CVaR_0.90_bps', 'CVaR90'), ('CVaR_0.95_bps', 'CVaR95'),
                      ('max_IS_bps', 'Max'), ('cap_frac', 'cap-frac')]

# CVaR-alpha ladder columns (the alpha column is rendered separately).
ALPHA_COLUMNS = [('mean_IS_bps', 'Mean'), ('std_IS_bps', 'Std'),
                 ('CVaR_0.90_bps', 'CVaR90'), ('CVaR_0.95_bps', 'CVaR95'),
                 ('max_IS_bps', 'Max')]

_NA = 'n/a'
_MIN_COL_W = 8


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------

def _fmt_cell(value, decimals) -> str:
    """Format a numeric cell to ``decimals`` dp; ``None``/non-numeric -> 'n/a'."""
    if value is None:
        return _NA
    try:
        return f'{float(value):.{decimals}f}'
    except (TypeError, ValueError):
        return _NA


def _decimals_for(key) -> int:
    """cap-frac is a plain fraction (3 dp); bps metrics get 4 dp."""
    return 3 if key == 'cap_frac' else 4


# ---------------------------------------------------------------------------
# Main comparison table (adds the cap-frac column)
# ---------------------------------------------------------------------------

def format_comparison_table(all_results, order=None, title=''):
    """Fixed-width comparison table, one row per agent.

    Args:
        all_results: list of per-agent metric dicts, each with 'agent_name' plus
                     the COMPARISON_COLUMNS keys (bps metrics + a plain
                     'cap_frac' fraction in [0, 1]). Missing keys render 'n/a'.
        order: optional list of agent_name to force row order (default: input
               order). Agents present but not listed are appended in input order.
        title: optional header line printed above the table.

    Returns:
        A newline-terminated string (Agent column + the 6 COMPARISON_COLUMNS).
    """
    # Row ordering.
    if order:
        by_name = {r.get('agent_name'): r for r in all_results}
        wanted = set(order)
        rows_data = [by_name[n] for n in order if n in by_name]
        rows_data += [r for r in all_results if r.get('agent_name') not in wanted]
    else:
        rows_data = list(all_results)

    def cell(r, key):
        return _fmt_cell(r.get(key), _decimals_for(key))

    # Dynamic column widths for clean alignment.
    name_w = max([len('Agent')]
                 + [len(str(r.get('agent_name', '???'))) for r in rows_data])
    name_w = max(name_w, 16)
    col_w = []
    for key, label in COMPARISON_COLUMNS:
        w = max([len(label)] + [len(cell(r, key)) for r in rows_data] or [0])
        col_w.append(max(w, _MIN_COL_W))

    lines = []
    if title:
        lines.append(title)
    header = f'{"Agent":<{name_w}s}' + ''.join(
        f' | {label:>{col_w[i]}s}' for i, (_k, label) in enumerate(COMPARISON_COLUMNS))
    lines.append(header)
    lines.append('-' * len(header))
    for r in rows_data:
        name = str(r.get('agent_name', '???'))
        row = f'{name:<{name_w}s}'
        for i, (key, _label) in enumerate(COMPARISON_COLUMNS):
            row += f' | {cell(r, key):>{col_w[i]}s}'
        lines.append(row)
    return '\n'.join(lines) + '\n'


def write_comparison_table(all_results, out_path, order=None, title=''):
    """Format the comparison table and write it to ``out_path``.

    Creates parent directories. Returns the formatted text.
    """
    txt = format_comparison_table(all_results, order=order, title=title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt)
    return txt


# ---------------------------------------------------------------------------
# Seed-aggregated summary (reuses aggregate_seeds wholesale)
# ---------------------------------------------------------------------------

def write_seed_aggregate(seed_dirs, name, out_dir, caption='', label=''):
    """Aggregate per-seed results into mean +/- std tables (text + LaTeX).

    Reuses ``aggregate_seeds.{load_seed_results, aggregate_metrics,
    to_plain_text, to_latex}`` and ``aggregate_seeds.METRICS``.

    Args:
        seed_dirs: list of per-seed result dirs, each containing
                   ``logs/all_results.json`` (a list of per-agent metric dicts
                   with 'agent_name' + *_bps keys) — exactly what
                   ``load_seed_results`` expects.
        name: basename for the output files (``<name>_seed_summary.{txt,tex}``).
        out_dir: directory to write into (created if absent).
        caption/label: LaTeX caption/label; sensible defaults derived from
                       ``name`` when empty.

    Returns:
        (txt_path, tex_path) as ``pathlib.Path``.
    """
    per_seed, used = aggregate_seeds.load_seed_results(seed_dirs)
    if not per_seed:
        raise ValueError(
            f'write_seed_aggregate: no per-seed all_results.json found under '
            f'{list(seed_dirs)!r} (expected <dir>/logs/all_results.json).')

    agents, rows = aggregate_seeds.aggregate_metrics(per_seed, metrics=METRICS)
    n = len(per_seed)
    title = (f'Seed-aggregated results ({name}): {n} seed(s) from '
             f'{len(used)} dir(s).')
    if not caption:
        caption = (f'{name}: execution performance, mean $\\pm$ std over '
                   f'{n} seeds (bps).')
    if not label:
        label = f'tab:{name}_seed_summary'

    txt = aggregate_seeds.to_plain_text(agents, rows, n, metrics=METRICS,
                                        title=title)
    tex = aggregate_seeds.to_latex(agents, rows, n, metrics=METRICS,
                                   caption=caption, label=label)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f'{name}_seed_summary.txt'
    tex_path = out_dir / f'{name}_seed_summary.tex'
    txt_path.write_text(txt)
    tex_path.write_text(tex)
    return txt_path, tex_path


# ---------------------------------------------------------------------------
# CVaR-alpha ladder (reuses sweep_cvar_alpha.evaluate_at_alpha)
# ---------------------------------------------------------------------------

def run_alpha_ladder(iqn_agent, env, alphas, n_eval, seed):
    """Evaluate a fixed-weight IQN agent across CVaR-truncation levels alpha.

    Reuses ``sweep_cvar_alpha.evaluate_at_alpha`` (which flips
    ``iqn_agent.cfg.cvar_alpha`` — inference-time tau truncation only — and never
    touches the network weights). The agent's original ``cfg.cvar_alpha`` is
    restored before returning.

    Args:
        iqn_agent: a trained IQNAgent (fixed weights).
        env: the evaluation environment.
        alphas: iterable of alpha levels to evaluate.
        n_eval: number of evaluation episodes per alpha.
        seed: RNG seed passed through to ``evaluate_at_alpha``.

    Returns:
        list of row dicts (ascending alpha), each with keys 'alpha',
        'mean_IS_bps', 'std_IS_bps', 'CVaR_0.90_bps', 'CVaR_0.95_bps',
        'max_IS_bps'.
    """
    import sweep_cvar_alpha  # lazy: pulls in torch/agents

    original_alpha = iqn_agent.cfg.cvar_alpha
    rows = []
    try:
        for a in sorted(alphas):
            m = sweep_cvar_alpha.evaluate_at_alpha(iqn_agent, env, n_eval, seed, a)
            rows.append({
                'alpha':         float(a),
                'mean_IS_bps':   float(m['mean_IS_bps']),
                'std_IS_bps':    float(m['std_IS_bps']),
                'CVaR_0.90_bps': float(m['CVaR_0.90_bps']),
                'CVaR_0.95_bps': float(m['CVaR_0.95_bps']),
                'max_IS_bps':    float(m['max_IS_bps']),
            })
    finally:
        iqn_agent.cfg.cvar_alpha = original_alpha
    return rows


def _alpha_str(row) -> str:
    a = row.get('alpha')
    if a is None:
        return _NA
    try:
        return f'{float(a):g}'
    except (TypeError, ValueError):
        return _NA


def format_alpha_ladder(rows, title=''):
    """Fixed-width text table: alpha | Mean | Std | CVaR90 | CVaR95 | Max (bps)."""
    a_strs = [_alpha_str(r) for r in rows]
    alpha_w = max([len('alpha')] + [len(s) for s in a_strs])

    def cell(r, key):
        return _fmt_cell(r.get(key), 4)

    col_w = []
    for key, label in ALPHA_COLUMNS:
        w = max([len(label)] + [len(cell(r, key)) for r in rows] or [0])
        col_w.append(max(w, _MIN_COL_W))

    lines = []
    if title:
        lines.append(title)
    lines.append('CVaR-alpha ladder; all IS metrics in bps.')
    header = f'{"alpha":<{alpha_w}s}' + ''.join(
        f' | {label:>{col_w[i]}s}' for i, (_k, label) in enumerate(ALPHA_COLUMNS))
    lines.append(header)
    lines.append('-' * len(header))
    for r, s in zip(rows, a_strs):
        row = f'{s:<{alpha_w}s}'
        for i, (key, _label) in enumerate(ALPHA_COLUMNS):
            row += f' | {cell(r, key):>{col_w[i]}s}'
        lines.append(row)
    return '\n'.join(lines) + '\n'


def alpha_ladder_latex(rows, caption='', label=''):
    """booktabs LaTeX table for the CVaR-alpha ladder (matches to_latex style)."""
    col_spec = 'l' + 'r' * len(ALPHA_COLUMNS)
    out = []
    out.append(r'\begin{table}[H]')
    out.append(r'    \centering')
    if caption:
        out.append(f'    \\caption{{{caption}}}')
    if label:
        out.append(f'    \\label{{{label}}}')
    out.append(f'    \\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}')
    out.append(r'        \toprule')
    head = r'$\alpha$ & ' + ' & '.join(lab for _k, lab in ALPHA_COLUMNS) + r' \\'
    out.append('        ' + head)
    out.append(r'        \midrule')
    for r in rows:
        cells = [_fmt_cell(r.get(key), 4) for key, _lab in ALPHA_COLUMNS]
        out.append('        ' + _alpha_str(r) + ' & ' + ' & '.join(cells) + r' \\')
    out.append(r'        \bottomrule')
    out.append(r'    \end{tabular}')
    out.append(r'\end{table}')
    return '\n'.join(out) + '\n'


def write_alpha_ladder(rows, out_dir, basename='alpha_ladder', caption='', label=''):
    """Write the alpha-ladder text + LaTeX tables.

    Returns (txt_path, tex_path) as ``pathlib.Path``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = format_alpha_ladder(rows)
    tex = alpha_ladder_latex(rows, caption=caption, label=label)
    txt_path = out_dir / f'{basename}.txt'
    tex_path = out_dir / f'{basename}.tex'
    txt_path.write_text(txt)
    tex_path.write_text(tex)
    return txt_path, tex_path
