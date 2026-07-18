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


# ---------------------------------------------------------------------------
# Regime breakdown: stress-hit vs calm (Design-v2 B3, table T-RG-3)
# ---------------------------------------------------------------------------

# Columns of the stress-hit vs calm table. Each is (source_key, display label,
# decimals). ``hit_frac`` is a plain fraction from the top level of a
# ``conditional_stats`` result (3 dp); the other four are IS in bps (4 dp)
# pulled from the 'hit' (stress) and 'calm' groups.
REGIME_COLUMNS = [('hit_frac',      'hit-frac',      3),
                  ('stress_mean',   'stress Mean',   4),
                  ('stress_cvar95', 'stress CVaR95', 4),
                  ('calm_mean',     'calm Mean',     4),
                  ('calm_cvar95',   'calm CVaR95',   4)]


def _fmt_num(value, decimals) -> str:
    """Format a numeric cell to ``decimals`` dp; ``None``/NaN/non-numeric -> 'n/a'.

    Unlike ``_fmt_cell`` this also maps NaN to 'n/a' — ``conditional_stats``
    reports NaN for an empty stress/calm group, which must render as missing.
    """
    if value is None:
        return _NA
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _NA
    if f != f:  # NaN (empty group from conditional_stats)
        return _NA
    return f'{f:.{decimals}f}'


def _regime_cells(stats) -> dict:
    """Flatten one ``conditional_stats`` result into the REGIME_COLUMNS keys.

    ``stats`` is the dict returned by ``evaluation.metrics.conditional_stats``:
    ``{'hit': {...}, 'calm': {...}, 'hit_frac': float, 'n': int}``. The 'hit'
    group is the stress group. Missing keys/groups yield ``None`` -> 'n/a'.
    """
    stats = stats or {}
    hit  = stats.get('hit')  or {}
    calm = stats.get('calm') or {}
    return {
        'hit_frac':      stats.get('hit_frac'),
        'stress_mean':   hit.get('mean_IS_bps'),
        'stress_cvar95': hit.get('CVaR_0.95_bps'),
        'calm_mean':     calm.get('mean_IS_bps'),
        'calm_cvar95':   calm.get('CVaR_0.95_bps'),
    }


def _regime_order(per_agent, order):
    """Row order: honour ``order`` first, then append any unlisted agents."""
    names = list(per_agent.keys())
    if order:
        wanted  = set(order)
        ordered = [n for n in order if n in per_agent]
        ordered += [n for n in names if n not in wanted]
        return ordered
    return names


def regime_breakdown_table(per_agent, order=None, title='') -> str:
    """Fixed-width stress-hit vs calm IS table, one row per agent.

    Args:
        per_agent: dict ``{agent_name: conditional_stats_result}`` where each
                   value is the dict returned by
                   ``evaluation.metrics.conditional_stats`` (keys 'hit', 'calm',
                   'hit_frac', 'n'). The 'hit' group is the stress group.
        order:     optional list of agent_name to force row order (unlisted
                   agents are appended in dict order).
        title:     optional header line printed above the table.

    Returns:
        A newline-terminated string with columns:
        Agent | hit-frac | stress Mean | stress CVaR95 | calm Mean | calm CVaR95.
        All IS metrics in bps (4 dp); hit-frac 3 dp; NaN/missing render 'n/a'.
    """
    ordered = _regime_order(per_agent, order)
    cells_by_name = {n: _regime_cells(per_agent[n]) for n in ordered}

    name_w = max([len('Agent')] + [len(str(n)) for n in ordered])
    name_w = max(name_w, 16)
    col_w = []
    for key, label, dec in REGIME_COLUMNS:
        w = max([len(label)]
                + [len(_fmt_num(cells_by_name[n].get(key), dec)) for n in ordered]
                or [0])
        col_w.append(max(w, _MIN_COL_W))

    lines = []
    if title:
        lines.append(title)
    lines.append('Stress-hit vs calm IS breakdown; IS metrics in bps.')
    header = f'{"Agent":<{name_w}s}' + ''.join(
        f' | {label:>{col_w[i]}s}'
        for i, (_k, label, _d) in enumerate(REGIME_COLUMNS))
    lines.append(header)
    lines.append('-' * len(header))
    for n in ordered:
        row = f'{str(n):<{name_w}s}'
        for i, (key, _label, dec) in enumerate(REGIME_COLUMNS):
            row += f' | {_fmt_num(cells_by_name[n].get(key), dec):>{col_w[i]}s}'
        lines.append(row)
    return '\n'.join(lines) + '\n'


def write_regime_breakdown(per_agent, out_path, order=None, title='') -> str:
    """Format the stress-hit vs calm table and write it to ``out_path``.

    Creates parent directories. Returns the formatted text.
    """
    txt = regime_breakdown_table(per_agent, order=order, title=title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt)
    return txt


def regime_breakdown_latex(per_agent, order=None, caption='', label='') -> str:
    """booktabs LaTeX version of the stress-hit vs calm table.

    Same columns as ``regime_breakdown_table``; matches the
    ``aggregate_seeds.to_latex`` style (table[H] / centering / booktabs rules).
    """
    ordered = _regime_order(per_agent, order)
    col_spec = 'l' + 'r' * len(REGIME_COLUMNS)
    out = []
    out.append(r'\begin{table}[H]')
    out.append(r'    \centering')
    if caption:
        out.append(f'    \\caption{{{caption}}}')
    if label:
        out.append(f'    \\label{{{label}}}')
    out.append(f'    \\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}')
    out.append(r'        \toprule')
    head = 'Agent & ' + ' & '.join(lab for _k, lab, _d in REGIME_COLUMNS) + r' \\'
    out.append('        ' + head)
    out.append(r'        \midrule')
    for n in ordered:
        cells_map = _regime_cells(per_agent[n])
        cells = [_fmt_num(cells_map.get(key), dec)
                 for key, _lab, dec in REGIME_COLUMNS]
        safe = str(n).replace('_', r'\_')
        out.append('        ' + safe + ' & ' + ' & '.join(cells) + r' \\')
    out.append(r'        \bottomrule')
    out.append(r'    \end{tabular}')
    out.append(r'\end{table}')
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------------------
# Action-vs-spread heatmap (input to the sigma-hat on/off decision)
# ---------------------------------------------------------------------------

def action_spread_heatmap(agent, env, n_episodes, seed, out_dir,
                          basename='action_spread_heatmap',
                          n_spread_bins=10):
    """Roll ``agent`` on ``env`` and heatmap chosen action vs normalised spread.

    At every decision step of ``n_episodes`` greedy rollouts, records
    ``(spread_norm = state[3], chosen action index)``. Bins the spread into
    ``n_spread_bins`` equal-width bins spanning the observed range and builds an
    ``(n_spread_bins x n_actions)`` count matrix, row-normalised to a
    within-spread-bin action frequency. Writes both a CSV (matrix + spread-bin
    edges + action fractions) and a PNG imshow heatmap.

    This diagnostic answers the sigma-hat on/off question: if the chosen action
    varies with spread, the spread feature (state[3]) carries policy-relevant
    information.

    Args:
        agent:        an agent with ``select_action(state, eval_mode=True)``
                      (and optionally ``reset()``); ``n_actions = env.n_actions``.
        env:          a RegimeJumpEnv (or any BaseExecutionEnv); reseeded via
                      ``env.seed(seed)`` for reproducibility.
        n_episodes:   number of greedy rollouts to collect.
        seed:         RNG seed for the environment.
        out_dir:      directory to write into (created if absent).
        basename:     basename for the ``.csv`` / ``.png`` outputs.
        n_spread_bins: number of equal-width spread bins (heatmap rows).

    Returns:
        ``(csv_path, png_path)`` as ``pathlib.Path``.
    """
    import numpy as np
    import csv as _csv
    import matplotlib
    matplotlib.use('Agg')          # non-interactive backend (no display)
    import matplotlib.pyplot as plt

    n_actions = int(env.n_actions)

    # Action fractions for labelling: prefer the agent's grid, fall back to the
    # env's, else a plain index range.
    fracs = getattr(agent, 'action_fracs', None)
    if fracs is None:
        fracs = getattr(env, '_action_fracs', None)
    if fracs is None:
        fracs = np.arange(n_actions)
    action_fracs = np.asarray(fracs, dtype=np.float64)

    # --- Collect (spread_norm, action) pairs over greedy rollouts -----------
    env.seed(seed)
    spreads, actions = [], []
    for _ep in range(int(n_episodes)):
        state = env.reset()
        if hasattr(agent, 'reset') and callable(getattr(agent, 'reset')):
            agent.reset()
        done = False
        while not done:
            a = agent.select_action(state, eval_mode=True)
            spreads.append(float(state[3]))          # spread_norm = spread / p0
            actions.append(int(a))
            state, _reward, done, _info = env.step(a)

    spreads = np.asarray(spreads, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)

    # --- Spread-bin edges over the observed range ---------------------------
    if spreads.size:
        lo, hi = float(spreads.min()), float(spreads.max())
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, n_spread_bins + 1)

    # digitize -> [0, n_spread_bins-1] (max value lands in the last bin).
    bin_idx = np.clip(np.digitize(spreads, edges) - 1, 0, n_spread_bins - 1)

    counts = np.zeros((n_spread_bins, n_actions), dtype=np.float64)
    for b, a in zip(bin_idx, actions):
        if 0 <= a < n_actions:
            counts[int(b), int(a)] += 1.0

    # Row-normalise to within-spread-bin action frequency.
    row_sums = counts.sum(axis=1, keepdims=True)
    freq = np.divide(counts, row_sums, out=np.zeros_like(counts),
                     where=row_sums > 0)

    # --- Write outputs ------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'{basename}.csv'
    png_path = out_dir / f'{basename}.png'

    with csv_path.open('w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(['# action_spread_heatmap: within-spread-bin action frequency'])
        w.writerow([f'# n_episodes={int(n_episodes)}', f'seed={seed}',
                    f'n_steps={int(spreads.size)}'])
        w.writerow(['action_fracs'] + [f'{v:g}' for v in action_fracs.tolist()])
        w.writerow(['spread_bin_edges'] + [f'{e:.8g}' for e in edges.tolist()])
        w.writerow(['spread_bin', 'spread_lo', 'spread_hi']
                   + [f'action_{i}' for i in range(n_actions)])
        for b in range(n_spread_bins):
            w.writerow([b, f'{edges[b]:.8g}', f'{edges[b + 1]:.8g}']
                       + [f'{freq[b, i]:.6f}' for i in range(n_actions)])

    # --- PNG imshow heatmap (modelled on Visualizer.plot_action_heatmap) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(freq, aspect='auto', cmap='YlOrRd', origin='lower',
                   interpolation='nearest', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(
        [f'{action_fracs[i]:g}' if i < len(action_fracs) else str(i)
         for i in range(n_actions)], fontsize=9)
    ax.set_yticks(range(n_spread_bins))
    ax.set_yticklabels([f'{edges[b]:.4f}-{edges[b + 1]:.4f}'
                        for b in range(n_spread_bins)], fontsize=8)
    fig.colorbar(im, ax=ax, label='Action frequency (within spread bin)')
    ax.set_title('Action vs Spread Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Action (fraction of q0)', fontsize=12)
    ax.set_ylabel('Normalised spread bin (spread / p0)', fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return csv_path, png_path
