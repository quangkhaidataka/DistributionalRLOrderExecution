"""
experiments/run_v2_taq.py
-------------------------
Design-v2 B4 — full real-data TAQ study, staged CLI (same shape as run_v2_ac.py /
run_v2_regime.py) on the AAPL 2014 3-min split-adjusted bars, walk-forward folds.

Folds (fixed, one year of AAPL bars):
    train = months 1–7  (Jan–Jul)
    val   = months 8–9  (Aug–Sep)   ← CRN checkpoint selection
    test  = months 10–12 (Oct–Dec)  ← held-out test evaluation

Reuses run_v2_ac's proven staged machinery verbatim (``AC.train_segment`` with
byte-identical --episodes a:b resume — it captures/restores the TAQEnv's
``RandomState`` RNG as well as the sim envs' ``Generator`` — plus ``AC._parse_episodes``,
``AC._json_safe`` and the shared v2 constants), swapping in the TAQ env + config.

η-SCALE GATE: this runner REFUSES to train until the deterministic η-scale gate
(``scripts/eta_scale_check.py``) has written ``results/_v2_taq/eta_scale_report.json``
AND that report's top-level ``gate_ok`` is true. The gate is deterministic (no
training) so it is safe to run up-front; it guards against a mis-scaled impact
coefficient that would make every agent pin (or never touch) the cap.

NO full runs here (that is B5). ``--smoke`` shrinks everything for logic tests.

Usage
-----
  # (after: python scripts/eta_scale_check.py   — writes the η-scale gate report)
  python experiments/run_v2_taq.py --only-agent DQN         --seed 42
  python experiments/run_v2_taq.py --only-agent DDQN        --seed 42
  python experiments/run_v2_taq.py --only-agent IQN-neutral --episodes 0:25000 --total-episodes 50000 --seed 42
  python experiments/run_v2_taq.py --only-agent IQN-neutral --episodes 25000:50000 --total-episodes 50000 --seed 42
  # CRN-select (val fold) + test-eval (test fold) all agents + write tables
  python experiments/run_v2_taq.py --assemble-eval --seed 42
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # experiments/ siblings

import numpy as np
import pandas as pd
import torch

from envs.taq_env import TAQEnv, TAQConfig
from agents.baselines import (TWAPAgent, AlmgrenChrissAgent, MaxSpeedAgent,
                              DQNAgent, DDQNAgent, DeepRLConfig)
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.param_utils import expected_counts, assert_param_count
import run_simulation as RS
import run_v2_ac as AC                       # reuse train machinery + constants
import selection_lib
import evaluation.tables_v2 as tables_v2
from exp_utils import dump_config_json, prepare_v2_output_dir

# ---------------------------------------------------------------------------
# Constants (reuse run_v2_ac; TAQ-specific base config + folds + gate)
# ---------------------------------------------------------------------------

V2_GRID            = AC.V2_GRID
V2_CVAR_ALPHAS     = AC.V2_CVAR_ALPHAS
V2_REPLAY_CAPACITY = AC.V2_REPLAY_CAPACITY
LEARNED_TRAINABLE  = AC.LEARNED_TRAINABLE
MANIFEST_NAME      = AC.MANIFEST_NAME
VAL_SEED_OFFSET    = AC.VAL_SEED_OFFSET
TEST_SEED_OFFSET   = AC.TEST_SEED_OFFSET
SMOKE_TRAIN        = AC.SMOKE_TRAIN         # 200 eps / ckpt 100 / val 40 / test 100

DEFAULT_OUT_ROOT = 'results/_v2_taq'
STOCK            = 'AAPL'
YEAR             = 2014
BAR_SOURCE       = 'AAPL_2014_3min_adj.parquet'

# TAQ training schedule (B4): longer than the sim studies (50k) since a real-data
# episode = one day's N consecutive 3-min bars.
DEFAULT_TRAIN = dict(episodes=50_000, checkpoint_freq=2_000, n_val=1_200, n_test=10_000)

# Walk-forward fold split by month (1-indexed). Matches run_tag's expanding-window
# style but with a single fixed train/val/test partition of the 2014 bars.
FOLD_MONTHS = {'train': [1, 2, 3, 4, 5, 6, 7], 'val': [8, 9], 'test': [10, 11, 12]}

# η-scale gate report (written by scripts/eta_scale_check.py). Training refuses to
# start unless this exists AND its top-level "gate_ok" is true.
GATE_REPORT = PROJECT_ROOT / 'results' / '_v2_taq' / 'eta_scale_report.json'

# v2 TAQ base config. gamma / sigma stay at the EnvConfig defaults (2.5e-7 /
# 0.00095) — used only by the AC agent's schedule — matching scripts/eta_scale_check
# so the env replay is byte-identical to the gate run.
TAQ_BASE_CONFIG = dict(
    N=20, T=60.0, q0=5000, p0=100.0, eta=1e-5, a=1e-4,
    bar_source=BAR_SOURCE, bar_minutes=3,
    action_basis='q0', action_fracs=V2_GRID, use_rv_feature=False,
)


# ---------------------------------------------------------------------------
# Config, folds, gate
# ---------------------------------------------------------------------------

def build_taq_config() -> TAQConfig:
    return TAQConfig(data_dir=str(PROJECT_ROOT / 'data' / 'processed'),
                     stock=STOCK, year=YEAR, **TAQ_BASE_CONFIG)


def get_dates_for_months(all_dates: List[str], months: List[int]) -> List[str]:
    """Filter 'YYYY-MM-DD' date strings by month number (mirrors run_tag)."""
    return [d for d in all_dates if int(d[5:7]) in months]


def load_folds(cfg: TAQConfig):
    """Read all bar dates from the parquet and split into train/val/test folds."""
    parquet_path = Path(cfg.data_dir) / (cfg.bar_source or f'{cfg.stock}_{cfg.year}.parquet')
    if not parquet_path.exists():
        raise SystemExit(f'TAQ bars not found: {parquet_path}\n'
                         f'  Build them with data/build_taq_3min.py first.')
    full_df = pd.read_parquet(parquet_path, columns=['date'])
    all_dates = sorted(full_df['date'].unique().tolist())
    train = get_dates_for_months(all_dates, FOLD_MONTHS['train'])
    val   = get_dates_for_months(all_dates, FOLD_MONTHS['val'])
    test  = get_dates_for_months(all_dates, FOLD_MONTHS['test'])
    for nm, ds in (('train', train), ('val', val), ('test', test)):
        if not ds:
            raise SystemExit(f'No {nm} dates for months {FOLD_MONTHS[nm]} '
                             f'in {parquet_path}.')
    return all_dates, train, val, test


def require_eta_gate() -> dict:
    """Refuse to train unless the deterministic η-scale gate report says gate_ok."""
    if not GATE_REPORT.exists():
        raise SystemExit(
            f'η-scale gate report missing: {GATE_REPORT}\n'
            f'  The v2 TAQ study refuses to train until the deterministic η-scale '
            f'gate has run.\n'
            f'  Run first:  python scripts/eta_scale_check.py')
    data = json.loads(GATE_REPORT.read_text())
    if not bool(data.get('gate_ok', False)):
        raise SystemExit(
            f'η-scale gate is NOT OK (gate_ok={data.get("gate_ok")!r}) in {GATE_REPORT}.\n'
            f'  Re-run with a sane impact scale:  '
            f'python scripts/eta_scale_check.py --eta <value>')
    return data


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def build_taq_agents(cfg: TAQConfig, train_dates: List[str], seed: int, device_str: str):
    """All v2 agents in canonical order + a fresh TRAIN-fold TAQEnv.

    Learned agents get action_fracs/action_basis so they mask (q0 mode, cap 0.25);
    the rule-based TWAP/AC/MaxSpeed read the grid from the config. Replay 100k. The
    param guard uses the TRAIN env's (state_dim, n_actions).
    """
    device = torch.device(device_str)
    env = TAQEnv(cfg, train_dates)
    sd, na = env.state_dim, env.n_actions
    grid, basis = cfg.action_fracs, cfg.action_basis

    agents = {}
    agents['TWAP']     = TWAPAgent(cfg)
    agents['AC']       = AlmgrenChrissAgent(cfg, risk_aversion=1e-6)
    agents['MaxSpeed'] = MaxSpeedAgent(cfg)
    drl = DeepRLConfig(replay_capacity=V2_REPLAY_CAPACITY)
    agents['DQN']  = DQNAgent(drl, sd, na, device=device, seed=seed,
                              action_fracs=grid, action_basis=basis)
    agents['DDQN'] = DDQNAgent(drl, sd, na, device=device, seed=seed + 1,
                               action_fracs=grid, action_basis=basis)
    iqn = AgentConfig(cvar_alpha=1.0, replay_capacity=V2_REPLAY_CAPACITY)
    agents['IQN-neutral'] = IQNAgent(iqn, sd, na, device=device, seed=seed + 3,
                                     action_fracs=grid, action_basis=basis)
    for al in V2_CVAR_ALPHAS:
        agents[f'IQN-CVaR_{al:.2f}'] = IQNAgent(
            AgentConfig(cvar_alpha=al, replay_capacity=V2_REPLAY_CAPACITY),
            sd, na, device=device, seed=seed + 3,
            action_fracs=grid, action_basis=basis)

    exp_iqn, exp_mlp = expected_counts(sd, na)
    for name, ag in agents.items():
        if hasattr(ag, 'get_num_params'):
            assert_param_count(ag, exp_iqn if name.startswith('IQN') else exp_mlp, name)
    return agents, env


# ---------------------------------------------------------------------------
# Manifest (consistency across staged jobs sharing a dir)
# ---------------------------------------------------------------------------

_MANIFEST_MATCH_KEYS = ('taq_config', 'folds', 'seed', 'total_episodes',
                        'checkpoint_freq', 'n_val', 'n_test', 'device', 'smoke')


def _manifest_payload(taq_config, folds, seed, total, ckpt_freq, n_val, n_test,
                      device, smoke) -> dict:
    return {'env': 'taq', 'stock': STOCK, 'year': YEAR,
            'taq_config': asdict(taq_config), 'folds': folds, 'seed': seed,
            'total_episodes': total, 'checkpoint_freq': ckpt_freq,
            'n_val': n_val, 'n_test': n_test, 'device': device, 'smoke': smoke}


def write_or_check_manifest(log_dir: Path, payload: dict) -> dict:
    path = log_dir / MANIFEST_NAME
    if not path.exists():
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        return payload
    existing = json.loads(path.read_text())
    diffs = [k for k in _MANIFEST_MATCH_KEYS if existing.get(k) != payload.get(k)]
    if diffs:
        raise SystemExit(
            f'Staged manifest mismatch in {path} on {diffs}.\n'
            f'  All staged jobs for one dir must share config/folds/seed/schedule/device.\n'
            f'  Archive the dir and start over, or fix the CLI args.')
    return existing


def _resolve_out(args) -> Path:
    # Per-seed namespacing: results/_v2_taq/<STOCK>/seed<S>/ (mirrors run_v2_ac /
    # run_v2_regime). The shared, data-level η-scale gate report lives ABOVE this
    # (results/_v2_taq/eta_scale_report.json) and is reused across seeds. An
    # explicit --out-dir still overrides to that exact path (unchanged).
    out = Path(args.out_dir) if args.out_dir else (
        Path(DEFAULT_OUT_ROOT) / ('_smoke' if args.smoke else '') / STOCK
        / f'seed{args.seed}')
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


# ---------------------------------------------------------------------------
# Train one agent (segment)  — delegates to run_v2_ac.train_segment
# ---------------------------------------------------------------------------

def cmd_only_agent(args):
    name = args.only_agent
    require_eta_gate()                       # REFUSE unless η-scale gate is OK
    taq_config = build_taq_config()
    _all, train_dates, _val, _test = load_folds(taq_config)
    tr = SMOKE_TRAIN if args.smoke else DEFAULT_TRAIN
    ckpt_freq = args.checkpoint_freq or tr['checkpoint_freq']
    ep_start, ep_end = (AC._parse_episodes(args.episodes) if args.episodes is not None
                        else (0, tr['episodes']))
    total = args.total_episodes or ep_end
    if ep_end > total:
        raise SystemExit(f'--episodes end {ep_end} > --total-episodes {total}')

    out = _resolve_out(args)
    prepare_v2_output_dir(out, config=None, force_resume=True)   # shared staged dir
    ckpt_dir, log_dir = out / 'checkpoints', out / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    write_or_check_manifest(log_dir, _manifest_payload(
        taq_config, FOLD_MONTHS, args.seed, total, ckpt_freq, tr['n_val'],
        tr['n_test'], args.device, args.smoke))

    log_path    = log_dir / f'{name}_training.json'
    resume_path = log_dir / f'{name}_resume.pt'
    if ep_start == 0 and log_path.exists() and not args.force_resume:
        raise SystemExit(f'{name} already trained here ({log_path}). Archive to redo.')

    agents, train_env = build_taq_agents(taq_config, train_dates, args.seed, args.device)
    if name not in agents:
        raise SystemExit(f'--only-agent must be one of {LEARNED_TRAINABLE}')

    print(f'\n### v2-TAQ STAGED TRAIN {name} eps({ep_start}:{ep_end}/{total}) '
          f'ckpt/{ckpt_freq} seed={args.seed} device={args.device} '
          f'folds(train={FOLD_MONTHS["train"]}) '
          f'{"[SMOKE]" if args.smoke else ""} -> {out}')
    log = AC.train_segment(agents[name], train_env, name, ep_start, ep_end, total,
                          args.seed, ckpt_dir, ckpt_freq, resume_path)

    with open(log_path, 'w') as f:
        json.dump(log, f)
    dump_config_json(
        {'kind': 'v2_taq_staged_train', 'agent': name, 'stock': STOCK,
         'taq_config': taq_config, 'folds': FOLD_MONTHS, 'seed': args.seed,
         'episodes': f'{ep_start}:{ep_end}', 'total_episodes': total,
         'checkpoint_freq': ckpt_freq, 'device': args.device, 'smoke': args.smoke},
        log_dir / f'staged_{name}_config.json')
    print(f'  {name} segment done in {log["wall_time"]:.1f}s '
          f'(episodes_done={log["episodes_done"]}). log -> {log_path}')


# ---------------------------------------------------------------------------
# Assemble-eval  (CRN selection on VAL fold + test on TEST fold + tables)
# ---------------------------------------------------------------------------

def cmd_assemble_eval(args):
    out = _resolve_out(args)
    ckpt_dir, log_dir = out / 'checkpoints', out / 'logs'
    if not (log_dir / MANIFEST_NAME).exists():
        raise SystemExit(f'No staged manifest in {log_dir}; train agents first.')
    manifest = json.loads((log_dir / MANIFEST_NAME).read_text())
    if args.device != manifest['device']:
        raise SystemExit(f'--device {args.device} != training device '
                         f'{manifest["device"]} (FP bits differ).')

    taq_config = TAQConfig(**manifest['taq_config'])
    seed   = manifest['seed']
    n_val  = manifest['n_val']
    n_test = manifest['n_test']
    val_seed, test_seed = seed + VAL_SEED_OFFSET, seed + TEST_SEED_OFFSET

    _all, train_dates, val_dates, test_dates = load_folds(taq_config)
    agents, _train_env = build_taq_agents(taq_config, train_dates, seed, args.device)
    canon = list(agents.keys())

    # ── Checkpoint selection on the VAL fold (CRN, min val-CVaR₉₅) ──────────
    for name in LEARNED_TRAINABLE:
        paths = selection_lib.ckpt_paths_for(ckpt_dir, name)
        if not paths:
            raise SystemExit(f'No checkpoints for {name} in {ckpt_dir}. '
                             f'Train it before --assemble-eval.')
        sel = selection_lib.select_checkpoints(
            agents[name], paths, TAQEnv(taq_config, val_dates), val_seed,
            n_val=n_val, alpha=0.95)
        agents[name].load(sel['best_cvar'])
        print(f'  {name}: selected {Path(sel["best_cvar"]).name} '
              f'(best-by-val-CVaR₉₅ of {len(paths)} ckpts; '
              f'plateau={len(sel.get("plateau", []))})')
    RS.share_iqn_weights(agents)

    # ── Test evaluation on the TEST fold (all agents; cap_frac via compute_metrics)
    print(f'\n### v2-TAQ TEST EVAL {n_test} eps seed={test_seed} '
          f'test-months={FOLD_MONTHS["test"]}')
    all_results, trackers = RS.evaluate_all(agents, TAQEnv(taq_config, test_dates),
                                            n_eval=n_test, seed=test_seed)

    # ── Comparison table (with cap-frac column) ────────────────────────────
    title = (f'v2-TAQ comparison ({STOCK}, seed {seed}, {n_test} test eps)'
             f'{" [SMOKE]" if manifest.get("smoke") else ""}')
    table = tables_v2.write_comparison_table(
        all_results, log_dir / 'comparison_table.txt', order=canon, title=title)
    print('\n' + table)

    # ── α-ladder on the selected IQN-neutral checkpoint (eval-only) ────────
    ladder = sorted(set(V2_CVAR_ALPHAS + [1.0]))
    rows = tables_v2.run_alpha_ladder(
        agents['IQN-neutral'], TAQEnv(taq_config, test_dates),
        ladder, n_test, test_seed)
    tables_v2.write_alpha_ladder(
        rows, log_dir,
        caption=f'v2-TAQ CVaR-$\\alpha$ ladder ({STOCK}, seed {seed}).',
        label='tab:v2_taq_alpha')
    print('\n' + tables_v2.format_alpha_ladder(rows, title='α-ladder (IQN-neutral)'))

    # ── Raw arrays + config ────────────────────────────────────────────────
    is_dict = {r['agent_name']: (np.array(trackers[r['agent_name']].is_values) * 1e4)
               for r in all_results}                            # bps
    with open(log_dir / 'all_results.json', 'w') as f:
        json.dump([AC._json_safe(r) for r in all_results], f, indent=2)
    with open(log_dir / 'is_arrays.pkl', 'wb') as f:
        pickle.dump({k: v for k, v in is_dict.items()}, f)
    dump_config_json(
        {'kind': 'v2_taq_assemble_eval', 'stock': STOCK, 'seed': seed,
         'folds': FOLD_MONTHS, 'n_val': n_val, 'n_test': n_test,
         'val_seed': val_seed, 'test_seed': test_seed, 'alphas': ladder,
         'device': args.device},
        log_dir / 'assemble_config.json')
    print(f'\nAssembled -> {log_dir} '
          f'(comparison_table.txt, all_results.json, is_arrays.pkl, '
          f'alpha_ladder.{{txt,tex}}, assemble_config.json)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Design-v2 B4 staged TAQ study CLI')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--only-agent', choices=list(LEARNED_TRAINABLE),
                      help='Train exactly this one agent into the shared out-dir')
    mode.add_argument('--assemble-eval', action='store_true',
                      help='CRN-select (val fold), share, test-eval (test fold), tables')

    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--episodes', default=None,
                    help="'a:b' segment or 'N' (default: full/ smoke episodes)")
    ap.add_argument('--total-episodes', type=int, default=None,
                    help='Full training length (for split resume; default = segment end)')
    ap.add_argument('--checkpoint-freq', type=int, default=None)
    ap.add_argument('--out-dir', default=None,
                    help='Override output dir (default results/_v2_taq[/_smoke]/AAPL)')
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--smoke', action='store_true',
                    help='Tiny config (200 eps, ckpt/100, val 40, test 100) under _smoke/')
    ap.add_argument('--force-resume', action='store_true',
                    help='Allow writing into a non-empty output dir')
    args = ap.parse_args()

    if args.only_agent:
        cmd_only_agent(args)
    else:
        cmd_assemble_eval(args)


if __name__ == '__main__':
    main()
