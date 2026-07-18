"""
experiments/run_il_baseline.py  (Batch E1)
------------------------------------------
Evaluate the Immediate-Liquidation (IL) baseline — sell 100% at t=0 — in all
three settings, EVAL-ONLY, using the SAME 10k-episode protocol and eval seeds as
the stored main tables, then write IL's row + an IL-augmented comparison table
into a NEW dir (results/_il_baseline/), never overwriting existing results.

  sim (AC, JD-locked): eval seed = 42 + 99_999 (matches run_phase's final eval);
                       augmented from results/_seeds/seed42/<env>/logs/all_results.json
  TAQ (kept setup)   : fold-1 test env, eval seed = 42 + 99_999 (matches run_seeds);
                       augmented from results/taq/AAPL_seed42/logs/all_results.json

Usage:
    python experiments/run_il_baseline.py --setting ac
    python experiments/run_il_baseline.py --setting jd
    python experiments/run_il_baseline.py --setting taq
    python experiments/run_il_baseline.py --setting all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from envs import AlmgrenChrissEnv, JumpDiffusionEnv, SimConfig
from envs.base_env import ACTION_FRACS
from agents.baselines import ImmediateLiquidationAgent
from evaluation.metrics import format_comparison_table
from exp_utils import dump_config_json
import run_simulation as RS

IL_NAME = 'Immediate-Liquidation'
EVAL_SEED = RS.DEFAULT_TRAIN['seed'] + 99_999   # 100041, matches the main tables
N_EVAL = RS.DEFAULT_TRAIN['n_eval_episodes']    # 10000
OUT = PROJECT_ROOT / 'results' / '_il_baseline'


def _json_safe(r: dict) -> dict:
    o = {}
    for k, v in r.items():
        if isinstance(v, (np.floating, np.integer)):
            o[k] = float(v)
        elif isinstance(v, np.ndarray):
            o[k] = v.tolist()
        else:
            o[k] = v
    return o


def _augment_and_write(tag, il_result, il_tracker, existing_all_results_path, extra_cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    dump = 1.0   # IL always selects the full-liquidation action at t0 (deterministic)
    is_bps = (np.array(il_tracker.is_values) * 1e4).tolist() if il_tracker is not None else None

    with open(OUT / f'{tag}_il.json', 'w') as f:
        json.dump({'result': _json_safe(il_result), 'dump_fraction': dump,
                   'is_bps': is_bps}, f, indent=2)

    existing = json.load(open(existing_all_results_path))
    # Insert IL right after TWAP (the two naive-schedule extremes).
    names = [r['agent_name'] for r in existing]
    pos = names.index('TWAP') + 1 if 'TWAP' in names else 0
    augmented = existing[:pos] + [_json_safe(il_result)] + existing[pos:]
    with open(OUT / f'{tag}_all_results_with_il.json', 'w') as f:
        json.dump(augmented, f, indent=2)
    table = format_comparison_table(augmented, bps=True)
    (OUT / f'{tag}_with_il_comparison_table.txt').write_text(table)
    dump_config_json({'kind': 'il_baseline', 'setting': tag, 'eval_seed': EVAL_SEED,
                      'n_eval': N_EVAL, 'dump_fraction': dump, **extra_cfg},
                     OUT / f'{tag}_config.json')
    print(f'\n[{tag}] IL: mean={il_result["mean_IS_bps"]:.4f} '
          f'CVaR95={il_result["CVaR_0.95_bps"]:.4f} Max={il_result["max_IS_bps"]:.4f} '
          f'Std={il_result["std_IS_bps"]:.4f} dump={dump:.2f}')
    print(table)
    print(f'  -> {OUT}/{tag}_*.{{json,txt}}')


def run_sim(env_name):
    cfg = SimConfig(**RS.DEFAULT_SIM_CONFIG)
    env = (AlmgrenChrissEnv(cfg) if env_name == 'almgren_chriss'
           else JumpDiffusionEnv(cfg))
    il = ImmediateLiquidationAgent(cfg)
    results, trackers = RS.evaluate_all({IL_NAME: il}, env, n_eval=N_EVAL, seed=EVAL_SEED)
    existing = PROJECT_ROOT / 'results' / '_seeds' / 'seed42' / env_name / 'logs' / 'all_results.json'
    _augment_and_write(env_name, results[0], trackers[IL_NAME], existing,
                       {'env': env_name, 'sim_config': cfg.__dict__.copy()})


def run_taq():
    import pandas as pd
    import run_tag as RT
    RT.DEVICE = 'cpu'
    cfg = RT.build_taq_config()
    parquet = Path(cfg.data_dir) / f'{cfg.stock}_{cfg.year}.parquet'
    all_dates = sorted(pd.read_parquet(parquet, columns=['date'])['date'].unique().tolist())
    _, _, test_months = RT.FOLDS[0]
    test_dates = RT.get_dates_for_months(all_dates, test_months)
    test_env = RT.TAQEnv(cfg, test_dates)
    il = ImmediateLiquidationAgent(cfg)
    results, trackers = RT.evaluate_all({IL_NAME: il}, test_env, N_EVAL, EVAL_SEED)
    il_result = results[0] if isinstance(results, list) else results
    tr = trackers[IL_NAME] if isinstance(trackers, dict) else None
    existing = PROJECT_ROOT / 'results' / 'taq' / 'AAPL_seed42' / 'logs' / 'all_results.json'
    _augment_and_write('taq_AAPL', il_result, tr, existing,
                       {'env': 'taq', 'stock': cfg.stock, 'fold': 1, 'test_months': test_months})


def main():
    ap = argparse.ArgumentParser(description='E1 Immediate-Liquidation baseline (eval-only)')
    ap.add_argument('--setting', choices=['ac', 'jd', 'taq', 'all'], required=True)
    args = ap.parse_args()
    if args.setting in ('ac', 'all'):
        run_sim('almgren_chriss')
    if args.setting in ('jd', 'all'):
        run_sim('jump_diffusion')
    if args.setting in ('taq', 'all'):
        run_taq()


if __name__ == '__main__':
    main()
