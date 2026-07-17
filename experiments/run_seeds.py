"""
experiments/run_seeds.py
------------------------
R1 multi-seed orchestration (Phase-3 batch script — TRAINS models; do not run
during code review).

  Sim : {ac, jump} × seeds  → results/_seeds/seed<seed>/<env_name>/   (run_phase)
  TAQ : DQN/DDQN  × seeds    → results/taq/AAPL_seed<seed>/
        IQN is NOT retrained — the kept results/taq/AAPL IQN-neutral checkpoint
        is reused and shared to the CVaR variants (D2).

Every run dumps its full config as JSON; existing non-empty output dirs are
refused (R-2) so prior results are never overwritten.

NOTE on layout: PLAN.md sketched `results/<env>_seed<seed>/`; to reuse
run_simulation.run_phase unchanged (it selects the env from a fixed env_name
and writes results_dir/<env_name>/), the sim layout is
`results/_seeds/seed<seed>/<env_name>/`. aggregate_seeds.py globs this.

Usage:
    python experiments/run_seeds.py --sim --seeds 42 123 7 2024 31 --device mps
    python experiments/run_seeds.py --taq --seeds 42 123 7 2024 31 --device mps
    python experiments/run_seeds.py --sim --envs ac jump --seeds 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # experiments/ siblings

import numpy as np
import pandas as pd

from envs import SimConfig
import run_simulation as RS
import run_tag as RT
from exp_utils import dump_config_json, refuse_if_nonempty

DEFAULT_SEEDS = [42, 123, 7, 2024, 31]
SIM_ENV_NAMES = {'ac': 'almgren_chriss', 'ou': 'mean_reverting', 'jump': 'jump_diffusion'}


# ---------------------------------------------------------------------------
# Simulated envs (AC / MeanReverting / Jump) via run_phase
# ---------------------------------------------------------------------------

def run_sim_seeds(env_shorts, seeds, episodes, eval_episodes, device, base,
                  jump_overrides=None):
    base = Path(base)
    for seed in seeds:
        for short in env_shorts:
            env_name = SIM_ENV_NAMES[short]
            out_root = base / f'seed{seed}'
            refuse_if_nonempty(out_root / env_name)
            cfg_dict = dict(RS.DEFAULT_SIM_CONFIG)
            if jump_overrides:                 # e.g. the scan-selected (λ_J, σ_J)
                cfg_dict.update(jump_overrides)
            sim_config = SimConfig(**cfg_dict)
            print(f'\n### SIM {env_name} seed={seed} -> {out_root}/{env_name}')
            RS.run_phase(
                env_name=env_name, sim_config=sim_config,
                n_episodes=episodes, n_eval=eval_episodes,
                eval_freq=RS.DEFAULT_TRAIN['eval_freq'],
                checkpoint_freq=RS.DEFAULT_TRAIN['checkpoint_freq'],
                seed=seed, results_dir=out_root, device=device,
            )
            dump_config_json(
                {'kind': 'sim', 'env': env_name, 'seed': seed,
                 'episodes': episodes, 'eval_episodes': eval_episodes,
                 'device': device, 'sim_config': sim_config},
                out_root / env_name / 'logs' / 'run_config.json')


# ---------------------------------------------------------------------------
# TAQ: retrain ONLY DQN/DDQN per seed; reuse the kept IQN checkpoint (D2)
# ---------------------------------------------------------------------------

def run_taq_dqn_ddqn_seeds(seeds, episodes, test_eval, device, kept_iqn_ckpt, base):
    base = Path(base)
    kept_iqn_ckpt = Path(kept_iqn_ckpt)
    if not kept_iqn_ckpt.exists():
        raise FileNotFoundError(
            f'Kept IQN checkpoint not found: {kept_iqn_ckpt}\n'
            f'  D2 requires reusing the existing TAQ IQN-neutral checkpoint; '
            f'pass --iqn-ckpt to point at it.')

    RT.DEVICE = device                        # build_all_agents reads this global
    cfg = RT.build_taq_config()
    parquet = Path(cfg.data_dir) / f'{cfg.stock}_{cfg.year}.parquet'
    all_dates = sorted(pd.read_parquet(parquet, columns=['date'])['date'].unique().tolist())
    train_months, val_months, test_months = RT.FOLDS[0]
    train_dates = RT.get_dates_for_months(all_dates, train_months)
    val_dates   = RT.get_dates_for_months(all_dates, val_months)
    test_dates  = RT.get_dates_for_months(all_dates, test_months)

    for seed in seeds:
        out = base / f'AAPL_seed{seed}'
        refuse_if_nonempty(out)
        print(f'\n### TAQ DQN/DDQN seed={seed} -> {out}  (IQN reused, D2)')

        train_env = RT.TAQEnv(cfg, train_dates)
        val_env   = RT.TAQEnv(cfg, val_dates)
        test_env  = RT.TAQEnv(cfg, test_dates)
        sd, na = train_env.state_dim, train_env.n_actions
        agents = RT.build_all_agents(sd, na, seed)   # builds all incl IQN (param-guarded)

        ckpt_dir = out / 'fold1' / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for name in ['DQN', 'DDQN']:
            print(f'  -- training {name} (seed {seed}) --')
            RT.train_with_validation(agents[name], train_env, val_env,
                                     episodes, ckpt_dir, seed)

        # D2: reuse kept IQN-neutral weights, then share to CVaR variants
        agents['IQN-neutral'].load(str(kept_iqn_ckpt))
        RT.share_iqn_weights(agents)

        all_results, trackers = RT.evaluate_all(agents, test_env, test_eval, seed + 99_999)
        _save_taq(out, all_results, trackers)
        dump_config_json(
            {'kind': 'taq', 'seed': seed, 'episodes': episodes,
             'test_eval': test_eval, 'device': device,
             'kept_iqn_ckpt': str(kept_iqn_ckpt),
             'retrained': ['DQN', 'DDQN'], 'iqn_source': 'kept (D2)'},
            out / 'run_config.json')


def _save_taq(out, all_results, trackers):
    logs = Path(out) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    serial = []
    for r in all_results:
        serial.append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in r.items()})
    with open(logs / 'all_results.json', 'w') as f:      # same *_bps schema as sim
        json.dump(serial, f, indent=2)
    is_dict = {name: (np.array(t.is_values) * 1e4).tolist()
               for name, t in trackers.items()}
    with open(logs / 'is_arrays.json', 'w') as f:
        json.dump(is_dict, f)


def main():
    ap = argparse.ArgumentParser(description='R1 multi-seed orchestration')
    ap.add_argument('--sim', action='store_true', help='Run the simulated envs')
    ap.add_argument('--taq', action='store_true', help='Run TAQ DQN/DDQN (IQN kept)')
    ap.add_argument('--envs', nargs='+', default=['ac', 'jump'],
                    choices=list(SIM_ENV_NAMES.keys()))
    ap.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--episodes', type=int, default=None,
                    help='Training episodes (sim default 30000, taq default 50000)')
    ap.add_argument('--eval-episodes', type=int, default=10000)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--sim-base', default='results/_seeds')
    ap.add_argument('--taq-base', default='results/taq')
    ap.add_argument('--iqn-ckpt',
                    default='results/taq/AAPL/fold1/checkpoints/IQN-neutral_best.pt',
                    help='Kept TAQ IQN-neutral checkpoint to reuse (D2)')
    # Optional jump-param overrides (e.g. the scan-selected calibration) — sim only.
    ap.add_argument('--jump-intensity', type=float, default=None, help='Override λ_J (sim)')
    ap.add_argument('--jump-std', type=float, default=None, help='Override σ_J (sim)')
    ap.add_argument('--jump-mean', type=float, default=None, help='Override μ_J (sim; default 0.0)')
    args = ap.parse_args()

    if not (args.sim or args.taq):
        ap.error('specify --sim and/or --taq')

    jump_overrides = {}
    if args.jump_intensity is not None:
        jump_overrides['jump_intensity'] = args.jump_intensity
    if args.jump_std is not None:
        jump_overrides['jump_std'] = args.jump_std
    if args.jump_mean is not None:
        jump_overrides['jump_mean'] = args.jump_mean

    if args.sim:
        ep = args.episodes or RS.DEFAULT_TRAIN['n_episodes']
        run_sim_seeds(args.envs, args.seeds, ep, args.eval_episodes,
                      args.device, PROJECT_ROOT / args.sim_base,
                      jump_overrides=jump_overrides or None)
    if args.taq:
        ep = args.episodes or RT.N_EPISODES
        run_taq_dqn_ddqn_seeds(args.seeds, ep, args.eval_episodes, args.device,
                               PROJECT_ROOT / args.iqn_ckpt,
                               PROJECT_ROOT / args.taq_base)


if __name__ == '__main__':
    main()
