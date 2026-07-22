"""
experiments/run_v2_regime.py
----------------------------
Design-v2 B3 — full Regime-Switching study, staged CLI (same shape as
run_v2_ac.py) on the RegimeJumpEnv with the LOCKED calibration cell.

The cell (σ_high, p₀₁ [, use_rv_feature]) is read from
``results/_v2_regime/locked_cell.json`` — written by the user AFTER the pilot
scan (run_v2_regime_scan.py) recommends a cell. This runner ERRORS clearly if
that file is missing (it must not guess a cell).

Reuses run_v2_ac's proven staged machinery verbatim (train_segment with
byte-identical --episodes a:b resume, param guard, config JSON), swapping in the
regime env + config. ``--assemble-eval`` additionally writes the regime-conditional
breakdown (stress-hit vs calm) — the T-RG-3 table.

NO full runs here (that is B5). ``--smoke`` shrinks everything for logic tests.

Usage
-----
  # (after the user writes results/_v2_regime/locked_cell.json)
  python experiments/run_v2_regime.py --only-agent DQN         --seed 42
  python experiments/run_v2_regime.py --only-agent IQN-neutral --episodes 0:20000 --total-episodes 40000 --seed 42
  python experiments/run_v2_regime.py --assemble-eval --seed 42
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from envs import RegimeJumpEnv, SimConfig
from agents.baselines import (TWAPAgent, AlmgrenChrissAgent, MaxSpeedAgent,
                              DQNAgent, DDQNAgent, DeepRLConfig)
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.param_utils import expected_counts, assert_param_count
from evaluation.metrics import EpisodeTracker, conditional_stats
import run_simulation as RS
import run_v2_ac as AC                       # reuse train machinery + constants
import selection_lib
import evaluation.tables_v2 as tables_v2
from exp_utils import dump_config_json, prepare_v2_output_dir

# ---------------------------------------------------------------------------
# Constants (reuse run_v2_ac; regime-specific base config + locked cell)
# ---------------------------------------------------------------------------

V2_GRID            = AC.V2_GRID
V2_CVAR_ALPHAS     = AC.V2_CVAR_ALPHAS
V2_REPLAY_CAPACITY = AC.V2_REPLAY_CAPACITY
LEARNED_TRAINABLE  = AC.LEARNED_TRAINABLE
MANIFEST_NAME      = AC.MANIFEST_NAME
VAL_SEED_OFFSET    = AC.VAL_SEED_OFFSET
TEST_SEED_OFFSET   = AC.TEST_SEED_OFFSET
DEFAULT_TRAIN      = AC.DEFAULT_TRAIN
SMOKE_TRAIN        = AC.SMOKE_TRAIN

DEFAULT_OUT_ROOT = 'results/_v2_regime'
LOCKED_CELL_PATH = Path(DEFAULT_OUT_ROOT) / 'locked_cell.json'

# Base regime config; σ_high / p₀₁ (and optional use_rv_feature) come from the
# locked cell. RegimeJumpEnv FIXES σ_low=0.0005 and p₁₀=0.40; the stress jump
# params are set here (λ_J=0.1/period, σ_J=0.16, μ_J=0). cfg.sigma is the AC
# baseline vol used only by the AC agent's schedule.
REGIME_BASE_CONFIG = dict(
    N=20, T=60.0, q0=100_000, p0=100.0,
    eta=2.5e-6, gamma=2.5e-7, a=0.001, penalty_type='quadratic',
    sigma=0.00095, discount=0.99,
    action_basis='q0', action_fracs=V2_GRID,
    jump_intensity=0.1, jump_std=0.16, jump_mean=0.0,
)


def load_locked_cell() -> dict:
    """Read the user-locked calibration cell; error clearly if absent."""
    path = PROJECT_ROOT / LOCKED_CELL_PATH
    if not path.exists():
        raise SystemExit(
            f'No locked cell at {path}.\n'
            f'  Run the pilot scan first:\n'
            f'    python experiments/run_v2_regime_scan.py   (then --summarize)\n'
            f'  and, once you pick a cell, write {path} as e.g.\n'
            f'    {{"sigma_high": 0.004, "p_01": 0.10, "use_rv_feature": false}}')
    cell = json.loads(path.read_text())
    for k in ('sigma_high', 'p_01'):
        if k not in cell:
            raise SystemExit(f'locked_cell.json missing required key "{k}".')
    return cell


# Base jump params (the locked-cell / main-study values); the robustness sweep
# overrides ONE of these at a time via CLI. Defaults keep behaviour byte-identical.
BASE_JUMP_STD       = REGIME_BASE_CONFIG['jump_std']        # 0.16
BASE_JUMP_INTENSITY = REGIME_BASE_CONFIG['jump_intensity']  # 0.10


def build_regime_config(cell: dict, jump_std: float = None,
                        jump_intensity: float = None) -> SimConfig:
    cfg = dict(REGIME_BASE_CONFIG)
    cfg['sigma_high']     = float(cell['sigma_high'])
    cfg['p_01']           = float(cell['p_01'])
    cfg['use_rv_feature'] = bool(cell.get('use_rv_feature', False))
    if jump_std is not None:
        cfg['jump_std'] = float(jump_std)
    if jump_intensity is not None:
        cfg['jump_intensity'] = float(jump_intensity)
    return SimConfig(**cfg)


def _effective_jumps(args):
    """Resolve the effective (jump_std, jump_intensity) for this invocation."""
    js = args.jump_std if getattr(args, 'jump_std', None) is not None else BASE_JUMP_STD
    ji = (args.jump_intensity if getattr(args, 'jump_intensity', None) is not None
          else BASE_JUMP_INTENSITY)
    return float(js), float(ji)


def build_regime_agents(sim_config: SimConfig, seed: int, device_str: str):
    """All agents in canonical order + a fresh RegimeJumpEnv (masked learned)."""
    device = torch.device(device_str)
    env = RegimeJumpEnv(sim_config)
    sd, na = env.state_dim, env.n_actions
    grid, basis = sim_config.action_fracs, sim_config.action_basis

    agents = {}
    agents['TWAP']     = TWAPAgent(sim_config)
    agents['AC']       = AlmgrenChrissAgent(sim_config, risk_aversion=1e-6)
    agents['MaxSpeed'] = MaxSpeedAgent(sim_config)
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
            sd, na, device=device, seed=seed + 3, action_fracs=grid, action_basis=basis)

    exp_iqn, exp_mlp = expected_counts(sd, na)
    for name, ag in agents.items():
        if hasattr(ag, 'get_num_params'):
            assert_param_count(ag, exp_iqn if name.startswith('IQN') else exp_mlp, name)
    return agents, env


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

_MANIFEST_MATCH_KEYS = ('sim_config', 'seed', 'total_episodes', 'checkpoint_freq',
                        'n_val', 'n_test', 'device', 'smoke')


def _manifest_payload(sim_config, cell, seed, total, ckpt_freq, n_val, n_test,
                      device, smoke) -> dict:
    return {'env': 'regime_jump', 'cell': cell, 'sim_config': asdict(sim_config),
            'seed': seed, 'total_episodes': total, 'checkpoint_freq': ckpt_freq,
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
        raise SystemExit(f'Staged manifest mismatch in {path} on {diffs}. '
                         f'Archive the dir and start over, or fix the CLI args.')
    return existing


def _resolve_out(args) -> Path:
    if args.out_dir:
        out = Path(args.out_dir)
    else:
        base = Path(DEFAULT_OUT_ROOT) / ('_smoke' if args.smoke else '')
        js, ji = _effective_jumps(args)
        if (js, ji) != (BASE_JUMP_STD, BASE_JUMP_INTENSITY):
            # jump-robustness sweep: namespace by the swept params so the locked-cell
            # (base-jump) results at seed{S}/ are never touched.
            out = base / '_robustness' / f'sJ{js:g}_lJ{ji:g}' / f'seed{args.seed}'
        else:
            out = base / f'seed{args.seed}'
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


# ---------------------------------------------------------------------------
# Train one agent (segment)  — delegates to run_v2_ac.train_segment
# ---------------------------------------------------------------------------

def cmd_only_agent(args):
    name = args.only_agent
    cell = load_locked_cell()
    js, ji = _effective_jumps(args)
    sim_config = build_regime_config(cell, jump_std=js, jump_intensity=ji)
    swept = (js, ji) != (BASE_JUMP_STD, BASE_JUMP_INTENSITY)
    print(f'  [jumps] jump_std={js:g} jump_intensity={ji:g} '
          f'{"(ROBUSTNESS SWEEP)" if swept else "(base/locked)"} | param guard: '
          f'state_dim=5 n_actions=11 -> IQN 11787 / DQN·DDQN 5515')
    tr = SMOKE_TRAIN if args.smoke else DEFAULT_TRAIN
    ckpt_freq = args.checkpoint_freq or tr['checkpoint_freq']
    ep_start, ep_end = (AC._parse_episodes(args.episodes) if args.episodes is not None
                        else (0, tr['episodes']))
    total = args.total_episodes or ep_end
    if ep_end > total:
        raise SystemExit(f'--episodes end {ep_end} > --total-episodes {total}')

    out = _resolve_out(args)
    prepare_v2_output_dir(out, config=None, force_resume=True)
    ckpt_dir, log_dir = out / 'checkpoints', out / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    write_or_check_manifest(log_dir, _manifest_payload(
        sim_config, cell, args.seed, total, ckpt_freq, tr['n_val'], tr['n_test'],
        args.device, args.smoke))

    log_path    = log_dir / f'{name}_training.json'
    resume_path = log_dir / f'{name}_resume.pt'
    if ep_start == 0 and log_path.exists() and not args.force_resume:
        raise SystemExit(f'{name} already trained here ({log_path}). Archive to redo.')

    agents, train_env = build_regime_agents(sim_config, args.seed, args.device)
    print(f'\n### v2-REGIME TRAIN {name} cell(σ_high={cell["sigma_high"]},'
          f'p01={cell["p_01"]}) eps({ep_start}:{ep_end}/{total}) ckpt/{ckpt_freq} '
          f'seed={args.seed} {"[SMOKE]" if args.smoke else ""} -> {out}')
    log = AC.train_segment(agents[name], train_env, name, ep_start, ep_end, total,
                          args.seed, ckpt_dir, ckpt_freq, resume_path)

    with open(log_path, 'w') as f:
        json.dump(log, f)
    dump_config_json(
        {'kind': 'v2_regime_staged_train', 'agent': name, 'cell': cell,
         'sim_config': sim_config, 'seed': args.seed,
         'episodes': f'{ep_start}:{ep_end}', 'total_episodes': total,
         'checkpoint_freq': ckpt_freq, 'device': args.device, 'smoke': args.smoke},
        log_dir / f'staged_{name}_config.json')
    print(f'  {name} segment done in {log["wall_time"]:.1f}s '
          f'(episodes_done={log["episodes_done"]}). log -> {log_path}')


# ---------------------------------------------------------------------------
# Regime-aware evaluation (collects per-episode stress_hit flags)
# ---------------------------------------------------------------------------

def regime_eval(agents, env, n_test, seed):
    """Like RS.evaluate_all but also records each episode's stress_hit flag."""
    all_results, is_dict, flags_dict = [], {}, {}
    for name, agent in agents.items():
        env.seed(seed)
        RS._seed_global_rng(seed)
        tracker = EpisodeTracker()
        flags = []
        for _ in range(n_test):
            s = env.reset()
            if hasattr(agent, 'reset') and callable(agent.reset):
                agent.reset()
            tracker.begin_episode()
            done, info = False, {}
            while not done:
                a = agent.select_action(s, eval_mode=True)
                s, r, done, info = env.step(a)
                tracker.step(r, info)
            tracker.end_episode(info)
            flags.append(bool(info.get('stress_hit', False)))
        res = tracker.compute_metrics()
        res['agent_name'] = name
        all_results.append(res)
        # RAW IS fractions — conditional_stats converts to bps itself (avoid a
        # double ×1e4). is_arrays.pkl multiplies to bps at save time.
        is_dict[name]    = np.array(tracker.is_values)
        flags_dict[name] = np.array(flags, dtype=bool)
    return all_results, is_dict, flags_dict


def cmd_assemble_eval(args):
    out = _resolve_out(args)
    ckpt_dir, log_dir = out / 'checkpoints', out / 'logs'
    if not (log_dir / MANIFEST_NAME).exists():
        raise SystemExit(f'No staged manifest in {log_dir}; train agents first.')
    manifest = json.loads((log_dir / MANIFEST_NAME).read_text())
    if args.device != manifest['device']:
        raise SystemExit(f'--device {args.device} != training device {manifest["device"]}.')

    sim_config = SimConfig(**manifest['sim_config'])
    cell   = manifest['cell']
    seed   = manifest['seed']
    n_val  = manifest['n_val']
    n_test = manifest['n_test']
    val_seed, test_seed = seed + VAL_SEED_OFFSET, seed + TEST_SEED_OFFSET

    agents, eval_env = build_regime_agents(sim_config, seed, args.device)
    canon = list(agents.keys())

    for name in LEARNED_TRAINABLE:
        paths = selection_lib.ckpt_paths_for(ckpt_dir, name)
        if not paths:
            raise SystemExit(f'No checkpoints for {name} in {ckpt_dir}.')
        sel = selection_lib.select_checkpoints(
            agents[name], paths, RegimeJumpEnv(sim_config), val_seed,
            n_val=n_val, alpha=0.95)
        agents[name].load(sel['best_cvar'])
        print(f'  {name}: selected {Path(sel["best_cvar"]).name} '
              f'(best-by-val-CVaR₉₅ of {len(paths)} ckpts)')
    RS.share_iqn_weights(agents)

    print(f'\n### v2-REGIME TEST EVAL {n_test} eps seed={test_seed} '
          f'cell(σ_high={cell["sigma_high"]},p01={cell["p_01"]})')
    all_results, is_dict, flags_dict = regime_eval(agents, eval_env, n_test, test_seed)

    title = (f'v2-Regime comparison (seed {seed}, cell σ_high={cell["sigma_high"]} '
             f'p01={cell["p_01"]}, {n_test} eps){" [SMOKE]" if manifest.get("smoke") else ""}')
    table = tables_v2.write_comparison_table(
        all_results, log_dir / 'comparison_table.txt', order=canon, title=title)
    print('\n' + table)

    # T-RG-3 regime-conditional breakdown (stress-hit vs calm)
    per_agent = {n: conditional_stats(is_dict[n], flags_dict[n]) for n in canon}
    rb = tables_v2.write_regime_breakdown(
        per_agent, log_dir / 'regime_breakdown.txt', order=canon,
        title=f'Stress-hit vs calm (seed {seed}, cell σ_high={cell["sigma_high"]} p01={cell["p_01"]})')
    (log_dir / 'regime_breakdown.tex').write_text(
        tables_v2.regime_breakdown_latex(per_agent, order=canon,
            caption=f'v2-Regime stress-hit vs calm breakdown (seed {seed}).',
            label='tab:v2_regime_breakdown'))
    print('\n' + rb)

    # action-vs-spread heatmap for IQN-neutral (σ̂ on/off decision input)
    try:
        csv_p, png_p = tables_v2.action_spread_heatmap(
            agents['IQN-neutral'], RegimeJumpEnv(sim_config),
            n_episodes=min(n_test, 500), seed=test_seed, out_dir=log_dir)
        print(f'  action-vs-spread heatmap -> {csv_p.name}, {png_p.name}')
    except Exception as e:                       # heatmap is diagnostic, not gating
        print(f'  [warn] heatmap dump failed: {e!r}')

    # α-ladder on the selected IQN-neutral (thesis headline T-RG-2)
    ladder = sorted(set(V2_CVAR_ALPHAS + [1.0]))
    rows = tables_v2.run_alpha_ladder(agents['IQN-neutral'], RegimeJumpEnv(sim_config),
                                      ladder, n_test, test_seed)
    tables_v2.write_alpha_ladder(rows, log_dir,
        caption=f'v2-Regime CVaR-$\\alpha$ ladder (seed {seed}).',
        label='tab:v2_regime_alpha')
    print('\n' + tables_v2.format_alpha_ladder(rows, title='α-ladder (IQN-neutral)'))

    with open(log_dir / 'all_results.json', 'w') as f:
        json.dump([AC._json_safe(r) for r in all_results], f, indent=2)
    with open(log_dir / 'is_arrays.pkl', 'wb') as f:
        pickle.dump({'is': {k: (v * 1e4).tolist() for k, v in is_dict.items()},   # bps
                     'stress_hit': {k: v.tolist() for k, v in flags_dict.items()}}, f)
    dump_config_json(
        {'kind': 'v2_regime_assemble_eval', 'cell': cell, 'seed': seed,
         'n_val': n_val, 'n_test': n_test, 'val_seed': val_seed,
         'test_seed': test_seed, 'alphas': ladder, 'device': args.device},
        log_dir / 'assemble_config.json')
    print(f'\nAssembled -> {log_dir} (comparison_table.txt, regime_breakdown.{{txt,tex}}, '
          f'alpha_ladder.{{txt,tex}}, all_results.json, is_arrays.pkl, heatmap)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Design-v2 B3 staged Regime study CLI')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--only-agent', choices=list(LEARNED_TRAINABLE))
    mode.add_argument('--assemble-eval', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--episodes', default=None, help="'a:b' segment or 'N'")
    ap.add_argument('--total-episodes', type=int, default=None)
    ap.add_argument('--checkpoint-freq', type=int, default=None)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--force-resume', action='store_true')
    ap.add_argument('--jump-std', type=float, default=None,
                    help='override jump_std (default = locked/base 0.16); '
                         'non-default routes output to _robustness/')
    ap.add_argument('--jump-intensity', type=float, default=None,
                    help='override jump_intensity (default = locked/base 0.10); '
                         'non-default routes output to _robustness/')
    args = ap.parse_args()

    if args.only_agent:
        cmd_only_agent(args)
    else:
        cmd_assemble_eval(args)


if __name__ == '__main__':
    main()
