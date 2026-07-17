"""
experiments/run_jd_staged.py
----------------------------
STAGED (one-agent-per-job) jump-diffusion retrain, so the ~30k-episode monolithic
JD run splits into sub-34-min jobs (the harness reaps long background jobs at
~34 min). It reuses run_simulation's build_agents / train_agent / evaluate_all /
share_iqn_weights / save_* helpers verbatim, so the assembled result is IDENTICAL
in format and value to the monolithic `run_phase` path.

Why this is exact: agents consume the GLOBAL torch/np RNG (ε-greedy, replay
sampling, IQN τ). run_simulation.train_agent and evaluate_all now reseed that RNG
per agent (order-independent), so each agent's train+eval depends only on its own
seed — independent of anything trained/evaluated earlier in the process. Trained
weights cross job boundaries only via checkpoints, so --checkpoint-freq must
divide --episodes (the best-by-val-CVaR95 checkpoint must exist on disk).

Modes
-----
  --only-agent {DQN,DDQN,IQN-neutral}
      Build ALL agents (identical construction), train ONLY this one into the
      shared out-dir (checkpoints/ + logs/<name>_training.json). Tolerant of the
      other agents' existing files; refuses to overwrite its OWN; verifies the
      shared run manifest matches (same config/seed/schedule/device).

  --assemble-eval [--eval-agents NAME ...]
      Build all agents, restore each learned agent's best-by-val-CVaR95
      checkpoint, share IQN-neutral weights to the CVaR variants, evaluate the
      requested subset (default: all 7), and write per-agent partials. Once all
      7 agents' partials exist, assemble comparison_table.txt + all_results.json
      + is_arrays.pkl in canonical order — byte-identical to run_phase. Use
      --eval-agents to split the eval across two jobs if one risks >30 min.

Usage
-----
  # train one agent per job (each its own kept-awake sub-34-min job)
  python experiments/run_jd_staged.py --only-agent DQN         --device mps
  python experiments/run_jd_staged.py --only-agent DDQN        --device mps
  python experiments/run_jd_staged.py --only-agent IQN-neutral --device mps
  # then evaluate + assemble (optionally split with --eval-agents)
  python experiments/run_jd_staged.py --assemble-eval --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # experiments/ siblings

import numpy as np

from envs import JumpDiffusionEnv, SimConfig
import run_simulation as RS
from exp_utils import dump_config_json

LEARNED_TRAINABLE = ('DQN', 'DDQN', 'QR-DQN', 'IQN-neutral')
DEFAULT_OUT = 'results/_seeds/seed42/jump_diffusion'
MANIFEST_NAME = 'staged_manifest.json'


# ---------------------------------------------------------------------------
# Config + manifest
# ---------------------------------------------------------------------------

def build_sim_config(args) -> SimConfig:
    """SimConfig from DEFAULT_SIM_CONFIG with optional jump overrides (fallback)."""
    cfg_dict = dict(RS.DEFAULT_SIM_CONFIG)
    if args.jump_intensity is not None:
        cfg_dict['jump_intensity'] = args.jump_intensity
    if args.jump_std is not None:
        cfg_dict['jump_std'] = args.jump_std
    if args.jump_mean is not None:
        cfg_dict['jump_mean'] = args.jump_mean
    return SimConfig(**cfg_dict)


def _manifest_payload(sim_config, args) -> dict:
    return {
        'sim_config': asdict(sim_config),
        'seed': args.seed,
        'episodes': args.episodes,
        'eval_freq': args.eval_freq,
        'checkpoint_freq': args.checkpoint_freq,
        'eval_episodes': args.eval_episodes,
        'device': args.device,
    }


# Keys that MUST match across all staged jobs sharing a dir (device included:
# CPU vs MPS can differ in the last FP bits, which would break bit-equivalence).
_MANIFEST_MATCH_KEYS = ('sim_config', 'seed', 'episodes', 'eval_freq',
                        'checkpoint_freq', 'eval_episodes', 'device')


def write_or_check_manifest(log_dir: Path, payload: dict) -> dict:
    """Write the manifest on first job; on later jobs refuse on any mismatch."""
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
            f'  existing={{k: existing.get(k) for k in diffs}} vs '
            f'new={{k: payload.get(k) for k in diffs}}\n'
            f'  All staged jobs for one dir must share config/seed/schedule/device.\n'
            f'  Archive this dir and start over, or fix the CLI args.')
    return existing


_SELECT_KEY = {'cvar': 'CVaR_0.95_bps', 'mean': 'mean_IS_bps'}


def best_ep_from_log(log_path: Path, select_by: str = 'cvar'):
    """Best-by-validation epoch: min CVaR_0.95_bps (cvar) or min mean_IS_bps (mean).

    'cvar' mirrors train_agent's restore (the default everywhere). 'mean' is the
    D1 diagnostic: pick the neutral checkpoint by lowest validation mean IS.
    """
    log = json.loads(Path(log_path).read_text())
    hist = log.get('eval_history', [])
    if not hist:
        return None
    key = _SELECT_KEY[select_by]
    best = min(hist, key=lambda x: x.get(key, float('inf')))
    return best['episode']


# ---------------------------------------------------------------------------
# Agent set
# ---------------------------------------------------------------------------

def build_all(sim_config, args):
    """Build the full agent dict + a JD train/eval env pair (identical to run_phase)."""
    train_env = JumpDiffusionEnv(sim_config)
    eval_env = JumpDiffusionEnv(sim_config)
    sd, na = train_env.state_dim, train_env.n_actions
    agents = RS.build_agents(sim_config, sd, na, seed=args.seed, device_str=args.device)
    return agents, train_env, eval_env


def canonical_order(agents):
    return list(agents.keys())


def learned_names(agents):
    return [n for n in agents if n in LEARNED_TRAINABLE]


# ---------------------------------------------------------------------------
# Mode: train one agent
# ---------------------------------------------------------------------------

def cmd_only_agent(args):
    name = args.only_agent
    sim_config = build_sim_config(args)
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    ckpt_dir = out / 'checkpoints'
    log_dir = out / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    write_or_check_manifest(log_dir, _manifest_payload(sim_config, args))

    log_path = log_dir / f'{name}_training.json'
    if log_path.exists():
        raise SystemExit(
            f'{name} already trained here ({log_path} exists). Refusing to '
            f'overwrite. Archive the dir to retrain from scratch.')

    agents, train_env, eval_env = build_all(sim_config, args)
    if name not in learned_names(agents):
        raise SystemExit(f'--only-agent must be one of {learned_names(agents)}')

    print(f'\n### STAGED TRAIN {name}  (JD λ={sim_config.jump_intensity} '
          f'σ={sim_config.jump_std} μ={sim_config.jump_mean}) -> {out}')
    print(f'    episodes={args.episodes} eval_freq={args.eval_freq} '
          f'ckpt_freq={args.checkpoint_freq} seed={args.seed} device={args.device}')

    log = RS.train_agent(
        agent=agents[name], env=train_env, n_episodes=args.episodes,
        eval_env=eval_env, eval_freq=args.eval_freq,
        n_eval=min(args.eval_episodes, 200),          # matches run_phase
        checkpoint_dir=ckpt_dir, checkpoint_freq=args.checkpoint_freq,
        seed=args.seed)

    RS.save_training_log(log, log_path)
    dump_config_json(
        {'kind': 'jd_staged_train', 'agent': name, 'sim_config': sim_config,
         'seed': args.seed, 'episodes': args.episodes,
         'eval_freq': args.eval_freq, 'checkpoint_freq': args.checkpoint_freq,
         'eval_episodes': args.eval_episodes, 'device': args.device},
        log_dir / f'staged_{name}_config.json')

    best_ep = best_ep_from_log(log_path)
    print(f'  {name} done in {log["wall_time"]:.0f}s. best-by-val-CVaR95 ep={best_ep}. '
          f'Log -> {log_path}')


# ---------------------------------------------------------------------------
# Mode: assemble eval
# ---------------------------------------------------------------------------

def _partial_dir(log_dir: Path, tag: str = '') -> Path:
    d = log_dir / f'_eval_partial{tag}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json_safe(result: dict) -> dict:
    s = {}
    for k, v in result.items():
        if isinstance(v, (np.floating, np.integer)):
            s[k] = float(v)
        elif isinstance(v, np.ndarray):
            s[k] = v.tolist()
        else:
            s[k] = v
    return s


def cmd_assemble_eval(args):
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    ckpt_dir = out / 'checkpoints'
    log_dir = out / 'logs'
    if not (log_dir / MANIFEST_NAME).exists():
        raise SystemExit(f'No staged manifest in {log_dir}; train agents first.')
    manifest = json.loads((log_dir / MANIFEST_NAME).read_text())

    # The manifest is the source of truth for what was trained. Rebuild the SAME
    # sim_config / seed / eval_episodes the agents were trained with (training
    # schedule args like --episodes are irrelevant to eval and are ignored).
    sim_config = SimConfig(**manifest['sim_config'])
    seed = manifest['seed']
    eval_episodes = manifest['eval_episodes']
    if args.device != manifest['device']:
        raise SystemExit(
            f'--device {args.device} != training device {manifest["device"]}. '
            f'Evaluate on the same device the agents were trained on '
            f'(CPU vs MPS differ in the last FP bits).')

    args.seed = seed          # construction seed from manifest (agents get loaded over)
    agents, _train_env, eval_env = build_all(sim_config, args)
    canon = canonical_order(agents)

    # Checkpoint selection. IQN-neutral honours --select-by (cvar|mean); DQN/DDQN
    # always use best-by-val-CVaR95 (D1 changes ONLY the shared neutral weights).
    select_by = args.select_by
    tag = '' if select_by == 'cvar' else f'_{select_by}'
    for name in learned_names(agents):
        sb = select_by if name == 'IQN-neutral' else 'cvar'
        lp = log_dir / f'{name}_training.json'
        if not lp.exists():
            raise SystemExit(f'Missing training log for {name}: {lp}. '
                             f'Train all learned agents before --assemble-eval.')
        best_ep = best_ep_from_log(lp, select_by=sb)
        ckpt = ckpt_dir / f'{name}_ep{best_ep}.pt'
        if not ckpt.exists():
            raise SystemExit(
                f'Best checkpoint for {name} not found: {ckpt} (best ep={best_ep}). '
                f'Ensure --checkpoint-freq divides --episodes so it was saved.')
        agents[name].load(str(ckpt))
        print(f'  restored {name}: best-by-val-{sb} ep={best_ep}')

    # Share IQN-neutral (best) weights to the CVaR variants (D-key design).
    RS.share_iqn_weights(agents)

    # Which agents to evaluate this job.
    subset = args.eval_agents if args.eval_agents else canon
    bad = [n for n in subset if n not in agents]
    if bad:
        raise SystemExit(f'--eval-agents unknown: {bad}. Choose from {canon}')
    sub_agents = {n: agents[n] for n in canon if n in subset}   # canonical order

    print(f'\n### ASSEMBLE-EVAL (select-by={select_by}) evaluating {list(sub_agents)} '
          f'({eval_episodes} eps, seed={seed + 99_999})')
    all_results, trackers = RS.evaluate_all(
        sub_agents, eval_env, n_eval=eval_episodes, seed=seed + 99_999)

    # Write per-agent partials (tagged so cvar/mean selections don't collide).
    pdir = _partial_dir(log_dir, tag)
    for r in all_results:
        name = r['agent_name']
        is_bps = (np.array(trackers[name].is_values) * 1e4).tolist()
        with open(pdir / f'{name}.json', 'w') as f:
            json.dump({'result': _json_safe(r), 'is_bps': is_bps}, f)
        print(f'  wrote partial: {name}')

    # Assemble if every canonical agent now has a partial.
    have = {p.stem for p in pdir.glob('*.json')}
    missing = [n for n in canon if n not in have]
    if missing:
        print(f'\nPARTIAL: {len(have)}/{len(canon)} agents evaluated. '
              f'Still missing: {missing}\n'
              f'  Run --assemble-eval --select-by {select_by} '
              f'--eval-agents {" ".join(missing)} to finish.')
        return

    all_results_full, is_dict = [], {}
    for name in canon:
        payload = json.loads((pdir / f'{name}.json').read_text())
        all_results_full.append(payload['result'])
        is_dict[name] = np.array(payload['is_bps'])
    table = RS.save_eval_outputs(all_results_full, is_dict, log_dir, suffix=tag)
    print('\n' + table)
    print(f'\nAssembled -> {log_dir}/comparison_table{tag}.txt '
          f'(+ all_results{tag}.json, is_arrays{tag}.pkl)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Staged (one-agent-per-job) JD retrain')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--only-agent', choices=['DQN', 'DDQN', 'IQN-neutral'],
                      help='Train exactly this one agent into the shared out-dir')
    mode.add_argument('--assemble-eval', action='store_true',
                      help='Restore best checkpoints, share, evaluate, assemble the table')

    ap.add_argument('--out-dir', default=DEFAULT_OUT,
                    help='Shared phase dir (contains checkpoints/ and logs/)')
    ap.add_argument('--eval-agents', nargs='+', default=None,
                    help='(assemble-eval) evaluate only this subset this job')
    ap.add_argument('--select-by', choices=['cvar', 'mean'], default='cvar',
                    help='(assemble-eval) IQN-neutral checkpoint selection: '
                         'cvar=min val CVaR95 (default, = train_agent restore), '
                         'mean=min val mean IS (D1). DQN/DDQN always cvar. '
                         'mean writes *_mean.* outputs so cvar results are kept.')
    ap.add_argument('--episodes', type=int, default=RS.DEFAULT_TRAIN['n_episodes'])
    ap.add_argument('--eval-episodes', type=int, default=RS.DEFAULT_TRAIN['n_eval_episodes'])
    ap.add_argument('--eval-freq', type=int, default=RS.DEFAULT_TRAIN['eval_freq'])
    ap.add_argument('--checkpoint-freq', type=int, default=RS.DEFAULT_TRAIN['checkpoint_freq'])
    ap.add_argument('--seed', type=int, default=RS.DEFAULT_TRAIN['seed'])
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    # Optional jump overrides (fallback calibration; μ_J stays 0 unless set).
    ap.add_argument('--jump-intensity', type=float, default=None)
    ap.add_argument('--jump-std', type=float, default=None)
    ap.add_argument('--jump-mean', type=float, default=None)
    args = ap.parse_args()

    if args.only_agent:
        cmd_only_agent(args)
    else:
        cmd_assemble_eval(args)


if __name__ == '__main__':
    main()
