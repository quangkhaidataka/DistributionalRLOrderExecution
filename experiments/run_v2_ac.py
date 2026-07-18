"""
experiments/run_v2_ac.py
------------------------
Design-v2 B2 — staged CLI for the Almgren-Chriss (Gaussian sanity) study under
the v2 setting (N=20, dt=3', q0-grid with cap 0.25, feasible-action masking).

Modeled on ``run_jd_staged.py`` but with three v2 additions:
  1. The v2 config (AC env, N=20, q0-grid, replay 100k) — PLAN_V2 §B2 constants.
  2. Learned agents built WITH masking (action_basis='q0'); + MaxSpeed baseline;
     TWAP/AC execute continuous share amounts.
  3. ``--episodes a:b`` episode-range resume that is BYTE-IDENTICAL to a single
     run: it persists the agent (weights+optimizer+step), the global torch/np
     RNG, the replay buffer, AND the env RNG, so ``0:200`` + ``200:400`` == a
     single ``0:400`` run (the resume-split equivalence tested in B2.3).

All reads/writes stay under ``results/_v2_ac/``; a non-empty output dir is refused
unless ``--force-resume``. Every job dumps its config JSON. Checkpoint selection
at ``--assemble-eval`` uses the 1,200-episode CRN validation set (selection_lib).

NO full runs here (that is B5). ``--smoke`` shrinks everything for logic tests.

Usage
-----
  # one learned agent per staged job (IQN split into 2×20k)
  python experiments/run_v2_ac.py --only-agent DQN         --seed 42
  python experiments/run_v2_ac.py --only-agent DDQN        --seed 42
  python experiments/run_v2_ac.py --only-agent IQN-neutral --episodes 0:20000 --total-episodes 40000 --seed 42
  python experiments/run_v2_ac.py --only-agent IQN-neutral --episodes 20000:40000 --total-episodes 40000 --seed 42
  # select (1,200-ep CRN) + test 10k for all agents + write tables
  python experiments/run_v2_ac.py --assemble-eval --seed 42
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # experiments/ siblings

import numpy as np
import torch

from envs import AlmgrenChrissEnv, SimConfig
from agents.baselines import (TWAPAgent, AlmgrenChrissAgent, MaxSpeedAgent,
                              DQNAgent, DDQNAgent, DeepRLConfig)
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.param_utils import expected_counts, assert_param_count
import run_simulation as RS
import selection_lib
import evaluation.tables_v2 as tables_v2
from exp_utils import dump_config_json, prepare_v2_output_dir

# ---------------------------------------------------------------------------
# v2 design constants (PLAN_V2 §B2 / §Design constants)
# ---------------------------------------------------------------------------

V2_GRID = [round(0.025 * i, 3) for i in range(11)]          # 0 … 0.25, cap 0.25
V2_CVAR_ALPHAS = [0.30, 0.50, 0.70, 0.90, 0.95]             # eval-only α-ladder
V2_REPLAY_CAPACITY = 100_000

V2_SIM_CONFIG = dict(
    N=20, T=60.0, q0=100_000, p0=100.0,
    eta=2.5e-6, gamma=2.5e-7, a=0.001, penalty_type='quadratic',
    sigma=0.00095, discount=0.99,
    action_basis='q0', action_fracs=V2_GRID, use_rv_feature=False,
)

DEFAULT_TRAIN = dict(
    episodes=40_000, checkpoint_freq=2_000,
    n_val=1_200, n_test=10_000,
)
SMOKE_TRAIN = dict(
    episodes=200, checkpoint_freq=100,
    n_val=40, n_test=100,
)

LEARNED_TRAINABLE = ('DQN', 'DDQN', 'IQN-neutral')
_V2_SEED_OFFSET = {'DQN': 0, 'DDQN': 1, 'IQN-neutral': 3}
MANIFEST_NAME = 'staged_manifest.json'
DEFAULT_OUT_ROOT = 'results/_v2_ac'
VAL_SEED_OFFSET  = 54_321       # CRN validation seed = seed + this (≠ test seed)
TEST_SEED_OFFSET = 99_999       # test seed = seed + this (matches v1 convention)


# ---------------------------------------------------------------------------
# Config + agents
# ---------------------------------------------------------------------------

def build_sim_config() -> SimConfig:
    return SimConfig(**V2_SIM_CONFIG)


def _seed_global_rng(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_v2_agents(sim_config: SimConfig, seed: int, device_str: str):
    """All v2 agents in canonical order + a fresh AC env.

    Learned agents get action_fracs/action_basis so they mask (q0 mode); the
    rule-based TWAP/AC/MaxSpeed read the grid from the config. Replay 100k.
    """
    device = torch.device(device_str)
    env = AlmgrenChrissEnv(sim_config)
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

_MANIFEST_MATCH_KEYS = ('sim_config', 'seed', 'total_episodes', 'checkpoint_freq',
                        'n_val', 'n_test', 'device', 'smoke')


def _manifest_payload(sim_config, seed, total_episodes, ckpt_freq, n_val, n_test,
                      device, smoke) -> dict:
    return {
        'env': 'almgren_chriss',
        'sim_config': asdict(sim_config),
        'seed': seed,
        'total_episodes': total_episodes,
        'checkpoint_freq': ckpt_freq,
        'n_val': n_val,
        'n_test': n_test,
        'device': device,
        'smoke': smoke,
    }


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
            f'  All staged jobs for one dir must share config/seed/schedule/device.\n'
            f'  Archive the dir and start over, or fix the CLI args.')
    return existing


# ---------------------------------------------------------------------------
# Full-state resume  (byte-identical episode-range split)
# ---------------------------------------------------------------------------

def _save_full_state(path: Path, agent, train_env, log: dict, ep_done: int) -> None:
    """Persist EVERYTHING that determines future training so the next segment is
    byte-identical to a single run: agent weights+optimizer+step, the global
    torch/np RNG, the whole replay buffer, and the env RNG.
    """
    rb = agent.replay
    state = {
        'online_net': agent.online_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer':  agent.optimizer.state_dict(),
        'step':       agent._step,
        'torch_rng':  torch.get_rng_state(),
        'np_rng':     np.random.get_state(),
        'replay':     {'ptr': rb._ptr, 'size': rb._size,
                       'states': rb.states, 'actions': rb.actions,
                       'rewards': rb.rewards, 'next_states': rb.next_states,
                       'dones': rb.dones},
        'env_rng':    train_env._rng.bit_generator.state,
        'log':        log,
        'ep_done':    ep_done,
    }
    torch.save(state, str(path))


def _load_full_state(path: Path, agent, train_env):
    state = torch.load(str(path), map_location='cpu')
    agent.online_net.load_state_dict(state['online_net'])
    agent.target_net.load_state_dict(state['target_net'])
    agent.optimizer.load_state_dict(state['optimizer'])
    agent._step = state['step']
    torch.set_rng_state(state['torch_rng'])
    np.random.set_state(state['np_rng'])
    rb = agent.replay
    rb._ptr, rb._size = state['replay']['ptr'], state['replay']['size']
    rb.states[:]      = state['replay']['states']
    rb.actions[:]     = state['replay']['actions']
    rb.rewards[:]     = state['replay']['rewards']
    rb.next_states[:] = state['replay']['next_states']
    rb.dones[:]       = state['replay']['dones']
    train_env._rng.bit_generator.state = state['env_rng']
    return state['log'], state['ep_done']


# ---------------------------------------------------------------------------
# Training (one agent, an episode segment)
# ---------------------------------------------------------------------------

def _parse_episodes(spec: str):
    """'a:b' -> (a, b) ; 'N' -> (0, N)."""
    if ':' in spec:
        a, b = spec.split(':')
        return int(a), int(b)
    return 0, int(spec)


def train_segment(agent, train_env, name, ep_start, ep_end, total_episodes,
                  seed, ckpt_dir: Path, ckpt_freq: int, resume_path: Path) -> dict:
    """Train episodes (ep_start, ep_end]. Fresh if ep_start==0 (reseed), else
    resume from the saved full state (no reseed). Saves the full state at the end
    iff ep_end < total_episodes (so the next segment continues identically).
    """
    if ep_start == 0:
        train_env.seed(seed)
        _seed_global_rng(seed + _V2_SEED_OFFSET.get(name, 0))
        log = {'losses': [], 'episode_is': [], 'wall_time': 0.0, 'episodes_done': 0}
    else:
        if not resume_path.exists():
            raise SystemExit(f'--episodes {ep_start}:{ep_end} needs resume state '
                             f'{resume_path} — run 0:{ep_start} first.')
        log, ep_done = _load_full_state(resume_path, agent, train_env)
        if ep_done != ep_start:
            raise SystemExit(f'resume state is at ep {ep_done}, not {ep_start}.')

    t0 = time.time()
    for ep in range(ep_start + 1, ep_end + 1):
        s = train_env.reset()
        done, info = False, {}
        while not done:
            a = agent.select_action(s)
            ns, r, done, info = train_env.step(a)
            agent.store(s, a, r, ns, done)
            loss = agent.update()
            if loss is not None:
                log['losses'].append(loss)
            s = ns
        is_val = info.get('implementation_shortfall')
        log['episode_is'].append(float(is_val) if is_val is not None else None)
        if ckpt_dir is not None and ep % ckpt_freq == 0:
            agent.save(str(ckpt_dir / f'{name}_ep{ep}.pt'))

    log['wall_time'] += time.time() - t0
    log['episodes_done'] = ep_end
    if ep_end < total_episodes:
        _save_full_state(resume_path, agent, train_env, log, ep_end)
    elif resume_path.exists():
        resume_path.unlink()
    return log


def cmd_only_agent(args):
    name = args.only_agent
    sim_config = build_sim_config()
    tr = SMOKE_TRAIN if args.smoke else DEFAULT_TRAIN
    ckpt_freq = args.checkpoint_freq or tr['checkpoint_freq']
    total = args.total_episodes or (tr['episodes'] if args.episodes is None else None)
    ep_start, ep_end = (_parse_episodes(args.episodes) if args.episodes is not None
                        else (0, tr['episodes']))
    if total is None:
        total = ep_end
    if ep_end > total:
        raise SystemExit(f'--episodes end {ep_end} > --total-episodes {total}')

    out = _resolve_out(args)
    prepare_v2_output_dir(out, config=None, force_resume=True)   # shared staged dir
    ckpt_dir, log_dir = out / 'checkpoints', out / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    write_or_check_manifest(log_dir, _manifest_payload(
        sim_config, args.seed, total, ckpt_freq, tr['n_val'], tr['n_test'],
        args.device, args.smoke))

    log_path    = log_dir / f'{name}_training.json'
    resume_path = log_dir / f'{name}_resume.pt'
    if ep_start == 0 and log_path.exists() and not args.force_resume:
        raise SystemExit(f'{name} already trained here ({log_path}). Archive to redo.')

    agents, train_env = build_v2_agents(sim_config, args.seed, args.device)
    if name not in agents:
        raise SystemExit(f'--only-agent must be one of {LEARNED_TRAINABLE}')

    print(f'\n### v2-AC STAGED TRAIN {name} eps({ep_start}:{ep_end}/{total}) '
          f'ckpt/{ckpt_freq} seed={args.seed} device={args.device} '
          f'{"[SMOKE]" if args.smoke else ""} -> {out}')
    log = train_segment(agents[name], train_env, name, ep_start, ep_end, total,
                        args.seed, ckpt_dir, ckpt_freq, resume_path)

    with open(log_path, 'w') as f:
        json.dump(log, f)
    dump_config_json(
        {'kind': 'v2_ac_staged_train', 'agent': name, 'sim_config': sim_config,
         'seed': args.seed, 'episodes': f'{ep_start}:{ep_end}',
         'total_episodes': total, 'checkpoint_freq': ckpt_freq,
         'device': args.device, 'smoke': args.smoke},
        log_dir / f'staged_{name}_config.json')
    print(f'  {name} segment done in {log["wall_time"]:.1f}s '
          f'(episodes_done={log["episodes_done"]}). log -> {log_path}')


# ---------------------------------------------------------------------------
# Assemble-eval  (CRN selection + 10k test + tables)
# ---------------------------------------------------------------------------

def _json_safe(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def cmd_assemble_eval(args):
    out = _resolve_out(args)
    ckpt_dir, log_dir = out / 'checkpoints', out / 'logs'
    if not (log_dir / MANIFEST_NAME).exists():
        raise SystemExit(f'No staged manifest in {log_dir}; train agents first.')
    manifest = json.loads((log_dir / MANIFEST_NAME).read_text())
    if args.device != manifest['device']:
        raise SystemExit(f'--device {args.device} != training device '
                         f'{manifest["device"]} (FP bits differ).')

    sim_config = SimConfig(**manifest['sim_config'])
    seed   = manifest['seed']
    n_val  = manifest['n_val']
    n_test = manifest['n_test']
    val_seed  = seed + VAL_SEED_OFFSET
    test_seed = seed + TEST_SEED_OFFSET

    agents, eval_env = build_v2_agents(sim_config, seed, args.device)
    canon = list(agents.keys())

    # ── Checkpoint selection: 1,200-ep CRN validation, min val-CVaR₉₅ ──────
    for name in LEARNED_TRAINABLE:
        paths = selection_lib.ckpt_paths_for(ckpt_dir, name)
        if not paths:
            raise SystemExit(f'No checkpoints for {name} in {ckpt_dir}. '
                             f'Train it before --assemble-eval.')
        sel = selection_lib.select_checkpoints(
            agents[name], paths, AlmgrenChrissEnv(sim_config), val_seed,
            n_val=n_val, alpha=0.95)
        best = sel['best_cvar']
        agents[name].load(best)
        print(f'  {name}: selected {Path(best).name} '
              f'(best-by-val-CVaR₉₅ of {len(paths)} ckpts; '
              f'plateau={len(sel.get("plateau", []))})')

    RS.share_iqn_weights(agents)

    # ── Test evaluation (10k eps) for all agents ──────────────────────────
    print(f'\n### v2-AC TEST EVAL  {n_test} eps  seed={test_seed}')
    all_results, trackers = RS.evaluate_all(agents, eval_env, n_eval=n_test,
                                            seed=test_seed)

    # ── Write outputs (comparison table with cap-frac, arrays, config) ────
    title = f'v2-AC comparison (seed {seed}, {n_test} test eps){" [SMOKE]" if manifest.get("smoke") else ""}'
    table = tables_v2.write_comparison_table(
        all_results, log_dir / 'comparison_table.txt', order=canon, title=title)
    print('\n' + table)

    is_dict = {r['agent_name']: (np.array(trackers[r['agent_name']].is_values) * 1e4)
               for r in all_results}
    with open(log_dir / 'all_results.json', 'w') as f:
        json.dump([_json_safe(r) for r in all_results], f, indent=2)
    with open(log_dir / 'is_arrays.pkl', 'wb') as f:
        pickle.dump({k: v for k, v in is_dict.items()}, f)

    # ── α-ladder on the selected IQN-neutral checkpoint (eval-only) ───────
    ladder_alphas = sorted(set(V2_CVAR_ALPHAS + [1.0]))
    rows = tables_v2.run_alpha_ladder(
        agents['IQN-neutral'], AlmgrenChrissEnv(sim_config),
        ladder_alphas, n_test, test_seed)
    tables_v2.write_alpha_ladder(
        rows, log_dir,
        caption=f'v2-AC CVaR-$\\alpha$ ladder (seed {seed}).',
        label='tab:v2_ac_alpha')
    print('\n' + tables_v2.format_alpha_ladder(rows, title='α-ladder (IQN-neutral)'))

    dump_config_json(
        {'kind': 'v2_ac_assemble_eval', 'seed': seed, 'n_val': n_val,
         'n_test': n_test, 'val_seed': val_seed, 'test_seed': test_seed,
         'alphas': ladder_alphas, 'device': args.device},
        log_dir / 'assemble_config.json')
    print(f'\nAssembled -> {log_dir} '
          f'(comparison_table.txt, all_results.json, is_arrays.pkl, alpha_ladder.{{txt,tex}})')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_out(args) -> Path:
    out = Path(args.out_dir) if args.out_dir else (
        Path(DEFAULT_OUT_ROOT) / ('_smoke' if args.smoke else '') / f'seed{args.seed}')
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


def main():
    ap = argparse.ArgumentParser(description='Design-v2 B2 staged AC study CLI')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--only-agent', choices=list(LEARNED_TRAINABLE),
                      help='Train exactly this one agent into the shared out-dir')
    mode.add_argument('--assemble-eval', action='store_true',
                      help='CRN-select checkpoints, share, test-eval, write tables')

    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--episodes', default=None,
                    help="'a:b' segment or 'N' (default: full/ smoke episodes)")
    ap.add_argument('--total-episodes', type=int, default=None,
                    help='Full training length (for split resume; default = segment end)')
    ap.add_argument('--checkpoint-freq', type=int, default=None)
    ap.add_argument('--out-dir', default=None,
                    help='Override output dir (default results/_v2_ac[/_smoke]/seed{S})')
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
