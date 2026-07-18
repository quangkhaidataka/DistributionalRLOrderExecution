"""
experiments/run_selection_appendix.py  (Batch E3)
-------------------------------------------------
Selection-rule robustness appendix (EVAL-ONLY, no training). For a sim env + seed:

  1. Re-evaluate EVERY saved checkpoint of each learned agent on a fresh
     1,200-episode validation set using COMMON RANDOM NUMBERS (the val env AND
     the global torch/np RNG are reseeded to the SAME fixed VAL_SEED before every
     checkpoint), so checkpoints are compared on identical episodes + identical
     IQN τ draws.
  2. Re-select each agent's checkpoint under BOTH rules: (i) min val-CVaR₉₅,
     (ii) min val-mean-IS.
  3. Run the final 10k-episode TEST eval (test seed = seed+99_999, matching the
     main tables) for each selection; IQN-CVaR₀.₉₅ shares IQN-neutral's selected
     weights. Record mean IS, CVaR₉₅, Max, Std, dump fraction.

Per-unit output: results/_selection_appendix/<env>_seed<seed>.json
--summarize: per-env appendix tables (agent × {cvar,mean} × metrics), aggregated
mean±std across seeds, in txt + LaTeX booktabs, plus a per-seed DDQN un-dump report.

Main tables stay cvar-selected (this is appendix material only).

Usage:
    python experiments/run_selection_appendix.py --env jump_diffusion --seed 42
    python experiments/run_selection_appendix.py --summarize
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from envs import AlmgrenChrissEnv, JumpDiffusionEnv, SimConfig
from envs.base_env import N_ACTIONS
from agents.baselines import DQNAgent, DDQNAgent, DeepRLConfig
from agents.iqn_agents import IQNAgent, AgentConfig
from evaluation.metrics import cvar_alpha
from exp_utils import dump_config_json
import run_simulation as RS

ENV_CLASSES = {'almgren_chriss': AlmgrenChrissEnv, 'jump_diffusion': JumpDiffusionEnv}
ENV_SHORT = {'ac': 'almgren_chriss', 'jump': 'jump_diffusion'}
LEARNED = ['DQN', 'DDQN', 'IQN-neutral']
TEST_AGENTS = ['DQN', 'DDQN', 'IQN-neutral', 'IQN-CVaR_0.95']
SEEDS = [42, 123, 7, 2024, 31]
VAL_SEED = 8_675_309          # fixed CRN validation seed (distinct from train/test)
DUMP = N_ACTIONS - 1
OUT = PROJECT_ROOT / 'results' / '_selection_appendix'
METRICS = [('mean_IS_bps', 'Mean'), ('CVaR_0.95_bps', 'CVaR95'),
           ('max_IS_bps', 'Max'), ('std_IS_bps', 'Std'), ('dump_fraction', 'dump')]


def build_learned(sd, na, seed, device):
    dev = torch.device(device)
    return {
        'DQN': DQNAgent(DeepRLConfig(), sd, na, device=dev, seed=seed),
        'DDQN': DDQNAgent(DeepRLConfig(), sd, na, device=dev, seed=seed + 1),
        'IQN-neutral': IQNAgent(AgentConfig(cvar_alpha=1.0), sd, na, device=dev, seed=seed + 3),
        'IQN-CVaR_0.95': IQNAgent(AgentConfig(cvar_alpha=0.95), sd, na, device=dev, seed=seed + 3),
    }


def ckpt_eps(ckpt_dir, name):
    eps = [int(p.split('_ep')[-1].split('.pt')[0])
           for p in glob.glob(str(ckpt_dir / f'{name}_ep*.pt'))]
    return sorted(eps)


def _rollout(agent, env, n, seed, record_first=False):
    env.seed(seed)
    RS._seed_global_rng(seed)          # CRN: identical episodes + τ draws every call
    is_bps, firsts = [], []
    for _ in range(n):
        s = env.reset()
        a = agent.select_action(s, eval_mode=True)
        if record_first:
            firsts.append(int(a))
        done, info = False, {}
        while not done:
            s, _, done, info = env.step(a)
            if not done:
                a = agent.select_action(s, eval_mode=True)
        is_bps.append(float(info['implementation_shortfall']) * 1e4)
    arr = np.asarray(is_bps)
    return arr, np.asarray(firsts)


def test_metrics(agent, env, n, seed):
    arr, firsts = _rollout(agent, env, n, seed, record_first=True)
    return {'mean_IS_bps': float(arr.mean()),
            'std_IS_bps': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            'CVaR_0.95_bps': float(cvar_alpha(arr, 0.95)),
            'max_IS_bps': float(arr.max()),
            'dump_fraction': float((firsts == DUMP).mean())}


def run_unit(env_name, seed, n_val, n_test, device):
    cfg = SimConfig(**RS.DEFAULT_SIM_CONFIG)
    ckpt_dir = PROJECT_ROOT / 'results' / '_seeds' / f'seed{seed}' / env_name / 'checkpoints'
    env = ENV_CLASSES[env_name](cfg)
    sd, na = env.state_dim, env.n_actions
    agents = build_learned(sd, na, seed, device)
    test_seed = seed + 99_999

    print(f'\n### selection-appendix {env_name} seed={seed} '
          f'(val {n_val}ep CRN seed {VAL_SEED}, test {n_test}ep seed {test_seed})')
    sel = {}
    for name in LEARNED:
        eps = ckpt_eps(ckpt_dir, name)
        vals = []
        for ep in eps:
            agents[name].load(str(ckpt_dir / f'{name}_ep{ep}.pt'))
            arr, _ = _rollout(agents[name], env, n_val, VAL_SEED)
            vals.append({'ep': ep, 'val_mean': float(arr.mean()),
                         'val_cvar95': float(cvar_alpha(arr, 0.95))})
        cvar_ep = min(vals, key=lambda v: v['val_cvar95'])['ep']
        mean_ep = min(vals, key=lambda v: v['val_mean'])['ep']
        sel[name] = {'cvar': cvar_ep, 'mean': mean_ep, 'n_ckpts': len(eps)}
        print(f'  {name}: {len(eps)} ckpts | cvar-best ep={cvar_ep} | mean-best ep={mean_ep}')

    test = {}
    for rule in ['cvar', 'mean']:
        for name in LEARNED:
            agents[name].load(str(ckpt_dir / f'{name}_ep{sel[name][rule]}.pt'))
        agents['IQN-CVaR_0.95'].online_net.load_state_dict(
            agents['IQN-neutral'].online_net.state_dict())
        agents['IQN-CVaR_0.95'].target_net.load_state_dict(
            agents['IQN-neutral'].target_net.state_dict())
        rr = {}
        for name in TEST_AGENTS:
            m = test_metrics(agents[name], env, n_test, test_seed)
            m['selected_ep'] = (sel['IQN-neutral'][rule] if name == 'IQN-CVaR_0.95'
                                else sel[name][rule])
            rr[name] = m
        test[rule] = rr
        print(f'  [{rule}] DDQN Std={rr["DDQN"]["std_IS_bps"]:.4f} dump={rr["DDQN"]["dump_fraction"]:.3f}'
              f' | IQN-CVaR95={rr["IQN-CVaR_0.95"]["CVaR_0.95_bps"]:.4f}')

    result = {'env': env_name, 'seed': seed, 'val_seed': VAL_SEED,
              'n_val': n_val, 'n_test': n_test, 'test_seed': test_seed,
              'selected': sel, 'test': test}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f'{env_name}_seed{seed}.json', 'w') as f:
        json.dump(result, f, indent=2)
    dump_config_json({'kind': 'selection_appendix', 'env': env_name, 'seed': seed,
                      'val_seed': VAL_SEED, 'n_val': n_val, 'n_test': n_test,
                      'device': device}, OUT / f'{env_name}_seed{seed}_config.json')
    print(f'  -> {OUT}/{env_name}_seed{seed}.json')
    return result


# ---------------------------------------------------------------------------
# Summarize across seeds -> appendix tables (txt + LaTeX) + DDQN un-dump report
# ---------------------------------------------------------------------------

def _agg(units, env_name):
    """agent -> rule -> metric -> (mean, std) across available seeds."""
    per = {a: {'cvar': {m: [] for m, _ in METRICS}, 'mean': {m: [] for m, _ in METRICS}}
           for a in TEST_AGENTS}
    for u in units:
        if u['env'] != env_name:
            continue
        for rule in ['cvar', 'mean']:
            for a in TEST_AGENTS:
                for m, _ in METRICS:
                    per[a][rule][m].append(u['test'][rule][a][m])
    out = {}
    for a in TEST_AGENTS:
        out[a] = {}
        for rule in ['cvar', 'mean']:
            out[a][rule] = {m: (float(np.mean(per[a][rule][m])), float(np.std(per[a][rule][m], ddof=1)))
                            for m, _ in METRICS if per[a][rule][m]}
    return out


def _txt_table(env_name, agg, n_seeds):
    L = [f'Selection-rule appendix — {env_name} (mean ± std over {n_seeds} seeds, bps). '
         f'cvar = min val-CVaR95 · mean = min val-mean-IS.', '']
    hdr = f'{"Agent":<15}{"rule":<6}' + ''.join(f'{lbl:>16}' for _, lbl in METRICS)
    L += [hdr, '-' * len(hdr)]
    for a in TEST_AGENTS:
        for rule in ['cvar', 'mean']:
            row = f'{a:<15}{rule:<6}'
            for m, _ in METRICS:
                mu, sd = agg[a][rule].get(m, (float('nan'), 0.0))
                row += f'{mu:>8.3f}±{sd:<6.3f}'
            L.append(row)
        L.append('')
    return '\n'.join(L)


def _latex_table(env_name, agg, n_seeds):
    L = [r'\begin{table}[t]\centering',
         rf'\caption{{Selection-rule robustness ({env_name}); mean$\pm$std over {n_seeds} seeds (bps). '
         r'cvar = min val-CVaR$_{95}$, mean = min val-mean-IS.}}',
         r'\begin{tabular}{ll' + 'r' * len(METRICS) + '}', r'\toprule',
         'Agent & Rule & ' + ' & '.join(lbl for _, lbl in METRICS) + r' \\', r'\midrule']
    for a in TEST_AGENTS:
        for rule in ['cvar', 'mean']:
            cells = []
            for m, _ in METRICS:
                mu, sd = agg[a][rule].get(m, (float('nan'), 0.0))
                cells.append(f'{mu:.3f}$\\pm${sd:.3f}')
            L.append(f'{a.replace("_","-")} & {rule} & ' + ' & '.join(cells) + r' \\')
        L.append(r'\addlinespace')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


def summarize():
    units = [json.load(open(p)) for p in sorted(OUT.glob('*_seed*.json'))
             if '_config' not in p.name]
    if not units:
        raise SystemExit(f'No unit results in {OUT}')
    envs = sorted({u['env'] for u in units})
    lines_all = []
    for env_name in envs:
        eu = [u for u in units if u['env'] == env_name]
        agg = _agg(eu, env_name)
        (OUT / f'{env_name}_appendix.txt').write_text(_txt_table(env_name, agg, len(eu)) + '\n')
        (OUT / f'{env_name}_appendix.tex').write_text(_latex_table(env_name, agg, len(eu)) + '\n')
        lines_all.append(_txt_table(env_name, agg, len(eu)))

        # DDQN un-dump report (per seed): does mean-selection lift DDQN off the corner?
        rep = [f'DDQN un-dump check — {env_name} (per seed): '
               f'Std[dump] under cvar vs mean selection', '']
        rep.append(f'{"seed":>6} | {"cvar Std[dump]":>18} | {"mean Std[dump]":>18} | un-dumped?')
        for u in sorted(eu, key=lambda x: x['seed']):
            c, m = u['test']['cvar']['DDQN'], u['test']['mean']['DDQN']
            undump = (m['std_IS_bps'] > 0.05 or m['dump_fraction'] < 0.5) and \
                     (c['std_IS_bps'] <= 0.05 or c['dump_fraction'] >= 0.5)
            rep.append(f'{u["seed"]:>6} | {c["std_IS_bps"]:>8.4f}[{c["dump_fraction"]:>5.3f}] '
                       f'| {m["std_IS_bps"]:>8.4f}[{m["dump_fraction"]:>5.3f}] | {undump}')
        (OUT / f'{env_name}_ddqn_undump.txt').write_text('\n'.join(rep) + '\n')
        lines_all.append('\n'.join(rep))

    print('\n\n'.join(lines_all))
    print(f'\nWrote {OUT}/<env>_appendix.{{txt,tex}} and <env>_ddqn_undump.txt')


def main():
    ap = argparse.ArgumentParser(description='E3 selection-rule robustness appendix')
    ap.add_argument('--env', choices=list(ENV_CLASSES) + list(ENV_SHORT))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--n-val', type=int, default=1200)
    ap.add_argument('--n-test', type=int, default=10000)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()

    if args.summarize:
        summarize()
        return
    if not (args.env and args.seed is not None):
        ap.error('provide --env and --seed (or --summarize)')
    env_name = ENV_SHORT.get(args.env, args.env)
    run_unit(env_name, args.seed, args.n_val, args.n_test, args.device)


if __name__ == '__main__':
    main()
