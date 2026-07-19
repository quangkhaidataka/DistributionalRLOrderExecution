"""
experiments/run_v2_regime_scan.py
---------------------------------
Design-v2 B3 — RegimeJumpEnv calibration scan (PILOT budget; TRAINS models — do
not run during code review).

Selects the (sigma_high, p_01) operating point for the hidden two-state Markov
volatility + stress-only compound-Poisson env (``envs.regime_jump_env``) against
PRE-REGISTERED acceptance criteria, at a pilot training budget. The env's FIXED
class constants (SIGMA_LOW=0.0005, P_10=0.40) and the stress-jump params
(lambda_J=0.1, sigma_J=0.16, mu_J=0) are held constant; only sigma_high and p_01
are scanned.

Grid : sigma_high in {0.002, 0.004} x p_01 in {0.05, 0.10}, seed 42  (4 cells).
Per cell (pilot budget, PILOT/SMOKE episodes):
  - build TWAP + DDQN + IQN-neutral WITH feasible-action masking (q0 basis,
    replay 100k) on a fresh RegimeJumpEnv;
  - train DDQN and IQN-neutral (final-policy weights, no best-by-val restore —
    the same choice scan_jump_calibration.py makes to avoid the val-CVaR bias);
  - evaluate TWAP, DDQN, IQN-neutral (the alpha=1.0 policy);
  - run the eval-only CVaR alpha-ladder {0.95,0.90,0.70,0.50,0.30} on the SAME
    IQN-neutral weights (zero-cost risk control) for criterion (d).

Acceptance criteria (HARD = a,b,c ; SOFT = d, report only):
  (a) Meaningful tail   : TWAP CVaR_0.95 in [8, 20] bps.
  (b) Non-degeneracy    : Std IS > 0.05 bps for EVERY learned agent
                          (DDQN, IQN-neutral).
  (c) Not cap-saturated : IQN-neutral cap_frac < 0.50 (fewer than half of its
                          decision steps pin the per-step q0 cap).
  (d) Differentiation   : the neutral-CVaR gap
                          (IQN-neutral CVaR_0.95 - IQN-CVaR_alpha CVaR_0.95) is
                          > 0 for every ladder alpha AND non-decreasing as alpha
                          decreases across {0.95,0.90,0.70,0.50,0.30}. Reported
                          per-alpha; does NOT gate hard_pass.

hard_pass = a AND b AND c. RECOMMENDED = among hard-passing cells, the one with
the largest alpha=0.95 gap (tie-break). If no cell passes, RECOMMENDED = none
(all failing cells are reported). This script STOPS at the recommendation — it
does NOT write locked_cell.json / auto-lock; the USER locks the cell.

Output (writes ONLY under results/_v2_regime/):
  results/_v2_regime/_scan/<cell>/config.json , metrics.json   (per cell)
  results/_v2_regime/_scan/scan_summary.txt  (human table, PASS/FAIL per cell)
  results/_v2_regime/_scan/scan_summary.csv  (machine: params+metrics+criteria)
Refuses non-empty per-cell dirs (archive first).

Usage:
    # ONE cell per process (fits a short job), then assemble once all four ran:
    python3 experiments/run_v2_regime_scan.py --cell 0.004 0.10
    ...  # repeat for every (sigma_high, p_01) in the grid
    python3 experiments/run_v2_regime_scan.py --summarize

    # whole grid in one process:
    python3 experiments/run_v2_regime_scan.py --device mps

    # tiny logic test (NO full run — B3 only smoke-tests):
    python3 experiments/run_v2_regime_scan.py --cell 0.004 0.10 --smoke
    python3 experiments/run_v2_regime_scan.py --summarize --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # experiments/ siblings

import numpy as np
import torch

from envs import RegimeJumpEnv, SimConfig
from agents.baselines import TWAPAgent, DDQNAgent, DeepRLConfig
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.param_utils import expected_counts, assert_param_count
from exp_utils import dump_config_json, refuse_if_nonempty
import run_simulation as RS
import run_v2_ac as V2
from sweep_cvar_alpha import evaluate_at_alpha

# ---------------------------------------------------------------------------
# Scan design constants
# ---------------------------------------------------------------------------

SIGMA_HIGHS = [0.002, 0.004]        # stress-regime volatility (cfg.sigma_high)
P01S        = [0.05, 0.10]          # P(calm -> stress) per period (cfg.p_01)

# Stress compound-Poisson jump params held CONSTANT across the grid (the env
# reads these from the config; SIGMA_LOW / P_10 are FIXED env class constants).
JUMP_INTENSITY = 0.1                # lambda_J: stress per-period Poisson rate
JUMP_STD       = 0.16               # sigma_J : jump size volatility ($)
JUMP_MEAN      = 0.0                # mu_J    : symmetric, zero-drift jumps

# v2 base config + agents (reuse run_v2_ac's constants so B2/B3 stay in lock-step)
V2_SIM_CONFIG      = dict(V2.V2_SIM_CONFIG)          # N=20, q0-grid, replay-ready
V2_REPLAY_CAPACITY = V2.V2_REPLAY_CAPACITY           # 100k
LADDER_ALPHAS      = sorted(V2.V2_CVAR_ALPHAS, reverse=True)   # [0.95,0.9,0.7,0.5,0.3]

EVAL_AGENTS = ['TWAP', 'DDQN', 'IQN-neutral']
LEARNED     = ['DDQN', 'IQN-neutral']                # trained per cell

# Acceptance thresholds (pre-registered)
TWAP_TAIL_LO, TWAP_TAIL_HI = 8.0, 20.0   # (a) TWAP CVaR_0.95 band (bps)
STD_MIN_BPS                = 0.05        # (b) non-degeneracy floor (bps)
CAP_FRAC_MAX               = 0.50        # (c) IQN-neutral cap-saturation ceiling

# Budgets  (n_feat = episodes for the feature-scale diagnostic, per view;
#           ckpt_freq must divide `episodes` AND `episodes//2` so ep{total} lands
#           on a checkpoint — the reap-safe staging loads the final ep{total}.pt)
DEFAULT_SCAN = dict(episodes=5_000, n_eval=2_000, n_eval_train=200, n_eval_ladder=2_000,
                    n_feat=1_000, ckpt_freq=2_000)
SMOKE_SCAN   = dict(episodes=200,   n_eval=200,   n_eval_train=40,  n_eval_ladder=200,
                    n_feat=100, ckpt_freq=100)

# State-vector feature names (order = base_env._build_state).
FEATURE_NAMES5 = ['t*', 'q*', 'Δp*', 'spread*', 'imb*']
FEATURE_NAMES6 = FEATURE_NAMES5 + ['σ̂*']


def _feature_names(dim: int) -> list:
    return FEATURE_NAMES6 if dim == 6 else FEATURE_NAMES5

DEFAULT_OUT_ROOT = PROJECT_ROOT / 'results' / '_v2_regime' / '_scan'


# ---------------------------------------------------------------------------
# Config + agents
# ---------------------------------------------------------------------------

def cell_name(sigma_high: float, p01: float) -> str:
    return f'cell_sh{sigma_high:g}_p{p01:g}'


def build_cell_config(sigma_high: float, p01: float) -> SimConfig:
    """v2 base config specialised to this grid cell.

    We set ONLY sigma_high, p_01 and the (fixed) stress-jump params — the env
    fixes SIGMA_LOW / P_10 itself, so cfg.sigma_low / cfg.p_10 are intentionally
    left at their dataclass defaults and ignored by RegimeJumpEnv.
    """
    return SimConfig(**V2_SIM_CONFIG,
                     sigma_high=sigma_high, p_01=p01,
                     jump_intensity=JUMP_INTENSITY, jump_std=JUMP_STD,
                     jump_mean=JUMP_MEAN)


def build_regime_agents(sim_config: SimConfig, seed: int, device_str: str):
    """Pilot agent set (TWAP + DDQN + IQN-neutral) + a fresh RegimeJumpEnv.

    Learned agents are built WITH masking (action_fracs=grid, action_basis='q0')
    and replay 100k — identical construction to run_v2_ac.build_v2_agents, only
    the pilot subset. Seeds match the v2 convention (DDQN=seed+1, IQN=seed+3) so
    weights line up with the full B2 study.
    """
    device = torch.device(device_str)
    env = RegimeJumpEnv(sim_config)
    sd, na = env.state_dim, env.n_actions
    grid, basis = sim_config.action_fracs, sim_config.action_basis

    agents = {}
    agents['TWAP'] = TWAPAgent(sim_config)

    drl = DeepRLConfig(replay_capacity=V2_REPLAY_CAPACITY)
    agents['DDQN'] = DDQNAgent(drl, sd, na, device=device, seed=seed + 1,
                               action_fracs=grid, action_basis=basis)

    iqn = AgentConfig(cvar_alpha=1.0, replay_capacity=V2_REPLAY_CAPACITY)
    agents['IQN-neutral'] = IQNAgent(iqn, sd, na, device=device, seed=seed + 3,
                                     action_fracs=grid, action_basis=basis)

    exp_iqn, exp_mlp = expected_counts(sd, na)
    for name, ag in agents.items():
        if hasattr(ag, 'get_num_params'):
            assert_param_count(ag, exp_iqn if name.startswith('IQN') else exp_mlp, name)
    return agents, env


# ---------------------------------------------------------------------------
# Criteria checker
# ---------------------------------------------------------------------------

def score_criteria(agents_m: dict, ladder: list) -> dict:
    """Score one cell. `agents_m[name]` is a slim metric dict; `ladder` is the
    alpha-ladder rows (descending alpha) with a per-row 'gap' vs IQN-neutral."""
    a = TWAP_TAIL_LO <= agents_m['TWAP']['CVaR_0.95_bps'] <= TWAP_TAIL_HI
    b = all(agents_m[n]['std_IS_bps'] > STD_MIN_BPS for n in LEARNED)
    c = agents_m['IQN-neutral']['cap_frac'] < CAP_FRAC_MAX

    gaps = [row['gap'] for row in ladder]                 # descending-alpha order
    all_pos  = all(g > 0.0 for g in gaps)
    monotone = all(gaps[i + 1] >= gaps[i] for i in range(len(gaps) - 1))
    d = all_pos and monotone
    gap95 = next((row['gap'] for row in ladder if abs(row['alpha'] - 0.95) < 1e-9),
                 gaps[0] if gaps else 0.0)

    return {'a_tail': bool(a), 'b_nondeg': bool(b), 'c_cap': bool(c),
            'd_diff_positive': bool(all_pos), 'd_diff_monotone': bool(monotone),
            'd_diff_soft': bool(d), 'hard_pass': bool(a and b and c),
            'gap95': float(gap95)}


# ---------------------------------------------------------------------------
# Feature-scale diagnostic (PLAN_V2 Pipeline-2 step A3b)
# ---------------------------------------------------------------------------

def _roll_states(env, policy, n_episodes: int, seed: int) -> dict:
    """Collect every state vector over `n_episodes`; return per-feature std/min/max.

    policy=None → no-trade (action 0 each step); else roll the (eval-mode) policy.
    Env + global RNG reseeded for reproducibility.
    """
    env.seed(seed)
    RS._seed_global_rng(seed)
    states = []
    for _ in range(n_episodes):
        s = env.reset()
        states.append(np.asarray(s, dtype=np.float64))
        done = False
        while not done:
            a = 0 if policy is None else policy.select_action(s, eval_mode=True)
            s, _, done, _ = env.step(a)
            states.append(np.asarray(s, dtype=np.float64))
    arr = np.stack(states)                                   # (T, D)
    return {'std': arr.std(axis=0).tolist(),
            'min': arr.min(axis=0).tolist(),
            'max': arr.max(axis=0).tolist(),
            'n_states': int(arr.shape[0])}


def collect_feature_scale(env, iqn_agent, n_episodes: int, seed: int) -> dict:
    """Feature-scale diagnostic: state-vector per-feature std/min/max under two
    views — (i) no-trade episodes, (ii) the pilot IQN-neutral policy — to see
    whether the price channel (Δp*) is numerically suppressed vs the O(1)
    features (t*, q*). Cheap (no gradients); does not lock anything."""
    return {'n_episodes': int(n_episodes),
            'no_trade': _roll_states(env, None, n_episodes, seed),
            'policy':   _roll_states(env, iqn_agent, n_episodes, seed + 1)}


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------

def _slim(r: dict) -> dict:
    return {'mean_IS_bps': float(r['mean_IS_bps']),
            'std_IS_bps':  float(r['std_IS_bps']),
            'CVaR_0.95_bps': float(r['CVaR_0.95_bps']),
            'max_IS_bps':  float(r['max_IS_bps']),
            'cap_frac':    float(r['cap_frac'])}


CKPT_FREQ = 2000   # checkpoint every N episodes (reap-safe staging)


def _cell_dirs(out_root, sigma_high, p01):
    out = out_root / cell_name(sigma_high, p01)
    return out, out / 'checkpoints', out / 'logs'


def cell_train(sigma_high, p01, name, ep_start, ep_end, total, seed, device_str,
               out_root, ckpt_freq=CKPT_FREQ):
    """Staged training of ONE learned agent / IQN segment into the cell's ckpt dir.

    Mirrors run_v2_ac's staged path: `V2.train_segment` checkpoints every
    CKPT_FREQ and supports byte-identical `--episodes a:b` resume, so a reaped
    segment re-runs from the last completed segment — the cell is never restarted.
    Uses FINAL-policy weights (no best-by-val restore), matching the scan design.
    """
    out, ckpt_dir, log_dir = _cell_dirs(out_root, sigma_high, p01)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_cell_config(sigma_high, p01)
    agents, train_env = build_regime_agents(cfg, seed, device_str)
    resume_path = log_dir / f'{name}_resume.pt'
    print(f'  [cell sh{sigma_high} p{p01}] train {name} eps({ep_start}:{ep_end}/{total})')
    V2.train_segment(agents[name], train_env, name, ep_start, ep_end, total,
                     seed, ckpt_dir, ckpt_freq, resume_path)
    # Guarantee the FINAL-policy checkpoint exists (cell_assemble loads ep{total}.pt),
    # independent of whether total is divisible by ckpt_freq.
    if ep_end == total:
        agents[name].save(str(ckpt_dir / f'{name}_ep{total}.pt'))


def cell_assemble(sigma_high, p01, budget, seed, device_str, out_root, smoke):
    """Load the cell's FINAL-policy checkpoints, evaluate, score criteria (a)–(d)
    + feature-scale, write metrics.json. Eval / α-ladder / criteria / feature-scale
    logic is UNCHANGED from the original single-process run_cell (results-neutral)."""
    out, ckpt_dir, log_dir = _cell_dirs(out_root, sigma_high, p01)
    total = budget['episodes']
    cfg = build_cell_config(sigma_high, p01)
    agents, _te = build_regime_agents(cfg, seed, device_str)
    eval_env = RegimeJumpEnv(cfg)
    for name in LEARNED:                                     # DDQN, IQN-neutral
        ck = ckpt_dir / f'{name}_ep{total}.pt'
        if not ck.exists():
            raise SystemExit(f'Missing final checkpoint {ck}; train {name} first.')
        agents[name].load(str(ck))

    eval_seed = seed + 99_999
    all_results, _ = RS.evaluate_all(agents, eval_env, n_eval=budget['n_eval'],
                                     seed=eval_seed)
    agents_m = {r['agent_name']: _slim(r) for r in all_results}
    neutral_cvar = agents_m['IQN-neutral']['CVaR_0.95_bps']

    print('  alpha-ladder on IQN-neutral (zero-cost risk control) ...')
    ladder = []
    for alpha in LADDER_ALPHAS:
        m = evaluate_at_alpha(agents['IQN-neutral'], eval_env,
                              budget['n_eval_ladder'], eval_seed, alpha)
        cvar = float(m['CVaR_0.95_bps'])
        ladder.append({'alpha': float(alpha), 'CVaR_0.95_bps': cvar,
                       'mean_IS_bps': float(m['mean_IS_bps']),
                       'gap': float(neutral_cvar - cvar)})
        print(f'    alpha={alpha:<4} CVaR95={cvar:7.3f}  gap={neutral_cvar - cvar:+.3f}')
    agents['IQN-neutral'].cfg.cvar_alpha = 1.0

    print(f'  feature-scale diagnostic ({budget["n_feat"]} eps × 2 views) ...')
    feature_scale = collect_feature_scale(eval_env, agents['IQN-neutral'],
                                          budget['n_feat'], eval_seed + 7)
    _fn = _feature_names(len(feature_scale['no_trade']['std']))
    _nt = feature_scale['no_trade']['std']
    print('    no-trade std: ' + ' '.join(f'{n}={s:.4f}' for n, s in zip(_fn, _nt)) +
          f'  (Δp*/q* = {(_nt[2]/_nt[1] if _nt[1] else float("nan")):.5f})')

    crit = score_criteria(agents_m, ladder)
    print(f'  TWAP CVaR95={agents_m["TWAP"]["CVaR_0.95_bps"]:.3f} | '
          f'IQN-neutral: Std={agents_m["IQN-neutral"]["std_IS_bps"]:.3f} '
          f'cap={agents_m["IQN-neutral"]["cap_frac"]:.2f} '
          f'CVaR95={neutral_cvar:.3f} | gap95={crit["gap95"]:+.3f}')
    print(f'  criteria: (a)tail={crit["a_tail"]} (b)nondeg={crit["b_nondeg"]} '
          f'(c)cap<50%={crit["c_cap"]} -> HARD '
          f'{"PASS" if crit["hard_pass"] else "FAIL"} | '
          f'(d)diff={crit["d_diff_soft"]} (pos={crit["d_diff_positive"]}, '
          f'mono={crit["d_diff_monotone"]})')

    dump_config_json({'kind': 'v2_regime_scan_cell', 'sigma_high': sigma_high,
                      'p_01': p01, 'jump_intensity': JUMP_INTENSITY,
                      'jump_std': JUMP_STD, 'jump_mean': JUMP_MEAN,
                      'seed': seed, 'device': device_str, 'smoke': smoke,
                      'budget': budget, 'sim_config': cfg},
                     out / 'config.json')
    payload = {'sigma_high': sigma_high, 'p_01': p01, 'agents': agents_m,
               'ladder': ladder, 'criteria': crit, 'feature_scale': feature_scale}
    with open(out / 'metrics.json', 'w') as f:
        json.dump(payload, f, indent=2)
    return payload


def run_cell(sigma_high, p01, budget, seed, device_str, out_root, smoke, idx, total_cells):
    """Whole cell in one process (train all segments + assemble). Used for --smoke
    / single-process runs; the reap-safe path is the per-`--stage` phases."""
    print(f'\n{"=" * 68}\n  CELL {idx}/{total_cells}: sigma_high={sigma_high} p_01={p01}'
          f'  (lambda_J={JUMP_INTENSITY}, sigma_J={JUMP_STD}, mu_J={JUMP_MEAN})'
          f'{"  [SMOKE]" if smoke else ""}\n{"=" * 68}')
    total = budget['episodes']
    half = total // 2
    cf = budget['ckpt_freq']
    cell_train(sigma_high, p01, 'DDQN', 0, total, total, seed, device_str, out_root, cf)
    cell_train(sigma_high, p01, 'IQN-neutral', 0, half, total, seed, device_str, out_root, cf)
    cell_train(sigma_high, p01, 'IQN-neutral', half, total, total, seed, device_str, out_root, cf)
    return cell_assemble(sigma_high, p01, budget, seed, device_str, out_root, smoke)


# ---------------------------------------------------------------------------
# Summary + recommendation
# ---------------------------------------------------------------------------

def _gap_at(ladder, alpha):
    for row in ladder:
        if abs(row['alpha'] - alpha) < 1e-9:
            return row['gap']
    return float('nan')


def write_summary(cells, out_root, smoke=False):
    passing = [c for c in cells if c['criteria']['hard_pass']]
    rec = max(passing, key=lambda c: c['criteria']['gap95']) if passing else None

    # ── CSV (machine) ──────────────────────────────────────────────────────
    csv_rows = []
    for c in cells:
        m, cr = c['agents'], c['criteria']
        row = {
            'sigma_high': c['sigma_high'], 'p_01': c['p_01'],
            'jump_intensity': JUMP_INTENSITY, 'jump_std': JUMP_STD,
            'TWAP_CVaR95': round(m['TWAP']['CVaR_0.95_bps'], 4),
            'DDQN_std': round(m['DDQN']['std_IS_bps'], 4),
            'DDQN_CVaR95': round(m['DDQN']['CVaR_0.95_bps'], 4),
            'IQNn_std': round(m['IQN-neutral']['std_IS_bps'], 4),
            'IQNn_cap_frac': round(m['IQN-neutral']['cap_frac'], 4),
            'IQNn_CVaR95': round(m['IQN-neutral']['CVaR_0.95_bps'], 4),
        }
        for alpha in LADDER_ALPHAS:
            row[f'gap_{alpha:g}'] = round(_gap_at(c['ladder'], alpha), 4)
        row.update({
            'a_tail': cr['a_tail'], 'b_nondeg': cr['b_nondeg'], 'c_cap': cr['c_cap'],
            'd_positive': cr['d_diff_positive'], 'd_monotone': cr['d_diff_monotone'],
            'd_diff_soft': cr['d_diff_soft'], 'hard_pass': cr['hard_pass'],
            'gap95': round(cr['gap95'], 4),
            'recommended': bool(rec is not None and rec is c),
        })
        csv_rows.append(row)

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / 'scan_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # ── TXT (human) ────────────────────────────────────────────────────────
    lines = [
        f'RegimeJumpEnv (sigma_high x p_01) calibration scan — PILOT'
        f'{"  [SMOKE]" if smoke else ""}',
        f'Fixed: SIGMA_LOW=0.0005, P_10=0.40, lambda_J={JUMP_INTENSITY}, '
        f'sigma_J={JUMP_STD}, mu_J={JUMP_MEAN}.  Agents: TWAP + DDQN + IQN-neutral.',
        'Criteria: (a) TWAP CVaR95 in [8,20] bps; (b) Std IS>0.05 bps all learned;',
        '          (c) IQN-neutral cap_frac<0.50; (d, soft) neutral-CVaR gap >0 &',
        '          non-decreasing as alpha falls {0.95,0.90,0.70,0.50,0.30}.',
        '',
    ]
    hdr = (f'{"sig_hi":>7} {"p01":>5} | {"TWAP_CV95":>9} {"IQNn_std":>8} '
           f'{"IQNn_cap":>8} {"IQNn_CV95":>9} {"gap95":>7} | '
           f'{"a":>2} {"b":>2} {"c":>2} {"HARD":>5} {"d":>2}')
    lines += [hdr, '-' * len(hdr)]
    for c in cells:
        m, cr = c['agents'], c['criteria']
        mark = '  <== RECOMMENDED' if (rec is not None and rec is c) else ''
        lines.append(
            f'{c["sigma_high"]:>7g} {c["p_01"]:>5g} | '
            f'{m["TWAP"]["CVaR_0.95_bps"]:>9.3f} {m["IQN-neutral"]["std_IS_bps"]:>8.3f} '
            f'{m["IQN-neutral"]["cap_frac"]:>8.2f} {m["IQN-neutral"]["CVaR_0.95_bps"]:>9.3f} '
            f'{cr["gap95"]:>+7.3f} | '
            f'{"Y" if cr["a_tail"] else "N":>2} {"Y" if cr["b_nondeg"] else "N":>2} '
            f'{"Y" if cr["c_cap"] else "N":>2} '
            f'{"PASS" if cr["hard_pass"] else "FAIL":>5} '
            f'{"Y" if cr["d_diff_soft"] else "N":>2}{mark}')

    # Per-cell alpha-ladder gaps (soft criterion detail)
    lines += ['', 'alpha-ladder gaps (IQN-neutral CVaR95 - IQN-CVaR_alpha CVaR95, bps):']
    gap_hdr = (f'{"sig_hi":>7} {"p01":>5} | '
               + ' '.join(f'a={a:g}'.rjust(9) for a in LADDER_ALPHAS)
               + f' | {"pos":>3} {"mono":>4}')
    lines += [gap_hdr, '-' * len(gap_hdr)]
    for c in cells:
        cr = c['criteria']
        gaps = ' '.join(f'{_gap_at(c["ladder"], a):>+9.3f}' for a in LADDER_ALPHAS)
        lines.append(f'{c["sigma_high"]:>7g} {c["p_01"]:>5g} | {gaps} | '
                     f'{"Y" if cr["d_diff_positive"] else "N":>3} '
                     f'{"Y" if cr["d_diff_monotone"] else "N":>4}')

    lines.append('')
    if rec is not None:
        lines.append(f'RECOMMENDED: sigma_high={rec["sigma_high"]}, p_01={rec["p_01"]} '
                     f'(all hard criteria pass; largest alpha=0.95 gap = '
                     f'{rec["criteria"]["gap95"]:+.3f} bps).')
        lines.append('NOTE: recommendation only — this script does NOT lock the '
                     'cell. The user locks it (no locked_cell.json written).')
    else:
        lines.append('RECOMMENDED: NONE — no cell passed all hard criteria (a,b,c). '
                     'Widen the grid or relax thresholds; all failing cells above.')

    # ── Feature-scale diagnostic (PLAN_V2 Pipeline-2 step A3b) ─────────────
    dim = next((len(c['feature_scale']['no_trade']['std'])
                for c in cells if c.get('feature_scale')), None)
    if dim is not None:
        names = _feature_names(dim)
        lines += [
            '',
            'Feature-scale diagnostic — state-vector per-feature std over N episodes '
            '(2 views).',
            'Reading aid: Δp* std vs other features (t*, q*) — is the price channel '
            'numerically suppressed?',
            '(min/max ranges for the recommended cell below; full ranges per cell in '
            '<cell>/metrics.json.',
            ' The action-vs-spread heatmap is produced by '
            'run_v2_regime.py --assemble-eval on the LOCKED cell.)',
        ]
        fs_hdr = (f'{"cell(σ_hi,p01)":>14} {"view":>9} | '
                  + ' '.join(n.rjust(8) for n in names) + f' | {"Δp*/q*":>8}')
        lines += [fs_hdr, '-' * len(fs_hdr)]
        for c in cells:
            fs = c.get('feature_scale')
            if not fs:
                continue
            for view in ('no_trade', 'policy'):
                stds = fs[view]['std']
                ratio = (stds[2] / stds[1]) if stds[1] else float('nan')
                lines.append(
                    f'{c["sigma_high"]:g},{c["p_01"]:g}'.rjust(14) + f' {view:>9} | '
                    + ' '.join(f'{s:8.4f}' for s in stds) + f' | {ratio:8.5f}')
        if rec is not None and rec.get('feature_scale'):
            lines += ['', f'Recommended cell (σ_hi={rec["sigma_high"]:g}, '
                          f'p01={rec["p_01"]:g}) per-feature [min, max]:']
            for view in ('no_trade', 'policy'):
                v = rec['feature_scale'][view]
                rng = '  '.join(f'{names[i]} [{v["min"][i]:+.3f}, {v["max"][i]:+.3f}]'
                                for i in range(len(names)))
                lines.append(f'  {view:>9}: {rng}')

    (out_root / 'scan_summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print(f'\nWrote {out_root}/scan_summary.{{txt,csv}}')

    dump_config_json({'kind': 'v2_regime_scan_summary', 'smoke': smoke,
                      'sigma_highs': SIGMA_HIGHS, 'p01s': P01S,
                      'ladder_alphas': LADDER_ALPHAS,
                      'recommended': (None if rec is None else
                                      {'sigma_high': rec['sigma_high'],
                                       'p_01': rec['p_01'],
                                       'gap95': rec['criteria']['gap95']})},
                     out_root / 'scan_config.json')


def load_cells(out_root):
    """Reassemble the cells list from per-cell metrics.json (for --summarize),
    in grid order (by sigma_high then p_01)."""
    cells = []
    for d in sorted(out_root.glob('cell_sh*_p*')):
        met_p = d / 'metrics.json'
        if not met_p.exists():
            continue
        cells.append(json.loads(met_p.read_text()))
    cells.sort(key=lambda c: (c['sigma_high'], c['p_01']))
    return cells


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Design-v2 B3 RegimeJumpEnv (sigma_high x p_01) calibration scan')
    ap.add_argument('--sigma-highs', nargs='+', type=float, default=SIGMA_HIGHS)
    ap.add_argument('--p01s', nargs='+', type=float, default=P01S)
    ap.add_argument('--cell', nargs=2, type=float, metavar=('SIGMA_HIGH', 'P01'),
                    help='Run exactly ONE grid cell (sigma_high p_01), writing only '
                         'its per-cell config.json/metrics.json — no summary. Lets '
                         'each cell run as a separate short job; assemble afterwards '
                         'with --summarize.')
    ap.add_argument('--summarize', action='store_true',
                    help='Assemble scan_summary.{txt,csv} from existing per-cell '
                         'metrics.json (no training). Run after all --cell runs.')
    ap.add_argument('--episodes', type=int, default=None,
                    help='Override training episodes per learned agent')
    ap.add_argument('--n-eval', type=int, default=None,
                    help='Override eval / alpha-ladder episodes')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--smoke', action='store_true',
                    help='Tiny config (200 eps, eval 200) — logic test only, NO full run.')
    ap.add_argument('--out-root', default=None,
                    help='Output root (default results/_v2_regime/_scan).')
    ap.add_argument('--stage', choices=['ddqn', 'iqn_a', 'iqn_b', 'assemble'],
                    default=None,
                    help='(with --cell) run ONE reap-safe staged phase: ddqn (0:N), '
                         'iqn_a (0:N/2), iqn_b (N/2:N), assemble. Each phase is a '
                         'short job that survives the ~34-min background reap; '
                         're-run a phase to resume — the cell is never restarted.')
    args = ap.parse_args()

    out_root = Path(args.out_root) if args.out_root else DEFAULT_OUT_ROOT
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root

    budget = dict(SMOKE_SCAN if args.smoke else DEFAULT_SCAN)
    if args.episodes is not None:
        budget['episodes'] = args.episodes
    if args.n_eval is not None:
        budget['n_eval'] = args.n_eval
        budget['n_eval_ladder'] = args.n_eval

    if args.summarize:
        cells = load_cells(out_root)
        if not cells:
            raise SystemExit(f'No per-cell results found under {out_root} to summarize.')
        write_summary(cells, out_root, smoke=args.smoke)
        return

    if args.cell is not None:
        sigma_high, p01 = args.cell
        if args.stage is not None:
            total = budget['episodes']
            half = total // 2
            cf = budget['ckpt_freq']
            if args.stage == 'ddqn':
                cell_train(sigma_high, p01, 'DDQN', 0, total, total,
                           args.seed, args.device, out_root, cf)
            elif args.stage == 'iqn_a':
                cell_train(sigma_high, p01, 'IQN-neutral', 0, half, total,
                           args.seed, args.device, out_root, cf)
            elif args.stage == 'iqn_b':
                cell_train(sigma_high, p01, 'IQN-neutral', half, total, total,
                           args.seed, args.device, out_root, cf)
            elif args.stage == 'assemble':
                cell_assemble(sigma_high, p01, budget, args.seed, args.device,
                              out_root, args.smoke)
            print(f'\nCell (sh={sigma_high}, p={p01}) stage {args.stage} done.')
            return
        run_cell(sigma_high, p01, budget, args.seed, args.device, out_root,
                 args.smoke, 1, 1)
        print(f'\nCell (sigma_high={sigma_high}, p_01={p01}) done. '
              f'Run "--summarize" after all cells to assemble scan_summary.')
        return

    # Whole grid in one process.
    grid = [(sh, p) for sh in args.sigma_highs for p in args.p01s]
    cells = []
    for i, (sh, p) in enumerate(grid, 1):
        cells.append(run_cell(sh, p, budget, args.seed, args.device, out_root,
                              args.smoke, i, len(grid)))
    write_summary(cells, out_root, smoke=args.smoke)


if __name__ == '__main__':
    main()
