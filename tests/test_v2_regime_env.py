"""
tests/test_v2_regime_env.py
---------------------------
Design-v2 B3.5 — statistical validation of RegimeJumpEnv WITHOUT trading, plus a
short masked smoke-train of each learned agent.

Standalone script (no pytest). Run from the repo root:

    python3 tests/test_v2_regime_env.py

Statistical checks (10k no-trade episodes per 2×2 scan cell). It reports the
ACTUAL measured statistic next to the expected value, then PASS/FAIL:
    - stationary stress fraction ≈ p₀₁/(p₀₁+p₁₀)   (±0.01)
    - jumps occur ONLY in stress periods           (jumps-in-calm == 0)
    - empirical jump rate ≈ 0.1 per stress-period  (±0.02)
    - spread multiplier fires in stress            (stress/calm spread ≈ 3, ±0.3)
Plus: 200-ep masked smoke-train per learned agent (finite losses) → _smoke/.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import RegimeJumpEnv, SimConfig
from envs.regime_jump_env import RegimeJumpEnv as _RJE          # for FIXED constants
from agents.baselines import DQNAgent, DDQNAgent, DeepRLConfig
from agents.iqn_agents import IQNAgent, AgentConfig

_FAILS = []
def check(name, ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    print(f'  [{tag}] {name}' + (f'  — {detail}' if detail else ''))
    if not ok:
        _FAILS.append(name)

def section(t):
    print(f"\n{'─'*72}\n  {t}\n{'─'*72}")

GRID     = [round(0.025 * i, 3) for i in range(11)]
P_10     = _RJE.P_10          # 0.40 (fixed)
LAMBDA_J = 0.1                # stress jump rate
N_STAT   = 10_000            # no-trade episodes per cell
CELLS    = [(0.002, 0.05), (0.004, 0.05), (0.002, 0.10), (0.004, 0.10)]

def cfg(sig_high, p01, use_rv=False):
    return SimConfig(N=20, T=60.0, q0=100_000, p0=100.0, eta=2.5e-6, gamma=2.5e-7,
                     a=0.001, action_basis='q0', action_fracs=GRID, use_rv_feature=use_rv,
                     sigma_high=sig_high, p_01=p01,
                     jump_intensity=LAMBDA_J, jump_std=0.16, jump_mean=0.0)


# ════════════════════════════════════════════════════════════════════════════
# Statistical env validation (no trading)
# ════════════════════════════════════════════════════════════════════════════
section(f'Statistical env validation — {N_STAT} no-trade episodes per cell')

print(f'  {"cell (σ_high,p01)":<20} {"stress_frac":>12} {"π (exp)":>9} '
      f'{"jump/stress":>12} {"calm jumps":>11} {"spread s/c":>11}')
for sig_high, p01 in CELLS:
    c = cfg(sig_high, p01)
    env = RegimeJumpEnv(c)
    env.seed(20260718)
    pi = p01 / (p01 + P_10)
    n_periods = n_stress = jumps_stress = jumps_calm = 0
    spr_s, spr_c = [], []
    for _ in range(N_STAT):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(0)           # action 0 = no trade
            n_periods += 1
            if info['stress']:
                n_stress += 1
                jumps_stress += info['n_jumps']
                spr_s.append(info['spread'])
            else:
                jumps_calm += info['n_jumps']
                spr_c.append(info['spread'])
    frac      = n_stress / n_periods
    jump_rate = jumps_stress / max(n_stress, 1)
    spr_ratio = (np.mean(spr_s) / np.mean(spr_c)) if spr_c and spr_s else float('nan')
    print(f'  ({sig_high:<5}, {p01:<4})        {frac:>12.4f} {pi:>9.4f} '
          f'{jump_rate:>12.4f} {jumps_calm:>11d} {spr_ratio:>11.2f}')
    tag = f'cell(σ={sig_high},p01={p01})'
    check(f'{tag} stress_frac ≈ π', abs(frac - pi) < 0.01,
          f'{frac:.4f} vs π={pi:.4f} (Δ={abs(frac-pi):.4f}, tol 0.01)')
    check(f'{tag} jumps ONLY in stress', jumps_calm == 0,
          f'{jumps_calm} jumps occurred in calm periods (must be 0)')
    check(f'{tag} jump rate ≈ 0.10/stress-period', abs(jump_rate - LAMBDA_J) < 0.02,
          f'{jump_rate:.4f} vs {LAMBDA_J} (Δ={abs(jump_rate-LAMBDA_J):.4f}, tol 0.02)')
    check(f'{tag} spread ×3 in stress', abs(spr_ratio - 3.0) < 0.3,
          f'stress/calm={spr_ratio:.2f} vs 3.0 (tol 0.3)')

# use_rv on/off produces 5-D / 6-D state
check('use_rv_feature off → 5-D state', RegimeJumpEnv(cfg(0.004, 0.10, False)).state_dim == 5)
check('use_rv_feature on  → 6-D state', RegimeJumpEnv(cfg(0.004, 0.10, True)).state_dim == 6)

# stress_hit flag: True iff ≥1 stress period; consistent with per-step 'stress'
env = RegimeJumpEnv(cfg(0.004, 0.10)); env.seed(1)
mismatch = 0
for _ in range(500):
    env.reset(); done = False; any_stress = False
    while not done:
        _, _, done, info = env.step(0)
        any_stress = any_stress or info['stress']
    if bool(info['stress_hit']) != any_stress:
        mismatch += 1
check('stress_hit == (≥1 stress period this episode)', mismatch == 0,
      f'{mismatch}/500 episodes mismatched')


# ════════════════════════════════════════════════════════════════════════════
# 200-ep masked smoke-train per learned agent  → results/_v2_regime/_smoke/
# ════════════════════════════════════════════════════════════════════════════
section('200-ep masked smoke-train per learned agent (finite losses)')

SMOKE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'results', '_v2_regime', '_smoke')
os.makedirs(SMOKE_DIR, exist_ok=True)
DEV = torch.device('cpu')
c = cfg(0.004, 0.10)                      # representative (highest-stress) cell
sd, na = RegimeJumpEnv(c).state_dim, RegimeJumpEnv(c).n_actions

def make_agents():
    return {
        'DQN':  DQNAgent(DeepRLConfig(), sd, na, device=DEV, seed=0,
                         action_fracs=GRID, action_basis='q0'),
        'DDQN': DDQNAgent(DeepRLConfig(), sd, na, device=DEV, seed=1,
                          action_fracs=GRID, action_basis='q0'),
        'IQN-neutral': IQNAgent(AgentConfig(1.0), sd, na, device=DEV, seed=3,
                                action_fracs=GRID, action_basis='q0'),
    }

for name, ag in make_agents().items():
    env = RegimeJumpEnv(c); env.seed(42)
    torch.manual_seed(42); np.random.seed(42)
    losses = []
    for _ in range(200):
        s = env.reset(); done = False
        while not done:
            a = ag.select_action(s)
            ns, r, done, info = env.step(a)
            ag.store(s, a, r, ns, done)
            loss = ag.update()
            if loss is not None:
                losses.append(loss)
            s = ns
    # save a checkpoint into _smoke/ as the smoke artifact
    ag.save(os.path.join(SMOKE_DIR, f'smoke_{name}.pt'))
    ok = len(losses) > 0 and all(np.isfinite(l) for l in losses)
    check(f'{name}: 200-ep masked smoke, {len(losses)} finite updates', ok,
          f'last loss={losses[-1] if losses else None}')
    try:
        os.remove(os.path.join(SMOKE_DIR, f'smoke_{name}.pt'))
    except OSError:
        pass


# ── summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
if _FAILS:
    print(f'  test_v2_regime_env: {len(_FAILS)} FAILED → {_FAILS}')
    sys.exit(1)
print('  test_v2_regime_env: ALL CHECKS PASSED')
sys.exit(0)
