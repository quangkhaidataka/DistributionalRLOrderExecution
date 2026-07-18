"""
tests/test_v2_env.py
--------------------
Design-v2 B1 logic tests for the refactored execution engine (q0 action basis,
feasible-action masking, N-agnostic baselines, CRN determinism).

Standalone script (no pytest): prints a PASS/FAIL line per check and exits
non-zero if any check fails. Run from the repo root:

    python3 tests/test_v2_env.py

Covers the plan's invariants:
    T2  env invariants over 1,000 random action sequences (q0 basis)
    T3  mask consistency (selection + Bellman target-max share ONE mask fn)
    T4  baseline schedules @ N=20 (TWAP / AC / MaxSpeed) + zero-noise IS
    T6  common-random-numbers determinism
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import AlmgrenChrissEnv, SimConfig
from envs.base_env import feasible_action_mask
from agents.baselines import (TWAPAgent, AlmgrenChrissAgent, MaxSpeedAgent,
                              DQNAgent, DDQNAgent, DeepRLConfig)
from agents.iqn_agents import IQNAgent, AgentConfig

# ── tiny test harness ───────────────────────────────────────────────────────
_FAILS = []
def check(name: str, ok: bool, detail: str = '') -> None:
    tag = 'PASS' if ok else 'FAIL'
    print(f'  [{tag}] {name}' + (f'  — {detail}' if detail and not ok else ''))
    if not ok:
        _FAILS.append(name)

def section(title: str) -> None:
    print(f"\n{'─'*66}\n  {title}\n{'─'*66}")

# ── Design-v2 constants (PLAN_V2.md) ────────────────────────────────────────
N        = 20
Q0       = 100_000
P0       = 100.0
CAP      = 0.25
GRID     = [round(0.025 * i, 3) for i in range(11)]   # 0, .025, …, .25
DEVICE   = torch.device('cpu')

def v2_cfg(**over) -> SimConfig:
    base = dict(N=N, T=60.0, q0=Q0, p0=P0, eta=2.5e-6, gamma=2.5e-7,
                a=0.001, sigma=0.00095, action_basis='q0', action_fracs=GRID)
    base.update(over)
    return SimConfig(**base)

def roll(agent, env, seed=0):
    """Roll one full episode; return per-step (t, x_t) list and terminal IS."""
    env.seed(seed)
    s = env.reset()
    if hasattr(agent, 'reset'):
        agent.reset()
    steps, done, info, t = [], False, {}, 0
    while not done:
        a = agent.select_action(s, eval_mode=True)
        s, r, done, info = env.step(a)
        steps.append((t, info['x_t']))
        t += 1
    return steps, info.get('implementation_shortfall')


# ════════════════════════════════════════════════════════════════════════════
# T2 — Env invariants over random action sequences
# ════════════════════════════════════════════════════════════════════════════
section('T2 — env invariants (1,000 random action sequences, q0 basis)')

env = AlmgrenChrissEnv(v2_cfg())
rng = np.random.default_rng(12345)
bad_sum = bad_neg = bad_cap = bad_term = 0
for ep in range(1000):
    env.seed(int(rng.integers(0, 1_000_000)))
    s = env.reset()
    sold, done, t = 0.0, False, 0
    last_q = env.q
    while not done:
        a = int(rng.integers(0, len(GRID)))
        s, r, done, info = env.step(a)
        x = info['x_t']
        sold += x
        if info['q_remaining'] < -1e-6:
            bad_neg += 1
        # cap applies to every non-terminal-by-time step
        if t < N - 1 and x > CAP * Q0 + 1e-3:
            bad_cap += 1
        t += 1
        last_q = info['q_remaining']
    if abs(sold - Q0) > 1.0:                      # terminal force-sell => full liquidation
        bad_sum += 1
    if not (t <= N and last_q <= 1e-6):           # terminated by horizon or exhaustion
        bad_term += 1

check('Σx_t = q0 every episode (full liquidation)', bad_sum == 0, f'{bad_sum} violations')
check('q_t ≥ 0 always (no oversell)',               bad_neg == 0, f'{bad_neg} violations')
check('x_t ≤ 0.25·q0 for t < N−1 (cap)',            bad_cap == 0, f'{bad_cap} violations')
check('terminates by horizon or exhaustion',        bad_term == 0, f'{bad_term} violations')


# ════════════════════════════════════════════════════════════════════════════
# T3 — Mask consistency
# ════════════════════════════════════════════════════════════════════════════
section('T3 — mask consistency (selection & target-max share ONE mask fn)')

fr = np.asarray(GRID)

# (a) masked actions never chosen over 10k random states
iqn = IQNAgent(AgentConfig(1.0), 5, 11, device=DEVICE, seed=3,
               action_fracs=GRID, action_basis='q0')
dqn = DQNAgent(DeepRLConfig(), 5, 11, device=DEVICE, seed=0,
               action_fracs=GRID, action_basis='q0')
ddqn = DDQNAgent(DeepRLConfig(), 5, 11, device=DEVICE, seed=1,
                 action_fracs=GRID, action_basis='q0')
rng = np.random.default_rng(7)
picked_infeasible = 0
for _ in range(10_000):
    q_norm = float(rng.random())            # remaining fraction in [0,1)
    state = np.array([rng.random(), q_norm, 0.0, 2e-4, 0.0], dtype=np.float32)
    m = feasible_action_mask(q_norm, fr, 'q0')
    for ag in (iqn, dqn, ddqn):
        a = ag.select_action(state, eval_mode=True)
        if not m[a]:
            picked_infeasible += 1
check('learned agents never select a masked action (30k picks)',
      picked_infeasible == 0, f'{picked_infeasible} infeasible picks')

# (b) hand-built Q-table target-max ignores masked actions.
#     Force the UNMASKED argmax onto an infeasible action, confirm the masked
#     target-max (via _mask_bias) never lands there.
states = torch.tensor(
    [[0.5, 0.06, 0.0, 2e-4, 0.0]] * 4, dtype=torch.float32)   # q*=0.06 → feasible {0,1,2,3}
mask_row = feasible_action_mask(0.06, fr, 'q0')
infeasible_idx = int(np.flatnonzero(~mask_row)[0])            # e.g. 4 (0.10)
q_fake = torch.zeros(4, 11)
q_fake[:, infeasible_idx] = 10.0                             # unmasked argmax = infeasible
biased = q_fake + iqn._mask_bias(states)
a_star = biased.argmax(dim=1).tolist()
check('Bellman target-max never selects a masked action',
      all(a != infeasible_idx for a in a_star) and bool(mask_row[np.asarray(a_star)].all()),
      f'a*={a_star} infeasible={infeasible_idx}')

# (c) both sites route through the SAME function object (single source of truth)
import agents.iqn_agents as _iqn_mod
import agents.baselines as _base_mod
import envs.base_env as _env_mod
check('one mask function shared by env, iqn, baselines',
      _iqn_mod.feasible_action_mask is _env_mod.feasible_action_mask
      and _base_mod.feasible_action_mask is _env_mod.feasible_action_mask)

# (d) mask semantics: feasible = {frac ≤ q*} ∪ {smallest frac > q*} ∪ {0}
m = feasible_action_mask(0.06, fr, 'q0')
check('mask q*=0.06 → {0,.025,.05,.075}', np.flatnonzero(m).tolist() == [0, 1, 2, 3],
      str(np.flatnonzero(m).tolist()))
check('remaining basis → all-feasible (legacy no-op)',
      bool(feasible_action_mask(0.06, fr, 'remaining').all()))


# ════════════════════════════════════════════════════════════════════════════
# T4 — Baseline schedules @ N=20  (+ zero-noise IS hand-check)
# ════════════════════════════════════════════════════════════════════════════
section('T4 — baseline schedules @ N=20 and zero-noise IS')

env = AlmgrenChrissEnv(v2_cfg())

twap_steps, _ = roll(TWAPAgent(v2_cfg()), env)
twap_x = np.array([x for _, x in twap_steps])
check('TWAP = 0.05·q0 × 20',
      len(twap_x) == 20 and np.allclose(twap_x, 0.05 * Q0, atol=1.0),
      f'len={len(twap_x)} sum={twap_x.sum():.0f}')

ac_steps, _ = roll(AlmgrenChrissAgent(v2_cfg(), risk_aversion=1e-6), env)
ac_x = np.array([x for _, x in ac_steps])
check('AC(λ=1e-6) ≈ TWAP within 1%/period',
      np.all(np.abs(ac_x - 0.05 * Q0) < 0.01 * Q0) and abs(ac_x.sum() - Q0) < 1.0,
      f'max dev {100*np.abs(ac_x-0.05*Q0).max()/Q0:.3f}%')

ms_steps, _ = roll(MaxSpeedAgent(v2_cfg()), env)
ms_x = np.array([x for _, x in ms_steps])
check('MaxSpeed = 0.25·q0 × 4 then exhausted (env terminates on q=0)',
      len(ms_x) == 4 and np.allclose(ms_x, 0.25 * Q0, atol=1.0) and abs(ms_x.sum() - Q0) < 1.0,
      f'sched={(ms_x/Q0).round(3).tolist()}')

# Zero-noise IS = pure temporary-impact cost (σ=0, γ=0, no jumps).
#   p_exec = p0 − η·(x_t/dt);  IS = (η/dt)·Σx_t² / (p0·q0)
znc = v2_cfg(sigma=0.0, gamma=0.0)
zenv = AlmgrenChrissEnv(znc)
_, is_twap = roll(TWAPAgent(znc), zenv)
dt = znc.dt
sigma_x2 = 20 * (0.05 * Q0) ** 2
is_expected = (znc.eta / dt) * sigma_x2 / (P0 * Q0)      # fraction
check('zero-noise TWAP IS = hand-computed temporary-impact cost',
      abs(is_twap - is_expected) < 1e-9,
      f'env={is_twap*1e4:.5f} bps  hand={is_expected*1e4:.5f} bps')


# ════════════════════════════════════════════════════════════════════════════
# T6 — Common-random-numbers determinism
# ════════════════════════════════════════════════════════════════════════════
section('T6 — CRN determinism')

# (a) same seed twice + same fixed action sequence → identical price path & IS
acts = [2, 4, 1, 3, 0, 5, 2, 2, 1, 0]     # arbitrary fixed sequence (indices into GRID)
def fixed_roll(seed):
    e = AlmgrenChrissEnv(v2_cfg())
    e.seed(seed); e.reset()
    ph, done, i, info = [], False, 0, {}
    while not done:
        s, r, done, info = e.step(acts[i % len(acts)])
        ph.append(round(e.p, 10)); i += 1
    return ph, info.get('implementation_shortfall')
p1, is1 = fixed_roll(2024)
p2, is2 = fixed_roll(2024)
check('same seed → identical price path', p1 == p2)
check('same seed → identical IS', is1 == is2)

# (b) CRN independence from policy: with γ=0 the noise stream is fixed by the
#     seed, so two DIFFERENT action sequences see the IDENTICAL price path —
#     i.e. two "checkpoints" evaluated under one seed face identical paths.
def noise_path(seed, action):
    e = AlmgrenChrissEnv(v2_cfg(gamma=0.0))
    e.seed(seed); e.reset()
    ph, done = [], False
    while not done:
        s, r, done, info = e.step(action)          # policy-independent price (γ=0)
        ph.append(round(e.p, 10))
    return ph
pa = noise_path(999, 0)      # "never trade"
pb = noise_path(999, 1)      # "trade a little"
check('γ=0: identical price path under different policies (CRN)',
      pa == pb, f'lenA={len(pa)} lenB={len(pb)}')


# ── summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*66}")
if _FAILS:
    print(f'  test_v2_env: {len(_FAILS)} FAILED → {_FAILS}')
    sys.exit(1)
print('  test_v2_env: ALL CHECKS PASSED')
sys.exit(0)
