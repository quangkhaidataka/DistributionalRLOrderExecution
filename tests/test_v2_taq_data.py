"""
tests/test_v2_taq_data.py
-------------------------
Design-v2 B4.5 — TAQ data-pipeline unit tests + masked agent smoke.

Standalone script (no pytest). Run from the repo root:

    python3 tests/test_v2_taq_data.py

Requires data/processed/AAPL_2014_3min_adj.parquet (build it first:
    python3 data/build_taq_3min.py).

Checks (measured value reported next to the expectation):
    - split continuity: adjusted 2014-06-06 close ≈ 2014-06-09 open within a
      normal overnight move
    - no NaN bars in the key columns
    - bars/day ≈ 130 (full 6.5h session at 3-min)
    - TAQEnv v2 episode windows never straddle a day boundary
Plus a 200-episode masked smoke-train per learned agent → results/_v2_taq/_smoke/.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from envs.taq_env import TAQEnv, TAQConfig
from agents.baselines import DQNAgent, DDQNAgent, DeepRLConfig
from agents.iqn_agents import IQNAgent, AgentConfig

PARQUET = os.path.join(PROJECT_ROOT, 'data', 'processed', 'AAPL_2014_3min_adj.parquet')
BAR_SOURCE = 'AAPL_2014_3min_adj.parquet'
GRID = [round(0.025 * i, 3) for i in range(11)]
SPLIT_DATE = '2014-06-09'

_FAILS = []
def check(name, ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    print(f'  [{tag}] {name}' + (f'  — {detail}' if detail else ''))
    if not ok:
        _FAILS.append(name)

def section(t):
    print(f"\n{'─'*72}\n  {t}\n{'─'*72}")


if not os.path.exists(PARQUET):
    print(f'  [FAIL] {PARQUET} missing — run: python3 data/build_taq_3min.py')
    sys.exit(1)

df = pd.read_parquet(PARQUET)
df['date'] = df['date'].astype(str)


# ════════════════════════════════════════════════════════════════════════════
# Data-pipeline unit tests
# ════════════════════════════════════════════════════════════════════════════
section('Data pipeline (AAPL_2014_3min_adj.parquet)')

# 1. Split continuity: adjusted 06-06 close vs 06-09 open.
before = df[df['date'] == '2014-06-06']
after  = df[df['date'] == SPLIT_DATE]
close_b = float(before['mid_price'].iloc[-1])
open_a  = float(after['mid_price'].iloc[0])
gap = abs(open_a - close_b) / close_b
check('split continuity: adj 06-06 close ≈ 06-09 open (<3% overnight)',
      gap < 0.03,
      f'06-06 ${close_b:.2f} vs 06-09 ${open_a:.2f} → gap {gap*100:.2f}% (tol 3%)')
print(f'      measured: 06-06 adj close ${close_b:.2f}, 06-09 open ${open_a:.2f}, gap {gap*100:.2f}%')

# 2. NaN scan.
key = ['mid_price', 'spread', 'imbalance', 'best_bid', 'best_ask', 'ret', 'volatility']
key = [c for c in key if c in df.columns]
nan_total = int(df[key].isna().sum().sum())
check('no NaN bars in key columns', nan_total == 0,
      f'{nan_total} NaN across {key}')
print(f'      measured: NaN count = {nan_total} across {key}')

# 3. Bars/day ≈ 130.
bpd = df.groupby('date').size()
med = int(bpd.median())
frac_full = float((bpd >= 128).mean())
check('bars/day median ≈ 130', 125 <= med <= 132,
      f'median {med} (expected ~130), range [{bpd.min()},{bpd.max()}]')
print(f'      measured: bars/day median {med}, [{bpd.min()},{bpd.max()}], '
      f'{frac_full*100:.0f}% of days ≥128 bars')

# 4. Monotone timestamps per day.
non_mono = sum(0 if g['time'].is_monotonic_increasing else 1 for _, g in df.groupby('date'))
check('timestamps monotone within each day', non_mono == 0, f'{non_mono} non-monotone days')


# ════════════════════════════════════════════════════════════════════════════
# TAQEnv v2: episodes never straddle a day boundary
# ════════════════════════════════════════════════════════════════════════════
section('TAQEnv v2 — episode windows stay within one day')

dates = sorted(df['date'].unique().tolist())
vcfg = TAQConfig(N=20, T=60.0, q0=5000, p0=100.0, eta=1e-5, a=1e-4,
                 bar_source=BAR_SOURCE, bar_minutes=3,
                 action_basis='q0', action_fracs=GRID,
                 data_dir=os.path.join(PROJECT_ROOT, 'data', 'processed'))
env = TAQEnv(vcfg, dates)
check('v2 stride == 1 (one 3-min bar per decision)', env.step_stride == 1,
      f'stride={env.step_stride}')
check('v2 n_actions == 11, state_dim == 5', env.n_actions == 11 and env.state_dim == 5,
      f'n_actions={env.n_actions} state_dim={env.state_dim}')

# Every valid episode window [start, start+(N-1)*stride] must lie in ONE date.
straddle = 0
for start in env._valid_starts:
    end = start + (vcfg.N - 1) * env.step_stride
    if env.data.iloc[start]['date'] != env.data.iloc[end]['date']:
        straddle += 1
check('no episode straddles a day boundary', straddle == 0,
      f'{straddle}/{len(env._valid_starts)} windows straddle')
print(f'      measured: {len(env._valid_starts)} valid starts across {len(dates)} days, '
      f'{straddle} straddling')


# ════════════════════════════════════════════════════════════════════════════
# 200-ep masked smoke-train per learned agent → results/_v2_taq/_smoke/
# ════════════════════════════════════════════════════════════════════════════
section('200-ep masked smoke-train per learned agent (finite losses)')

SMOKE_DIR = os.path.join(PROJECT_ROOT, 'results', '_v2_taq', '_smoke')
os.makedirs(SMOKE_DIR, exist_ok=True)
DEV = torch.device('cpu')
smoke_days = dates[:25]                      # a small slice for a fast smoke
sd, na = env.state_dim, env.n_actions

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
    senv = TAQEnv(vcfg, smoke_days); senv.seed(42)
    torch.manual_seed(42); np.random.seed(42)
    losses = []
    for _ in range(200):
        s = senv.reset(); done = False
        while not done:
            a = ag.select_action(s)
            ns, r, done, info = senv.step(a)
            ag.store(s, a, r, ns, done)
            loss = ag.update()
            if loss is not None:
                losses.append(loss)
            s = ns
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
    print(f'  test_v2_taq_data: {len(_FAILS)} FAILED → {_FAILS}')
    sys.exit(1)
print('  test_v2_taq_data: ALL CHECKS PASSED')
sys.exit(0)
