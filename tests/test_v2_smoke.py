"""
tests/test_v2_smoke.py
----------------------
Design-v2 B1 smoke test (T5): a short training run of each learned agent on the
v2 q0-basis environment, exercising the masked training path end-to-end.

Standalone script (no pytest). Run from the repo root:

    python3 tests/test_v2_smoke.py

Checks:
    - 200-episode masked training per learned agent produces only FINITE losses
      (no NaN/Inf) — for both the 5-D and 6-D (σ̂ feature) state.
    - build-time param guard prints and asserts the expected trainable counts
      for the v2 configs (5,11)→(11787,5515) and (6,11)→(11851,5579).
    - checkpoint save/load round-trip restores weights exactly.

This is a SMOKE test (≤200 episodes); its outputs are throwaway and never
presented as results.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import AlmgrenChrissEnv, SimConfig
from agents.baselines import DQNAgent, DDQNAgent, DeepRLConfig
from agents.iqn_agents import IQNAgent, AgentConfig
from agents.param_utils import expected_counts, assert_param_count, param_count_table_str

_FAILS = []
def check(name: str, ok: bool, detail: str = '') -> None:
    tag = 'PASS' if ok else 'FAIL'
    print(f'  [{tag}] {name}' + (f'  — {detail}' if detail and not ok else ''))
    if not ok:
        _FAILS.append(name)

def section(title: str) -> None:
    print(f"\n{'─'*66}\n  {title}\n{'─'*66}")

GRID   = [round(0.025 * i, 3) for i in range(11)]   # 11 actions, cap 0.25
DEVICE = torch.device('cpu')
SCRATCH = os.environ.get(
    'CLAUDE_SCRATCH',
    '/private/tmp/claude-502/-Users-user-Desktop-DisRL/5f5256ba-9ed2-4919-be2e-b0d771bf2adb/scratchpad')

def v2_cfg(use_rv: bool) -> SimConfig:
    return SimConfig(N=20, T=60.0, q0=100_000, p0=100.0, eta=2.5e-6, gamma=2.5e-7,
                     a=0.001, sigma=0.00095, action_basis='q0', action_fracs=GRID,
                     use_rv_feature=use_rv)

def make_agents(env):
    sd, na = env.state_dim, env.n_actions
    return {
        'DQN':  DQNAgent(DeepRLConfig(), sd, na, device=DEVICE, seed=0,
                         action_fracs=GRID, action_basis='q0'),
        'DDQN': DDQNAgent(DeepRLConfig(), sd, na, device=DEVICE, seed=1,
                          action_fracs=GRID, action_basis='q0'),
        'IQN-neutral': IQNAgent(AgentConfig(1.0), sd, na, device=DEVICE, seed=3,
                                action_fracs=GRID, action_basis='q0'),
    }

def smoke_train(agent, env, n_episodes=200, seed=42):
    env.seed(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    losses = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = agent.select_action(s)
            ns, r, done, info = env.step(a)
            agent.store(s, a, r, ns, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            s = ns
    return losses


# ════════════════════════════════════════════════════════════════════════════
# Param guard (prints expected counts for the v2 configs)
# ════════════════════════════════════════════════════════════════════════════
section('T5 — build-time param guard (v2 configs)')
print(param_count_table_str())

for use_rv, sd in [(False, 5), (True, 6)]:
    env = AlmgrenChrissEnv(v2_cfg(use_rv))
    check(f'env.state_dim = {sd} (use_rv_feature={use_rv})', env.state_dim == sd,
          f'got {env.state_dim}')
    check('env.n_actions = 11', env.n_actions == 11, f'got {env.n_actions}')
    exp_iqn, exp_mlp = expected_counts(sd, 11)
    agents = make_agents(env)
    ok = True
    for name, ag in agents.items():
        exp = exp_iqn if name.startswith('IQN') else exp_mlp
        try:
            assert_param_count(ag, exp, name)
        except AssertionError as e:
            ok = False
            print('   ', e)
    check(f'param counts match for ({sd},11)', ok)


# ════════════════════════════════════════════════════════════════════════════
# 200-episode masked smoke training (finite losses)
# ════════════════════════════════════════════════════════════════════════════
section('T5 — 200-episode masked smoke training (5-D and 6-D)')

for use_rv, sd in [(False, 5), (True, 6)]:
    for name, ag in make_agents(AlmgrenChrissEnv(v2_cfg(use_rv))).items():
        losses = smoke_train(ag, AlmgrenChrissEnv(v2_cfg(use_rv)))
        finite = len(losses) > 0 and all(np.isfinite(l) for l in losses)
        check(f'{name} [{sd}-D]: {len(losses)} updates, all finite',
              finite, f'last={losses[-1] if losses else None}')


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint save/load round-trip
# ════════════════════════════════════════════════════════════════════════════
section('T5 — checkpoint save/load round-trip')

os.makedirs(SCRATCH, exist_ok=True)
env = AlmgrenChrissEnv(v2_cfg(False))
for name, ag in make_agents(env).items():
    smoke_train(ag, AlmgrenChrissEnv(v2_cfg(False)), n_episodes=20)
    path = os.path.join(SCRATCH, f'smoke_{name}.pt')
    ag.save(path)
    # Fresh agent, load, compare online-net weights tensor-by-tensor.
    fresh = make_agents(env)[name]
    fresh.load(path)
    sd_a = ag.online_net.state_dict()
    sd_b = fresh.online_net.state_dict()
    same = (sd_a.keys() == sd_b.keys()
            and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a))
    check(f'{name}: save/load restores online-net weights exactly', same)
    try:
        os.remove(path)
    except OSError:
        pass


# ── summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*66}")
if _FAILS:
    print(f'  test_v2_smoke: {len(_FAILS)} FAILED → {_FAILS}')
    sys.exit(1)
print('  test_v2_smoke: ALL CHECKS PASSED')
sys.exit(0)
