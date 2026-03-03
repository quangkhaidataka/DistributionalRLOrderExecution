"""
tests/run_tests.py
------------------
Standalone test runner — no pytest required.
Run with: python3 tests/run_tests.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import traceback

from envs import (
    AlmgrenChrissEnv, RegimeSwitchingEnv,
    SimConfig, N_ACTIONS,
)

PASS = '✓'
FAIL = '✗'
results = []

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ''))

def run_section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
cfg = SimConfig(N=10, T=60.0, q0=100_000, p0=100.0,
                sigma=0.00095, eta=2.5e-6, gamma=2.5e-7)
ac  = AlmgrenChrissEnv(cfg);    ac.seed(42)
rs  = RegimeSwitchingEnv(cfg);  rs.seed(42)

# ---------------------------------------------------------------
# Section 1: Interface contract
# ---------------------------------------------------------------
run_section("1 — Interface Contract")

try:
    state = ac.reset()
    check("reset() returns ndarray",   isinstance(state, np.ndarray))
    check("state dtype is float32",    state.dtype == np.float32)
    check("state shape is (6,)",       state.shape == (6,),   str(state.shape))
    check("state_dim == 6",            ac.state_dim == 6)
    check("n_actions == 5",            ac.n_actions == 5)

    s, r, done, info = ac.step(2)
    check("step returns ndarray state",  isinstance(s, np.ndarray))
    check("step returns float reward",   isinstance(r, float))
    check("step returns bool done",      isinstance(done, bool))
    check("step returns dict info",      isinstance(info, dict))
    check("info has x_t key",           'x_t' in info)
    check("info has p_exec key",        'p_exec' in info)
except Exception as e:
    check("Interface contract", False, str(e))

# ---------------------------------------------------------------
# Section 2: State normalization
# ---------------------------------------------------------------
run_section("2 — State Normalization")

try:
    state = ac.reset()
    check("t* = 0.0 at reset",    abs(state[0] - 0.0) < 1e-6,  f"got {state[0]}")
    check("q* = 1.0 at reset",    abs(state[1] - 1.0) < 1e-6,  f"got {state[1]}")
    check("Δp* ≈ 0.0 at reset",   abs(state[2] - 0.0) < 1e-6,  f"got {state[2]}")

    # Run full episode, check finite states
    ac.reset()
    all_finite = True
    for _ in range(cfg.N):
        action = np.random.default_rng(0).integers(0, N_ACTIONS)
        s, _, done, _ = ac.step(int(action))
        if not np.all(np.isfinite(s)):
            all_finite = False
            break
        if done: break
    check("All states finite throughout episode", all_finite)

    # t* increases monotonically
    ac.reset()
    prev_t = -1.0
    monotone = True
    for _ in range(cfg.N):
        s, _, done, _ = ac.step(2)
        if s[0] < prev_t: monotone = False
        prev_t = s[0]
        if done: break
    check("t* increases monotonically", monotone)

except Exception as e:
    check("State normalization", False, str(e))

# ---------------------------------------------------------------
# Section 3: Inventory constraint
# ---------------------------------------------------------------
run_section("3 — Inventory Constraint")

try:
    # Wait every step (action=0) — terminal step must force liquidation
    ac.reset()
    done = False
    while not done:
        _, _, done, _ = ac.step(0)
    check("Full liquidation with wait-only policy",
          ac.q < 1.0, f"q={ac.q:.2f} remaining")

    # Inventory non-negative
    ac.reset()
    neg_inventory = False
    for _ in range(cfg.N):
        _, _, done, _ = ac.step(int(np.random.default_rng(1).integers(0,5)))
        if ac.q < -1e-6: neg_inventory = True
        if done: break
    check("Inventory never negative", not neg_inventory)

    # Action 2 (50%) halves inventory
    ac.reset()
    q_before = ac.q
    ac.step(2)
    check("Action=2 sells 50% of inventory",
          abs(ac.q - q_before * 0.5) < 1.0,
          f"{q_before:.0f}→{ac.q:.0f}")

    # Action 0 (0%) preserves inventory
    ac.reset()
    q_before = ac.q
    ac.step(0)
    check("Action=0 preserves inventory",
          abs(ac.q - q_before) < 1e-6)

except Exception as e:
    check("Inventory constraint", False, str(e))

# ---------------------------------------------------------------
# Section 4: Reward and IS
# ---------------------------------------------------------------
run_section("4 — Reward and IS Calculation")

try:
    # IS returned at episode end
    ac.reset()
    done = False
    is_val = None
    while not done:
        _, _, done, info = ac.step(2)
    is_val = info['implementation_shortfall']
    check("IS returned at episode end",     is_val is not None)
    check("IS is finite",                   np.isfinite(is_val),  f"IS={is_val}")

    # Manual IS verification
    ac.reset()
    p0, q0 = ac.cfg.p0, ac.cfg.q0
    manual_revenue = 0.0
    done = False
    while not done:
        _, _, done, info = ac.step(3)
        manual_revenue += info['x_t'] * info['p_exec']
    expected_is = (p0 * q0 - manual_revenue) / (p0 * q0)
    reported_is = info['implementation_shortfall']
    check("IS formula correct",
          abs(expected_is - reported_is) < 1e-5,
          f"expected={expected_is:.8f} reported={reported_is:.8f}")

    # Reward range reasonable (many episodes)
    rng = np.random.default_rng(0)
    ep_rewards = []
    for ep in range(50):
        ac.reset()
        ep_r = 0.0
        done = False
        while not done:
            _, r, done, _ = ac.step(2)
            ep_r += r
        ep_rewards.append(ep_r)
    mean_r = np.mean(ep_rewards)
    check("Mean reward ∈ [-0.1, 0.1] (Brownian motion symmetric)",
          abs(mean_r) < 0.1,  f"mean_reward={mean_r:.6f}")

except Exception as e:
    check("Reward and IS", False, str(e))
    traceback.print_exc()

# ---------------------------------------------------------------
# Section 5: Regime switching
# ---------------------------------------------------------------
run_section("5 — Regime-Switching Environment")

try:
    # Both regimes visited over many episodes
    regimes_seen = set()
    for _ in range(100):
        rs.reset()
        for _ in range(cfg.N):
            rs.step(2)
            regimes_seen.add(rs.current_regime)
        if len(regimes_seen) == 2: break
    check("Both regimes visited over episodes", len(regimes_seen) == 2,
          str(regimes_seen))

    # Regime-switching has higher IS variance than pure AC
    def collect_is(env, seed, n=200):
        rng = np.random.default_rng(seed)
        env.seed(seed)
        vals = []
        for _ in range(n):
            env.reset()
            done = False
            while not done:
                _, _, done, info = env.step(2)
            vals.append(info['implementation_shortfall'])
        return np.array(vals)

    ac2 = AlmgrenChrissEnv(cfg);   ac2.seed(0)
    rs2 = RegimeSwitchingEnv(cfg); rs2.seed(0)
    ac_is = collect_is(ac2, 0)
    rs_is = collect_is(rs2, 0)
    check("Regime-switching has higher IS variance",
          np.std(rs_is) > np.std(ac_is),
          f"AC std={np.std(ac_is):.6f}, RS std={np.std(rs_is):.6f}")

    # State is still 6-dim (no regime flag leaked)
    rs.reset()
    state = rs._build_state()
    check("Regime NOT directly in state (dim=6)", state.shape == (6,))

except Exception as e:
    check("Regime switching", False, str(e))
    traceback.print_exc()

# ---------------------------------------------------------------
# Section 6: Full episode walkthrough (visual check)
# ---------------------------------------------------------------
run_section("6 — Full Episode Walkthrough")

print()
ac.seed(99)
ac.reset()
print(f"  Stock: simulated | p0={ac.cfg.p0} | q0={ac.cfg.q0:,} | N={ac.cfg.N}")
print(f"  {'t':>3} {'action':>6} {'frac':>5} {'x_t':>9} "
      f"{'p_exec':>8} {'reward':>10} {'q_rem':>9}")
print(f"  {'─'*3} {'─'*6} {'─'*5} {'─'*9} "
      f"{'─'*8} {'─'*10} {'─'*9}")

done  = False
total = 0.0
while not done:
    a = 2   # sell 50% each step
    s, r, done, info = ac.step(a)
    total += r
    is_str = f"  IS={info['implementation_shortfall']*10000:.2f} bps" if done else ""
    print(f"  {ac.t-1:>3} {a:>6} {0.5:>5.2f} "
          f"{info['x_t']:>9.0f} {info['p_exec']:>8.4f} "
          f"{r:>+10.6f} {info['q_remaining']:>9.0f}"
          + is_str)

print(f"\n  Total reward: {total:+.6f}")
check("Episode ran without errors", True)

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print(f"\n{'═'*55}")
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
print(f"  Results: {passed} passed, {failed} failed out of {len(results)} tests")
if failed == 0:
    print("  All tests passed ✓  Environment is ready for training.")
else:
    print("  Some tests failed. Fix before proceeding to training.")
print(f"{'═'*55}")

sys.exit(0 if failed == 0 else 1)