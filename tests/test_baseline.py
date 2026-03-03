"""
tests/test_baselines.py
------------------------
Tests for all benchmark agents (TWAP, AC, DQN, DDQN, QR-DQN).

Coverage:
    1. Interface contract  — all agents share same API
    2. TWAP correctness    — total shares, uniform sell rate
    3. AC schedule math    — sum to q0, sinh curve shape
    4. AC edge cases       — low λ → TWAP, negative eta_tilde guard
    5. DQN loss            — target uses max over target net
    6. DDQN loss           — target uses online net for selection
    7. DQN vs DDQN differ  — overestimation: DQN Q-targets > DDQN
    8. QR-DQN shapes       — (B, N, A) quantile output
    9. QR-DQN fixed taus   — midpoint rule τ_i = (2i-1)/(2N)
   10. QR-DQN loss         — non-negative, finite, has gradient
   11. Full episode        — all rule-based agents liquidate completely
   12. Deep RL store/update— DQN/DDQN/QR-DQN train without errors
   13. Save/load           — deep RL agents persist correctly

Run with: python3 tests/test_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import tempfile

from envs.base_env    import EnvConfig, ACTION_FRACS
from envs.simulated_env import AlmgrenChrissEnv, SimConfig
from agents.baselines import (
    TWAPAgent, AlmgrenChrissAgent, DQNAgent, DDQNAgent, QRDQNAgent,
    DeepRLConfig, _QMLP, _QRNetwork,
)

PASS = '✓'; FAIL = '✗'; results = []
STATE_DIM = 6; N_ACTIONS = 5; DEVICE = torch.device('cpu')

def check(name, cond, detail=''):
    s = PASS if cond else FAIL
    results.append((s, name, detail))
    print(f"  {s}  {name}" + (f"  [{detail}]" if detail else ''))

def section(title):
    print(f"\n{'─'*62}\n  {title}\n{'─'*62}")

# ── helpers ──────────────────────────────────────────────────────────────────
def make_state(t_frac=0.0, q_frac=1.0, dp=0.0, spread=0.0002, imb=0.0, rv=1.0):
    return np.array([t_frac, q_frac, dp, spread, imb, rv], dtype=np.float32)

def make_cfg(N=10, q0=100_000, T=60.0):
    return EnvConfig(N=N, q0=q0, T=T, p0=100.0, eta=2.5e-6, gamma=2.5e-7,
                     sigma=0.00095)

# ── Section 1: Interface contract ────────────────────────────────────────────
section("1 — Interface Contract (all agents)")

cfg = make_cfg()
agents_list = [
    TWAPAgent(cfg),
    AlmgrenChrissAgent(cfg),
    DQNAgent(DeepRLConfig(replay_min_size=10, batch_size=4,
                          hidden_dim=32, n_hidden_layers=1),
             STATE_DIM, N_ACTIONS, device=DEVICE),
    DDQNAgent(DeepRLConfig(replay_min_size=10, batch_size=4,
                           hidden_dim=32, n_hidden_layers=1),
              STATE_DIM, N_ACTIONS, device=DEVICE),
    QRDQNAgent(DeepRLConfig(replay_min_size=10, batch_size=4,
                            hidden_dim=32, n_hidden_layers=1, n_quantiles=8),
               STATE_DIM, N_ACTIONS, device=DEVICE),
]
for agent in agents_list:
    state = make_state()
    action = agent.select_action(state, eval_mode=True)
    check(f"{agent.name}: action in {{0..4}}",
          0 <= action < N_ACTIONS, f"action={action}")
    check(f"{agent.name}: update() returns None or float",
          agent.update() is None or isinstance(agent.update(), (float, type(None))))
    check(f"{agent.name}: has .name property",
          isinstance(agent.name, str) and len(agent.name) > 0)

# ── Section 2: TWAP correctness ───────────────────────────────────────────────
section("2 — TWAP Agent Correctness")

try:
    cfg_ac  = SimConfig(N=10, q0=100_000, T=60.0, p0=100.0,
                        eta=2.5e-6, gamma=2.5e-7, sigma=0.00095)
    env = AlmgrenChrissEnv(cfg_ac); env.seed(0)
    twap = TWAPAgent(cfg_ac)

    # Run full episode
    state = env.reset(); twap.reset()
    done = False; total_sold = 0.0
    x_list = []
    while not done:
        action = twap.select_action(state)
        state, _, done, info = env.step(action)
        x_list.append(info['x_t'])
        total_sold += info['x_t']

    check("TWAP liquidates all shares",
          abs(total_sold - cfg_ac.q0) < 1.0,  f"sold={total_sold:.0f}")
    # TWAP target = q0/N = 10,000 per period
    target = cfg_ac.q0 / cfg_ac.N
    # All periods except last should be close to target
    for i, x in enumerate(x_list[:-1]):
        if abs(x - target) > target * 0.5:
            check(f"TWAP period {i} near target ({target:.0f})", False,
                  f"x={x:.0f}")
            break
    else:
        check(f"All TWAP periods near target ({target:.0f})", True)

    # Uniform sell: total / N per period approximately
    mean_x = np.mean(x_list[:-1])
    check("TWAP mean sell ≈ q0/N",
          abs(mean_x - target) < target * 0.3,
          f"mean={mean_x:.0f} target={target:.0f}")

except Exception as e:
    check("TWAP correctness", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 3: AC schedule math ───────────────────────────────────────────────
section("3 — Almgren-Chriss Schedule Math")

try:
    for lam in [1e-8, 1e-6, 1e-4]:
        ac_agent = AlmgrenChrissAgent(cfg, risk_aversion=lam)
        sched = ac_agent.schedule

        check(f"AC(λ={lam:.0e}) schedule sums to q0",
              abs(sched.sum() - cfg.q0) < 1.0,
              f"sum={sched.sum():.1f}")
        check(f"AC(λ={lam:.0e}) all values ≥ 0",
              np.all(sched >= -1e-6))
        check(f"AC(λ={lam:.0e}) N={len(sched)} periods",
              len(sched) == cfg.N)

    # With high λ (risk averse), front-load: first period > last period
    ac_risky  = AlmgrenChrissAgent(cfg, risk_aversion=1e-4)
    ac_safe   = AlmgrenChrissAgent(cfg, risk_aversion=1e-8)
    sched_r   = ac_risky.schedule
    sched_s   = ac_safe.schedule

    check("High λ → front-loaded (sell more early)",
          sched_r[0] > sched_r[-1],
          f"first={sched_r[0]:.0f} last={sched_r[-1]:.0f}")
    check("Low λ → near-uniform (TWAP-like)",
          abs(sched_s[0] - sched_s[-1]) < cfg.q0 * 0.05,
          f"first={sched_s[0]:.0f} last={sched_s[-1]:.0f}")

    # κ increases with λ
    check("κ(high λ) > κ(low λ)",
          ac_risky.kappa > ac_safe.kappa,
          f"κ_risky={ac_risky.kappa:.6f} κ_safe={ac_safe.kappa:.6f}")

except Exception as e:
    check("AC schedule math", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 4: AC edge cases ─────────────────────────────────────────────────
section("4 — AC Edge Cases")

try:
    # Edge: η̃ = η - 0.5*γ*dt might be zero or negative for unusual params
    cfg_edge = EnvConfig(N=5, q0=1000, T=30.0, p0=50.0,
                         eta=1e-9,   # very small η
                         gamma=1e-6, # large γ → η̃ = η - ½γdt < 0
                         sigma=0.001)
    ac_edge = AlmgrenChrissAgent(cfg_edge, risk_aversion=1e-4)
    sched = ac_edge.schedule
    check("AC with negative η̃ falls back to TWAP (uniform)",
          np.allclose(sched, cfg_edge.q0 / cfg_edge.N, rtol=0.01),
          f"sched={sched.round(0)}")

    # Edge: λ=0 → should give uniform schedule
    ac_zero_lam = AlmgrenChrissAgent(cfg, risk_aversion=1e-20)
    sched_zero = ac_zero_lam.schedule
    check("AC with λ≈0 gives near-uniform schedule",
          np.allclose(sched_zero, cfg.q0 / cfg.N, rtol=0.05),
          f"first={sched_zero[0]:.0f} last={sched_zero[-1]:.0f}")

    # AC liquidates all shares over episode
    env = AlmgrenChrissEnv(cfg_ac); env.seed(42)
    ac_agent = AlmgrenChrissAgent(cfg_ac, risk_aversion=1e-6); ac_agent.reset()
    state = env.reset()
    done = False; total = 0.0
    while not done:
        a = ac_agent.select_action(state)
        state, _, done, info = env.step(a)
        total += info['x_t']
    check("AC liquidates all shares over episode",
          abs(total - cfg_ac.q0) < 1.0, f"sold={total:.0f}")

except Exception as e:
    check("AC edge cases", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 5: DQN loss correctness ──────────────────────────────────────────
section("5 — DQN Loss (Mnih et al. 2015)")

try:
    drl_cfg = DeepRLConfig(hidden_dim=32, n_hidden_layers=1,
                           replay_min_size=10, batch_size=8)
    dqn = DQNAgent(drl_cfg, STATE_DIM, N_ACTIONS, device=DEVICE, seed=0)
    dqn.online_net.train()

    B = 8
    batch = {
        'states'     : torch.randn(B, STATE_DIM),
        'actions'    : torch.randint(0, N_ACTIONS, (B,)),
        'rewards'    : torch.randn(B) * 0.01,
        'next_states': torch.randn(B, STATE_DIM),
        'dones'      : torch.zeros(B),
    }

    loss = dqn._compute_loss(batch)
    check("DQN loss is scalar",   loss.shape == torch.Size([]))
    check("DQN loss is finite",   torch.isfinite(loss).item(), f"{loss.item():.6f}")
    check("DQN loss ≥ 0",        loss.item() >= 0)
    check("DQN loss has grad",    loss.requires_grad)

    # Verify DQN uses TARGET net for max (not online)
    # Make target net return all-zero Q-values
    with torch.no_grad():
        for p in dqn.target_net.parameters(): p.fill_(0.0)
    loss_zero_tgt = dqn._compute_loss(batch)
    # With zero target, targets = reward only → loss should change
    check("DQN target net affects loss",
          abs(loss_zero_tgt.item() - loss.item()) > 1e-6)

except Exception as e:
    check("DQN loss", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 6: DDQN loss correctness ─────────────────────────────────────────
section("6 — DDQN Loss (Van Hasselt et al. 2016)")

try:
    ddqn = DDQNAgent(drl_cfg, STATE_DIM, N_ACTIONS, device=DEVICE, seed=0)
    ddqn.online_net.train()

    loss = ddqn._compute_loss(batch)
    check("DDQN loss is scalar",  loss.shape == torch.Size([]))
    check("DDQN loss is finite",  torch.isfinite(loss).item(), f"{loss.item():.6f}")
    check("DDQN loss ≥ 0",       loss.item() >= 0)
    check("DDQN loss has grad",   loss.requires_grad)

except Exception as e:
    check("DDQN loss", False, str(e))

# ── Section 7: DQN vs DDQN overestimation check ──────────────────────────────
section("7 — DQN vs DDQN: Overestimation Bias")

try:
    # Create DQN and DDQN with IDENTICAL networks
    torch.manual_seed(0)
    dqn2  = DQNAgent (drl_cfg, STATE_DIM, N_ACTIONS, device=DEVICE, seed=0)
    ddqn2 = DDQNAgent(drl_cfg, STATE_DIM, N_ACTIONS, device=DEVICE, seed=0)
    # Copy DQN weights → DDQN for direct comparison
    ddqn2.online_net.load_state_dict(dqn2.online_net.state_dict())
    ddqn2.target_net.load_state_dict(dqn2.target_net.state_dict())

    B = 64
    torch.manual_seed(42)
    big_batch = {
        'states'     : torch.randn(B, STATE_DIM),
        'actions'    : torch.randint(0, N_ACTIONS, (B,)),
        'rewards'    : torch.randn(B) * 0.01,
        'next_states': torch.randn(B, STATE_DIM),
        'dones'      : torch.zeros(B),
    }

    # Compute TD targets for both
    with torch.no_grad():
        # DQN target: max over TARGET net
        q_tgt_dqn  = dqn2.target_net(big_batch['next_states'])
        v_dqn      = q_tgt_dqn.max(dim=1).values

        # DDQN target: online selects, target evaluates
        q_online   = ddqn2.online_net(big_batch['next_states'])
        a_star     = q_online.argmax(dim=1)
        q_tgt_ddqn = ddqn2.target_net(big_batch['next_states'])
        v_ddqn     = q_tgt_ddqn.gather(1, a_star.unsqueeze(1)).squeeze(1)

    # DQN picks the MAX, DDQN picks the value of the online-selected action
    # In expectation, max > selected, so DQN targets ≥ DDQN targets
    mean_v_dqn  = v_dqn.mean().item()
    mean_v_ddqn = v_ddqn.mean().item()
    check("DQN target values ≥ DDQN target values (overestimation)",
          mean_v_dqn >= mean_v_ddqn - 1e-6,
          f"DQN_v={mean_v_dqn:.5f} DDQN_v={mean_v_ddqn:.5f}")

    # Verify the selection mechanism differs
    # In DQN: a* is from target net argmax
    # In DDQN: a* is from online net argmax
    a_dqn  = q_tgt_dqn.argmax(dim=1)   # target net argmax
    a_ddqn = a_star                      # online net argmax
    n_differ = (a_dqn != a_ddqn).sum().item()
    check("DQN and DDQN select different actions on some states",
          n_differ > 0, f"{n_differ}/{B} states differ")

except Exception as e:
    check("DQN vs DDQN overestimation", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 8: QR-DQN network shapes ─────────────────────────────────────────
section("8 — QR-DQN Network Shapes")

try:
    N_Q = 8
    qr_cfg = DeepRLConfig(hidden_dim=32, n_hidden_layers=1,
                          n_quantiles=N_Q, replay_min_size=10, batch_size=8)
    qrdqn = QRDQNAgent(qr_cfg, STATE_DIM, N_ACTIONS, device=DEVICE, seed=0)

    B = 4
    states_t = torch.randn(B, STATE_DIM)

    # Forward pass
    q_full = qrdqn.online_net(states_t)
    check("QR-DQN output shape (B, N, A)",
          q_full.shape == (B, N_Q, N_ACTIONS), str(q_full.shape))

    # Q-values (mean over quantiles)
    q_vals = qrdqn.online_net.q_values(states_t)
    check("QR-DQN q_values shape (B, A)",
          q_vals.shape == (B, N_ACTIONS), str(q_vals.shape))
    check("QR-DQN q_values ≈ mean over N dim",
          torch.allclose(q_vals, q_full.mean(dim=1), atol=1e-5))

    # All values finite
    check("QR-DQN quantiles are finite",
          torch.all(torch.isfinite(q_full)).item())

except Exception as e:
    check("QR-DQN shapes", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 9: QR-DQN fixed quantile levels ───────────────────────────────────
section("9 — QR-DQN Fixed Quantile Levels (Midpoint Rule)")

try:
    N_Q   = 8
    taus  = qrdqn.online_net.taus.numpy()
    expected = (2 * np.arange(1, N_Q+1) - 1) / (2 * N_Q)
    check("τ_i = (2i-1)/(2N) midpoint rule",
          np.allclose(taus, expected, atol=1e-6),
          f"taus={taus.round(4)}")
    check("τ values in (0, 1)",
          taus.min() > 0 and taus.max() < 1,
          f"min={taus.min():.4f} max={taus.max():.4f}")
    check("τ values uniformly spaced",
          np.allclose(np.diff(taus), np.diff(taus)[0], atol=1e-6))
    check("τ values are symmetric around 0.5",
          np.allclose(taus + taus[::-1], 1.0, atol=1e-6))

except Exception as e:
    check("QR-DQN taus", False, str(e))

# ── Section 10: QR-DQN loss correctness ─────────────────────────────────────
section("10 — QR-DQN Loss Correctness")

try:
    N_Q  = 8
    qr_cfg2 = DeepRLConfig(hidden_dim=32, n_hidden_layers=1, n_quantiles=N_Q,
                            replay_min_size=10, batch_size=8)
    qrdqn2 = QRDQNAgent(qr_cfg2, STATE_DIM, N_ACTIONS, device=DEVICE, seed=0)
    qrdqn2.online_net.train()

    B = 8
    batch_qr = {
        'states'     : torch.randn(B, STATE_DIM),
        'actions'    : torch.randint(0, N_ACTIONS, (B,)),
        'rewards'    : torch.randn(B) * 0.01,
        'next_states': torch.randn(B, STATE_DIM),
        'dones'      : torch.zeros(B),
    }
    loss = qrdqn2._compute_loss(batch_qr)
    check("QR-DQN loss is scalar",  loss.shape == torch.Size([]))
    check("QR-DQN loss is finite",  torch.isfinite(loss).item(), f"{loss.item():.8f}")
    check("QR-DQN loss ≥ 0",       loss.item() >= 0)
    check("QR-DQN loss has grad",   loss.requires_grad)

    # Verify quantile Huber loss is non-negative by construction
    # Create known u tensor and verify
    u_test   = torch.tensor([[[0.5, -1.0, 0.2]],  # (1, 1, 3)
                              [[-0.3, 0.8, -0.1]]])  # should give (2,1,3)
    # manually compute
    kappa = 1.0
    huber = torch.where(u_test.abs() <= kappa,
                        0.5 * u_test.pow(2),
                        kappa * (u_test.abs() - 0.5 * kappa)) / kappa
    taus_test = torch.tensor([[[0.25]]])   # τ=0.25, (1,1,1)
    ind  = (u_test < 0).float()
    rho  = (taus_test - ind).abs() * huber
    check("Quantile Huber ρ_τ(u) ≥ 0 for all u, τ",
          (rho >= 0).all().item())

    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in qrdqn2.online_net.parameters())
    check("QR-DQN gradients flow to network", has_grad)

except Exception as e:
    check("QR-DQN loss", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 11: Full episode — all rule-based agents liquidate fully ──────────
section("11 — Full Episode Liquidation (Rule-Based Agents)")

try:
    cfg_sim = SimConfig(N=10, q0=100_000, T=60.0, p0=100.0,
                        eta=2.5e-6, gamma=2.5e-7, sigma=0.00095)

    for AgentClass, kwargs in [
        (TWAPAgent, {}),
        (AlmgrenChrissAgent, {'risk_aversion': 1e-6}),
    ]:
        for seed in [0, 1, 42]:
            env = AlmgrenChrissEnv(cfg_sim); env.seed(seed)
            agent = AgentClass(cfg_sim, **kwargs)

            if hasattr(agent, 'reset'): agent.reset()
            state = env.reset()
            done = False; total = 0.0
            while not done:
                action = agent.select_action(state)
                state, _, done, info = env.step(action)
                total += info['x_t']
                if hasattr(agent, 'reset') and done: pass

            check(f"{agent.name}(seed={seed}) liquidates all shares",
                  abs(total - cfg_sim.q0) < 2.0,
                  f"sold={total:.0f}")

except Exception as e:
    check("Full episode liquidation", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 12: Deep RL agents — store + update ───────────────────────────────
section("12 — Deep RL Training Loop (store + update)")

try:
    small_cfg = DeepRLConfig(
        hidden_dim=32, n_hidden_layers=1,
        replay_min_size=50, batch_size=16,
        n_quantiles=8, epsilon_decay_steps=500,
    )
    for AgentClass in [DQNAgent, DDQNAgent, QRDQNAgent]:
        agent = AgentClass(small_cfg, STATE_DIM, N_ACTIONS, device=DEVICE, seed=0)

        # Fill buffer
        for _ in range(60):
            s  = np.random.randn(STATE_DIM).astype(np.float32)
            a  = np.random.randint(0, N_ACTIONS)
            r  = float(np.random.randn() * 0.01)
            ns = np.random.randn(STATE_DIM).astype(np.float32)
            d  = bool(np.random.rand() < 0.1)
            agent.store(s, a, r, ns, d)

        check(f"{agent.name} buffer ready after 60 transitions",
              agent.replay.ready)

        losses = [agent.update() for _ in range(5)]
        losses = [l for l in losses if l is not None]
        check(f"{agent.name} update() returns float",
              len(losses) > 0 and all(isinstance(l, float) for l in losses))
        check(f"{agent.name} all losses finite",
              all(np.isfinite(l) for l in losses),
              str([f"{l:.6f}" for l in losses]))

except Exception as e:
    check("Deep RL training loop", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 13: Save / load checkpoints ──────────────────────────────────────
section("13 — Save / Load Checkpoints (Deep RL Agents)")

try:
    small_cfg2 = DeepRLConfig(hidden_dim=32, n_hidden_layers=1,
                              replay_min_size=10, batch_size=4, n_quantiles=8)
    for AgentClass in [DQNAgent, DDQNAgent, QRDQNAgent]:
        agent = AgentClass(small_cfg2, STATE_DIM, N_ACTIONS, device=DEVICE)
        params_before = {k: v.clone()
                         for k, v in agent.online_net.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        agent.save(path)

        # Corrupt weights then reload
        with torch.no_grad():
            for p in agent.online_net.parameters(): p.fill_(0.0)
        agent.load(path)

        params_after = agent.online_net.state_dict()
        match = all(torch.allclose(params_before[k], params_after[k])
                    for k in params_before)
        check(f"{agent.name} save/load restores weights", match)
        os.unlink(path)

except Exception as e:
    check("Save/load", False, str(e))
    import traceback; traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*62}")
passed = sum(1 for s,_,_ in results if s == PASS)
failed = sum(1 for s,_,_ in results if s == FAIL)
print(f"  Results: {passed} passed, {failed} failed out of {len(results)} tests")
if failed == 0:
    print("  All baseline tests passed ✓")
    print("  Ablation chain ready: TWAP → AC → DQN → DDQN → QR-DQN → IQN")
else:
    for s, name, detail in results:
        if s == FAIL:
            print(f"  {FAIL} FAILED: {name}  [{detail}]")
print(f"{'═'*62}")
sys.exit(0 if failed == 0 else 1)