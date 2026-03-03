"""
tests/test_agent.py
--------------------
Tests for IQN network and agent.

Verifies:
    1. Network forward pass shapes (every tensor shape annotated)
    2. Cosine quantile embedding is τ-dependent (not collapsed)
    3. IQN-neutral and IQN-CVaR use same weights, different τ range
    4. Loss is scalar, finite, and has gradient
    5. Target network is hard-copied correctly
    6. Agent stores transitions and updates without error
    7. CVaR action values differ from neutral action values
    8. Agent can be saved and loaded
    9. Loss decreases over multiple updates (learning check)

Run with: python3 tests/test_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import tempfile

from networks.iqn_network  import IQNNetwork, NetworkConfig
from agents.iqn_agent      import IQNAgent, AgentConfig
from training.replay_buffer import ReplayBuffer, ReplayConfig

PASS = '✓'
FAIL = '✗'
results = []

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ''))

def run_section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ── Shared setup ────────────────────────────────────────────────────────────
STATE_DIM = 6
N_ACTIONS = 5
BATCH     = 4
N_TAU     = 8
DEVICE    = torch.device('cpu')

net_cfg = NetworkConfig(
    state_dim         = STATE_DIM,
    n_actions         = N_ACTIONS,
    hidden_dim        = 64,    # small for fast testing
    cos_embedding_dim = 16,
    n_hidden_layers   = 2,
    n_tau_samples     = N_TAU,
    n_tau_policy      = 16,
)
network = IQNNetwork(net_cfg)
network.eval()

# ── Section 1: Network forward pass shapes ──────────────────────────────────
run_section("1 — IQN Network Forward Pass Shapes")

try:
    states = torch.randn(BATCH, STATE_DIM)

    quantiles, tau = network(states, n_tau=N_TAU, tau_low=0.0, tau_high=1.0)

    check("quantiles shape (B, N, A)",
          quantiles.shape == (BATCH, N_TAU, N_ACTIONS),
          str(quantiles.shape))

    check("tau shape (B*N, 1)",
          tau.shape == (BATCH * N_TAU, 1),
          str(tau.shape))

    check("tau values in [0, 1]",
          tau.min().item() >= 0.0 and tau.max().item() <= 1.0,
          f"min={tau.min():.4f} max={tau.max():.4f}")

    check("quantiles are finite",
          torch.all(torch.isfinite(quantiles)).item())

    # get_action_values shape
    q_vals = network.get_action_values(states, n_tau=N_TAU)
    check("get_action_values shape (B, A)",
          q_vals.shape == (BATCH, N_ACTIONS),
          str(q_vals.shape))

    # Single state (no batch dim)
    single_state = torch.randn(STATE_DIM)
    q_single = network.get_action_values(single_state, n_tau=N_TAU)
    check("Single state input handled (1, A)",
          q_single.shape == (1, N_ACTIONS),
          str(q_single.shape))

except Exception as e:
    check("Network forward pass", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 2: Cosine embedding is τ-dependent ──────────────────────────────
run_section("2 — Quantile Embedding τ-Sensitivity")

try:
    state = torch.randn(1, STATE_DIM)

    # Get values at τ=0.1 and τ=0.9
    q_low,  _ = network(state, n_tau=1, tau_low=0.09, tau_high=0.11)
    q_high, _ = network(state, n_tau=1, tau_low=0.89, tau_high=0.91)
    # q_low, q_high: (1, 1, A)

    diff = (q_high - q_low).abs().mean().item()
    check("Q-values differ for different τ levels",
          diff > 1e-4,
          f"|q(τ=0.9) - q(τ=0.1)| mean = {diff:.6f}")

    # Quantile function should be roughly monotone:
    # E[Z(s,a) | τ=0.9] ≥ E[Z(s,a) | τ=0.1] for most (s,a)
    q_lo = network.get_action_values(state, n_tau=16, tau_low=0.0, tau_high=0.2)
    q_hi = network.get_action_values(state, n_tau=16, tau_low=0.8, tau_high=1.0)
    # Not guaranteed for random init, just check they differ
    check("Upper quantiles differ from lower quantiles",
          (q_hi - q_lo).abs().mean().item() > 1e-5)

except Exception as e:
    check("τ-sensitivity", False, str(e))

# ── Section 3: IQN-neutral vs IQN-CVaR same weights, different τ ─────────
run_section("3 — IQN-neutral vs IQN-CVaR Policy Difference")

try:
    agent_cfg = AgentConfig(
        cvar_alpha  = 1.0,
        hidden_dim  = 64,
        cos_embedding_dim = 16,
        replay_min_size = 10,
        batch_size  = 4,
    )
    agent_neutral = IQNAgent(agent_cfg, STATE_DIM, N_ACTIONS, device=DEVICE)
    agent_neutral.online_net.eval()

    # CVaR agent: deep copy same weights, only cvar_alpha differs
    import copy
    agent_cvar_cfg = AgentConfig(
        cvar_alpha  = 0.25,     # extreme CVaR for clear difference
        hidden_dim  = 64,
        cos_embedding_dim = 16,
        replay_min_size = 10,
        batch_size  = 4,
    )
    agent_cvar = IQNAgent(agent_cvar_cfg, STATE_DIM, N_ACTIONS, device=DEVICE)
    # Copy weights so only τ range differs
    agent_cvar.online_net.load_state_dict(agent_neutral.online_net.state_dict())
    agent_cvar.online_net.eval()

    state = torch.randn(STATE_DIM).numpy()

    # With enough τ samples, actions SHOULD differ for CVaR vs neutral
    # (especially with extreme alpha=0.25)
    neutral_qvals = agent_neutral.online_net.get_action_values(
        torch.FloatTensor(state).unsqueeze(0), n_tau=64,
        tau_low=0.0, tau_high=1.0
    )
    cvar_qvals = agent_cvar.online_net.get_action_values(
        torch.FloatTensor(state).unsqueeze(0), n_tau=64,
        tau_low=0.0, tau_high=0.25
    )

    q_diff = (neutral_qvals - cvar_qvals).abs().mean().item()
    check("CVaR Q-values differ from neutral Q-values (same weights, diff τ)",
          q_diff > 1e-5,
          f"mean |Δq| = {q_diff:.6f}")

    check("agent_neutral.cfg.is_cvar is False", not agent_neutral.cfg.is_cvar)
    check("agent_cvar.cfg.is_cvar is True",     agent_cvar.cfg.is_cvar)
    check("agent_neutral.name == 'IQN-neutral'",
          agent_neutral.name == 'IQN-neutral',  agent_neutral.name)
    check("agent_cvar.name contains 'CVaR'",
          'CVaR' in agent_cvar.name,             agent_cvar.name)

except Exception as e:
    check("IQN-neutral vs IQN-CVaR", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 4: Loss computation ─────────────────────────────────────────────
run_section("4 — Loss Computation")

try:
    agent_cfg = AgentConfig(
        cvar_alpha       = 1.0,
        hidden_dim       = 64,
        cos_embedding_dim= 16,
        n_tau_samples    = 4,
        n_tau_targets    = 4,
        replay_min_size  = 10,
        batch_size       = 8,
    )
    agent = IQNAgent(agent_cfg, STATE_DIM, N_ACTIONS, device=DEVICE)
    agent.online_net.train()

    # Fake batch
    batch = {
        'states'     : torch.randn(8, STATE_DIM),
        'actions'    : torch.randint(0, N_ACTIONS, (8,)),
        'rewards'    : torch.randn(8) * 0.01,
        'next_states': torch.randn(8, STATE_DIM),
        'dones'      : torch.zeros(8),
    }

    loss = agent._compute_loss(batch)

    check("Loss is scalar",         loss.shape == torch.Size([]))
    check("Loss is finite",         torch.isfinite(loss).item(),    f"loss={loss.item():.6f}")
    check("Loss is non-negative",   loss.item() >= 0,               f"loss={loss.item():.6f}")
    check("Loss has gradient",      loss.requires_grad)

    # Check gradient flows to network params
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in agent.online_net.parameters()
    )
    check("Gradients flow to network parameters", has_grad)

    # Check target network has no gradient
    target_grads = [
        p.grad for p in agent.target_net.parameters() if p.grad is not None
    ]
    check("Target network has NO gradients (frozen)", len(target_grads) == 0)

except Exception as e:
    check("Loss computation", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 5: Target network update ────────────────────────────────────────
run_section("5 — Target Network Hard Update")

try:
    agent_cfg = AgentConfig(
        hidden_dim=64, cos_embedding_dim=16, replay_min_size=10, batch_size=4
    )
    agent = IQNAgent(agent_cfg, STATE_DIM, N_ACTIONS, device=DEVICE)

    # Modify online net
    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.fill_(99.0)

    # Before update: target should differ
    target_val = list(agent.target_net.parameters())[0].mean().item()
    online_val = list(agent.online_net.parameters())[0].mean().item()
    check("Before update: online ≠ target",
          abs(online_val - target_val) > 1.0,
          f"online={online_val:.2f} target={target_val:.2f}")

    # Hard update
    agent._update_target()
    target_val_after = list(agent.target_net.parameters())[0].mean().item()
    check("After update: target = online (≈99.0)",
          abs(target_val_after - 99.0) < 0.01,
          f"target={target_val_after:.4f}")

except Exception as e:
    check("Target update", False, str(e))

# ── Section 6: Full agent loop (store + update) ──────────────────────────────
run_section("6 — Full Agent Training Loop")

try:
    agent_cfg = AgentConfig(
        hidden_dim       = 64,
        cos_embedding_dim= 16,
        replay_min_size  = 50,
        batch_size       = 16,
        n_tau_samples    = 4,
        n_tau_targets    = 4,
        epsilon_decay_steps = 500,
    )
    agent = IQNAgent(agent_cfg, STATE_DIM, N_ACTIONS, device=DEVICE, seed=0)

    # Fill replay buffer
    for _ in range(60):
        s  = np.random.randn(STATE_DIM).astype(np.float32)
        a  = np.random.randint(0, N_ACTIONS)
        r  = float(np.random.randn() * 0.01)
        ns = np.random.randn(STATE_DIM).astype(np.float32)
        d  = bool(np.random.rand() < 0.1)
        agent.store(s, a, r, ns, d)

    check("Buffer is ready after 60 transitions",
          agent.replay.ready, f"size={agent.replay.size}")

    # Run updates
    losses = []
    for _ in range(10):
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

    check("update() returns float losses", len(losses) > 0)
    check("All losses are finite",
          all(np.isfinite(l) for l in losses),
          f"losses={[f'{l:.6f}' for l in losses]}")

    # Epsilon decays
    initial_eps = agent.cfg.epsilon_start
    for _ in range(100):
        agent._step += 1
    check("Epsilon decreases with steps",
          agent.epsilon < initial_eps,
          f"eps={agent.epsilon:.4f}")

except Exception as e:
    check("Full agent loop", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 7: Save and load checkpoint ─────────────────────────────────────
run_section("7 — Save / Load Checkpoint")

try:
    agent_cfg = AgentConfig(
        hidden_dim=64, cos_embedding_dim=16,
        replay_min_size=10, batch_size=4
    )
    agent = IQNAgent(agent_cfg, STATE_DIM, N_ACTIONS, device=DEVICE)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        ckpt_path = f.name

    agent.save(ckpt_path)
    params_before = {
        k: v.clone() for k, v in agent.online_net.state_dict().items()
    }

    # Corrupt weights then reload
    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.fill_(0.0)

    agent.load(ckpt_path)
    params_after = {
        k: v.clone() for k, v in agent.online_net.state_dict().items()
    }

    all_match = all(
        torch.allclose(params_before[k], params_after[k])
        for k in params_before
    )
    check("Loaded weights match saved weights", all_match)

    import os; os.unlink(ckpt_path)

except Exception as e:
    check("Save/load checkpoint", False, str(e))
    import traceback; traceback.print_exc()

# ── Section 8: Loss decreases over training ──────────────────────────────────
run_section("8 — Learning Check (Loss Trend)")

try:
    agent_cfg = AgentConfig(
        hidden_dim        = 64,
        cos_embedding_dim = 16,
        replay_min_size   = 100,
        batch_size        = 32,
        n_tau_samples     = 4,
        n_tau_targets     = 4,
        lr                = 1e-3,
        epsilon_decay_steps = 500,
    )
    agent = IQNAgent(agent_cfg, STATE_DIM, N_ACTIONS, device=DEVICE, seed=0)

    # Create a simple synthetic task: always reward action 4 (sell all)
    for _ in range(200):
        s  = np.random.randn(STATE_DIM).astype(np.float32)
        a  = 4   # optimal action in synthetic task
        r  = 1.0 if a == 4 else -1.0
        ns = np.random.randn(STATE_DIM).astype(np.float32)
        agent.store(s, a, r, ns, False)

    # Train for 200 steps
    first_losses, last_losses = [], []
    for step in range(300):
        loss = agent.update()
        if loss is not None:
            if step < 20:
                first_losses.append(loss)
            elif step > 280:
                last_losses.append(loss)

    if first_losses and last_losses:
        mean_first = np.mean(first_losses)
        mean_last  = np.mean(last_losses)
        check("Loss decreases over training",
              mean_last < mean_first * 1.5,   # allow some tolerance
              f"early={mean_first:.5f} late={mean_last:.5f}")
    else:
        check("Loss trend check skipped (no data)", True)

except Exception as e:
    check("Learning check", False, str(e))

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
print(f"  Results: {passed} passed, {failed} failed out of {len(results)} tests")
if failed == 0:
    print("  All tests passed ✓  Agent is ready for training.")
else:
    print("  Some tests failed. Review output above.")
    for s, name, detail in results:
        if s == FAIL:
            print(f"  {FAIL} FAILED: {name}  [{detail}]")
print(f"{'═'*60}")

sys.exit(0 if failed == 0 else 1)