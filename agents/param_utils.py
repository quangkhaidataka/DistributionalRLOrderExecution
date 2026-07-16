"""
agents/param_utils.py
---------------------
Parameter-count helpers and a build-time assertion so that architecture
drift (e.g. an accidental hidden_dim / cos_embedding_dim change) is caught
immediately instead of surfacing as a silent, unreproducible result.

Unified architecture (PLAN.md D1):
    IQN       : hidden_dim=64, cos_embedding_dim=32, n_hidden_layers=2
    DQN/DDQN  : hidden_dim=64, n_hidden_layers=2   (same LayerNorm+ReLU backbone)

For the execution MDP (state_dim=5, n_actions=6) this gives:
    IQN trainable params      = 11462
    DQN/DDQN trainable params = 5190

Counts are over ``net.parameters()`` (trainable only). The IQN cosine
buffer ``quantile_embed.i_vals`` (32 values) is a registered buffer, NOT a
parameter, and is therefore excluded — summing ``state_dict()`` instead
would give 11494 for IQN (see PLAN.md P1-T1 note).

The ``expected_*`` helpers compute the count from the *intended* unified
constants, not from a possibly-drifted config, so a hidden_dim edit makes
``assert_param_count`` fail (drift self-detects).
"""

from __future__ import annotations

# Unified architecture constants (D1) — the single source of truth for the
# expected param counts. Do NOT read these from a live config: the whole point
# is to detect when a config drifts away from them.
UNIFIED_HIDDEN_DIM        = 64
UNIFIED_COS_EMBEDDING_DIM = 32
UNIFIED_N_HIDDEN_LAYERS   = 2


def _backbone_params(state_dim: int, hidden_dim: int, n_hidden_layers: int) -> int:
    """Params of a ``[Linear -> LayerNorm(-> ReLU)] * n`` stack (ReLU has none)."""
    total, in_dim = 0, state_dim
    for _ in range(n_hidden_layers):
        total += in_dim * hidden_dim + hidden_dim   # Linear(in, hidden)
        total += 2 * hidden_dim                     # LayerNorm (weight + bias)
        in_dim = hidden_dim
    return total


def mlp_param_count(state_dim: int, n_actions: int,
                    hidden_dim: int = UNIFIED_HIDDEN_DIM,
                    n_hidden_layers: int = UNIFIED_N_HIDDEN_LAYERS) -> int:
    """_QMLP (DQN/DDQN): backbone + Linear(hidden, n_actions)."""
    return (_backbone_params(state_dim, hidden_dim, n_hidden_layers)
            + hidden_dim * n_actions + n_actions)


def iqn_param_count(state_dim: int, n_actions: int,
                    hidden_dim: int = UNIFIED_HIDDEN_DIM,
                    cos_embedding_dim: int = UNIFIED_COS_EMBEDDING_DIM,
                    n_hidden_layers: int = UNIFIED_N_HIDDEN_LAYERS) -> int:
    """IQNNetwork: StateEncoder backbone + cosine Linear + OutputMLP."""
    state_encoder = _backbone_params(state_dim, hidden_dim, n_hidden_layers)
    cosine        = cos_embedding_dim * hidden_dim + hidden_dim   # Linear(cos, hidden)
    output_mlp    = (hidden_dim * hidden_dim + hidden_dim         # Linear(hidden, hidden)
                     + hidden_dim * n_actions + n_actions)        # Linear(hidden, n_actions)
    return state_encoder + cosine + output_mlp


def expected_counts(state_dim: int, n_actions: int) -> tuple[int, int]:
    """(iqn, mlp) trainable param counts for the unified architecture.

    Computed from the env's (state_dim, n_actions) — so TAQ (also 5/6) yields
    the same (11462, 5190) — and from the fixed unified constants, so any
    config drift is detected by ``assert_param_count``.
    """
    return (iqn_param_count(state_dim, n_actions),
            mlp_param_count(state_dim, n_actions))


def assert_param_count(agent, expected: int, label: str | None = None) -> int:
    """Print and assert an agent's trainable param count (drift guard)."""
    actual = agent.get_num_params()
    name = label or getattr(agent, 'name', agent.__class__.__name__)
    print(f'  [param-count] {name:<16s}: {actual:>6d} params (expected {expected})')
    assert actual == expected, (
        f'PARAM DRIFT: {name} has {actual} trainable params, expected {expected}. '
        f'Check hidden_dim / cos_embedding_dim / n_hidden_layers vs the unified '
        f'architecture (PLAN.md D1).')
    return actual
