"""
envs/__init__.py
----------------
Clean public API for the environments module.
Import from here, not from submodules directly.

Usage:
    from envs import AlmgrenChrissEnv, JumpDiffusionEnv, MeanRevertingEnv
    from envs import SimConfig, EnvConfig
    from envs.taq_env import TAQEnv, TAQConfig
"""

from envs.base_env        import (BaseExecutionEnv, EnvConfig, ACTION_FRACS,
                                  N_ACTIONS, feasible_action_mask)
from envs.simulated_env   import AlmgrenChrissEnv, RegimeSwitchingEnv, SimConfig
# P1-T0: lobster_env.py was moved to unnecessary_files/ and is not part of the
# sim/TAQ pipeline. Importing it here broke every `from envs ...` import.
# from envs.lobster_env     import LobsterEnv, LobsterConfig, LOBSTERLoader
from envs.simulated_env import JumpDiffusionEnv
# Add to imports:
from envs.simulated_env import MeanRevertingEnv
from envs.regime_jump_env import RegimeJumpEnv

__all__ = [
    # Base
    'BaseExecutionEnv',
    'EnvConfig',
    'ACTION_FRACS',
    'N_ACTIONS',
    'feasible_action_mask',
    # Simulated
    'AlmgrenChrissEnv',
    'MeanRevertingEnv',
    'JumpDiffusionEnv',
    'RegimeSwitchingEnv',
    'RegimeJumpEnv',
    'SimConfig',
]