"""
envs/__init__.py
----------------
Clean public API for the environments module.
Import from here, not from submodules directly.

Usage:
    from envs import AlmgrenChrissEnv, RegimeSwitchingEnv, LobsterEnv
    from envs import SimConfig, LobsterConfig, EnvConfig
"""

from envs.base_env        import BaseExecutionEnv, EnvConfig, ACTION_FRACS, N_ACTIONS
from envs.simulated_env   import AlmgrenChrissEnv, RegimeSwitchingEnv, SimConfig
from envs.lobster_env     import LobsterEnv, LobsterConfig, LOBSTERLoader
from envs.simulated_env import JumpDiffusionEnv
# Add to imports:
from envs.simulated_env import MeanRevertingEnv

__all__ = [
    # Base
    'BaseExecutionEnv',
    'EnvConfig',
    'ACTION_FRACS',
    'N_ACTIONS',
    # Simulated
    'AlmgrenChrissEnv',
    'RegimeSwitchingEnv',
    'SimConfig',
    # Real data
    'LobsterEnv',
    'LobsterConfig',
    'LOBSTERLoader',
]