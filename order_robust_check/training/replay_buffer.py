"""
training/replay_buffer.py
-------------------------
Experience replay buffer for off-policy RL training.

Design choices:
    - Circular numpy array (not deque) for O(1) sampling
    - Pre-allocate all memory at init — no dynamic allocation
      during training (important for MacBook Air RAM budget)
    - Returns torch tensors directly → no conversion overhead
      in the training loop

Memory budget for MacBook Air (8GB):
    50K transitions × (6 + 1 + 1 + 6 + 1) float32
    = 50K × 15 × 4 bytes = ~3 MB  ✓ very manageable
"""

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class ReplayConfig:
    capacity  : int   = 50_000   # max transitions stored
    batch_size: int   = 64       # minibatch size per update
    min_size  : int   = 1_000    # min transitions before training starts


class ReplayBuffer:
    """
    Fixed-size circular replay buffer.

    Stores transitions (s, a, r, s', done) as pre-allocated
    numpy arrays. Sampling returns torch tensors ready for
    the IQN loss computation.
    """

    def __init__(self, cfg: ReplayConfig, state_dim: int, device: torch.device):
        self.cfg       = cfg
        self.state_dim = state_dim
        self.device    = device
        self.capacity  = cfg.capacity

        # Pre-allocate numpy arrays — fills memory once at init
        self.states      = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(self.capacity,              dtype=np.int64)
        self.rewards     = np.zeros(self.capacity,              dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros(self.capacity,              dtype=np.float32)

        self._ptr  = 0      # write pointer (circular)
        self._size = 0      # current number of valid transitions

    def push(
        self,
        state      : np.ndarray,
        action     : int,
        reward     : float,
        next_state : np.ndarray,
        done       : bool,
    ) -> None:
        """Store one transition. Overwrites oldest if full."""
        idx = self._ptr % self.capacity

        self.states[idx]      = state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.next_states[idx] = next_state
        self.dones[idx]       = float(done)

        self._ptr  += 1
        self._size  = min(self._size + 1, self.capacity)

    def sample(self) -> dict[str, torch.Tensor]:
        """
        Sample a random minibatch.

        Returns dict of tensors on self.device:
            states      : (B, state_dim)
            actions     : (B,)
            rewards     : (B,)
            next_states : (B, state_dim)
            dones       : (B,)
        """
        assert self.ready, "Buffer not ready: not enough transitions yet."

        idx = np.random.randint(0, self._size, size=self.cfg.batch_size)

        return {
            'states'     : torch.FloatTensor(self.states[idx]).to(self.device),
            'actions'    : torch.LongTensor (self.actions[idx]).to(self.device),
            'rewards'    : torch.FloatTensor(self.rewards[idx]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[idx]).to(self.device),
            'dones'      : torch.FloatTensor(self.dones[idx]).to(self.device),
        }

    @property
    def ready(self) -> bool:
        """True once enough transitions are stored to start training."""
        return self._size >= self.cfg.min_size

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size