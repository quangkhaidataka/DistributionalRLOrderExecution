import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ================================================================
# CORRECTED & CLEAN Quantile Distributional RL Network (QR-DQN style)
# ================================================================

class QuantileDQN(nn.Module):
    """
    Quantile-based Distributional Q-Network (QR-DQN).
    
    OUTPUT DIMENSION:
        Shape: (batch_size, num_actions, num_quantiles)
        What each dimension represents:
            - batch_size     : number of states in the mini-batch
            - num_actions    : for EVERY possible discrete action
            - num_quantiles  : the learned quantile values Q(s, a, τ_k)
                               where τ_k = k / (num_quantiles + 1)   [e.g. 200 points]
            → This tensor completely represents the return distribution Z_θ(s,a)
            → Expected value E[Z(s,a)] = .mean(dim=2)
    """
    def __init__(self, state_dim, action_dim, num_quantiles=200, hidden_dim=128):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * num_quantiles)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        flat = self.network(x)                                      # (batch_size, action_dim * num_quantiles)
        quantiles = flat.view(batch_size, self.action_dim, self.num_quantiles)
        return quantiles
    
    def get_mean_q(self, quantiles):
        return quantiles.mean(dim=2)                                # (batch_size, action_dim)


# ================================================================
# Quantile Regression Loss (pinball loss)
# ================================================================

def quantile_loss(current_quantiles, target_quantiles, taus):
    """
    current_quantiles, target_quantiles: both (batch_size, num_quantiles)
    taus: (num_quantiles,)
    """
    errors = target_quantiles - current_quantiles
    huber = torch.where(errors >= 0,
                        taus * errors,
                        (taus - 1.0) * errors)
    return huber.mean()


# ================================================================
# FIXED Training Step (error fixed)
# ================================================================

def distributional_training_step(
    online_net, target_net, optimizer,
    batch_states, batch_actions, batch_rewards,
    batch_next_states, batch_dones,
    gamma=0.99, num_quantiles=200
):
    """
    One gradient update for Quantile Distributional RL.
    All batch_* tensors must be on the same device.
    """
    batch_size = batch_states.shape[0]

    # current_quantiles_all only gives us predictions for all actions.
    # batch_actions tells us which action was actually taken.
    
    # 1. Current quantiles for the actions actually taken
    # In other words, the network is directly outputting the quantile function (inverse CDF) of the return distribution Z(s,a).
    current_quantiles_all = online_net(batch_states)                # (batch_size, action_dim, num_quantiles)
    
    # FIXED indexing: batch_actions (batch_size,) → index (batch_size, 1, num_quantiles)
    index = batch_actions.view(batch_size, 1, 1).expand(-1, 1, num_quantiles)
    current_quantiles = current_quantiles_all.gather(dim=1, index=index).squeeze(1)
    # shape now: (batch_size, num_quantiles)

    # 2. Target quantiles (the "project" step)
    with torch.no_grad():
        next_quantiles_all = target_net(batch_next_states)          # (batch_size, action_dim, num_quantiles)
        
        # Greedy next action using mean Q
        next_mean_q = next_quantiles_all.mean(dim=2)                # (batch_size, action_dim)
        next_actions = next_mean_q.argmax(dim=1, keepdim=True)      # (batch_size, 1)
        
        index_next = next_actions.unsqueeze(2).expand(-1, 1, num_quantiles)
        next_quantiles = next_quantiles_all.gather(dim=1, index=index_next).squeeze(1)
        
        # Distributional Bellman operator (simple affine transform)
        # The (1 - done) mask already sets target = r for terminal states
        target_quantiles = (batch_rewards.unsqueeze(1) +
                           gamma * (1 - batch_dones.float().unsqueeze(1)) * next_quantiles)

    # 3. Fixed τ values
    taus = torch.linspace(1.0 / (num_quantiles + 1),
                          num_quantiles / (num_quantiles + 1),
                          num_quantiles,
                          device=batch_states.device)

    # 4. Loss & update
    loss = quantile_loss(current_quantiles, target_quantiles, taus)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


# ================================================================
# Example usage (copy-paste and run - now error-free)
# ================================================================

if __name__ == "__main__":
    # Example for CartPole (state_dim=4, action_dim=2)
    online_net = QuantileDQN(state_dim=4, action_dim=2, num_quantiles=200)
    target_net = QuantileDQN(state_dim=4, action_dim=2, num_quantiles=200)
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-4)
    
    # Dummy batch (replace with your replay buffer samples)
    batch_size = 32
    dummy_states = torch.randn(batch_size, 4)
    dummy_actions = torch.randint(0, 2, (batch_size,))          # shape (batch_size,)
    dummy_rewards = torch.randn(batch_size)
    dummy_next_states = torch.randn(batch_size, 4)
    dummy_dones = torch.randint(0, 2, (batch_size,)).float()
    
    loss_value = distributional_training_step(
        online_net, target_net, optimizer,
        dummy_states, dummy_actions, dummy_rewards,
        dummy_next_states, dummy_dones,
        gamma=0.99, num_quantiles=200
    )
    
    print(f"✅ Training step completed successfully!")
    print(f"   Loss = {loss_value:.4f}")
    print(f"   Network output shape = {online_net(dummy_states).shape}  ← (batch, actions, quantiles)")