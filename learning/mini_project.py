import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# ================================================================
# 1. Simple Replay Buffer (same for both agents)
# ================================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.int64),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ================================================================
# 2. Normal DQN Network + Training Step
# ================================================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def dqn_training_step(online_net, target_net, optimizer, batch_states, batch_actions,
                      batch_rewards, batch_next_states, batch_dones, gamma=0.99):
    batch_size = batch_states.shape[0]

    # Current Q-values for taken actions
    current_q = online_net(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

    # Target
    with torch.no_grad():
        next_q = target_net(batch_next_states)
        next_max_q = next_q.max(dim=1)[0]
        target_q = batch_rewards + gamma * (1 - batch_dones) * next_max_q

    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ================================================================
# 3. QR-DQN (Distributional) - exactly the version we debugged together
# ================================================================
class QuantileDQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles=200, hidden=128):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim * num_quantiles)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        flat = self.net(x)
        return flat.view(batch_size, self.action_dim, self.num_quantiles)

    def get_mean_q(self, quantiles):
        return quantiles.mean(dim=2)

def quantile_loss(current, target, taus):
    errors = target - current
    return torch.where(errors >= 0,
                       taus * errors,
                       (taus - 1.0) * errors).mean()

def qr_training_step(online_net, target_net, optimizer, batch_states, batch_actions,
                     batch_rewards, batch_next_states, batch_dones,
                     gamma=0.99, num_quantiles=51):
    batch_size = batch_states.shape[0]

    # Current quantiles for taken actions
    current_all = online_net(batch_states)  # (batch_size, action_dim, num_quantiles)
    index = batch_actions.view(batch_size, 1, 1).expand(-1, 1, num_quantiles)
    current_quantiles = current_all.gather(dim=1, index=index).squeeze(1) #(batch_size, num_quantiles) # ← θ_i for the real action

    # Target quantiles
    with torch.no_grad():
        next_all = target_net(batch_next_states)
        next_mean = next_all.mean(dim=2) #← sum_j q_j θ_j  (mean)
        next_actions = next_mean.argmax(dim=1, keepdim=True)  # ← a*
        index_next = next_actions.unsqueeze(2).expand(-1, 1, num_quantiles)
        next_quantiles = next_all.gather(dim=1, index=index_next).squeeze(1) # ← θ_j(x', a*)

        target_quantiles = (batch_rewards.unsqueeze(1) +
                            gamma * (1 - batch_dones.unsqueeze(1)) * next_quantiles) # ← T θ_j

    taus = torch.linspace(1.0/(num_quantiles+1), num_quantiles/(num_quantiles+1),
                          num_quantiles, device=batch_states.device)
    loss = quantile_loss(current_quantiles, target_quantiles, taus)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ================================================================
# 3. IQN (Distributional) - exactly the version we debugged together
# ================================================================

class IQN(nn.Module):
    """
    Implicit Quantile Network (IQN)
    """
    def __init__(self, state_dim, action_dim, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.final_net = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x, tau):
        """
        x   : (batch_size, state_dim)
        tau : (batch_size, 1)   ← sampled τ ~ Uniform[0,1]
        """
        state_emb = self.state_net(x)
        
        # Cosine embedding (standard in original IQN paper)
        pi = torch.pi
        i = torch.arange(1, self.embedding_dim + 1, device=x.device).view(1, -1).float() # for cosine embedding
        tau_emb = torch.cos(pi * tau * i)          # (batch_size, embedding_dim), ← embedding of τ (paper's reparameterization)
        
        combined = torch.cat([state_emb, tau_emb], dim=1) # condition on τ
        return self.final_net(combined) # outputs Z_τ(x, a) for all actions
    
def iqn_training_step(online_net, target_net, optimizer,
                      batch_states, batch_actions, batch_rewards,
                      batch_next_states, batch_dones,
                      gamma=0.99, num_tau=32):
    """
    IQN training step - exactly same style as your QR-DQN version
    """


    # num_tau is the number of $  \tau  $ samples per batch item



    batch_size = batch_states.shape[0]
    device = batch_states.device

    # 1. Sample random τ for current network
    tau = torch.rand(batch_size * num_tau, 1, device=device) # ← τ ~ U[0,1] for reparameterization

    # Repeat states for batched forward
    states_rep = batch_states.repeat_interleave(num_tau, dim=0) # ( batch*num_tau, state_dim) repeat batch in num_tau times

    # Current predictions
    current_all = online_net(states_rep, tau)                     # (batch*num_tau, action_dim), # Z_β(τ) for sampled τ (here β=identity)
    current_all = current_all.view(batch_size, num_tau, -1) # (batch,num_tau, action_dim)
    
    # Select only the taken action
    current_q = current_all.gather(2, batch_actions.view(batch_size, 1, 1).expand(-1, num_tau, 1)).squeeze(2) #(batch,num_tau)

    # 2. Target
    with torch.no_grad():
        tau_target = torch.rand(batch_size * num_tau, 1, device=device)
        next_states_rep = batch_next_states.repeat_interleave(num_tau, dim=0)
        
        next_all = target_net(next_states_rep, tau_target) # (batch*num_tau, action_dim)
        next_all = next_all.view(batch_size, num_tau, -1)# (batch,num_tau, action_dim) 
        
        # Greedy next action using mean over τ
        next_mean = next_all.mean(dim=1) # ← Q_β = E_τ [Z_β(τ)] (risk-neutral mean)
        next_actions = next_mean.argmax(dim=1, keepdim=True)
        
        next_q = next_all.gather(2, next_actions.unsqueeze(1).expand(-1, num_tau, 1)).squeeze(2)
        
        target_q = (batch_rewards.unsqueeze(1) +
                    gamma * (1 - batch_dones.unsqueeze(1)) * next_q)

    # 3. Pinball loss
    errors = target_q - current_q
    loss = torch.where(errors >= 0,
                       tau.view(batch_size, num_tau) * errors,
                       (tau.view(batch_size, num_tau) - 1.0) * errors).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ================================================================
# 4. Training Loop (common for both agents)
# ================================================================
def train_agent(env_name="CartPole-v1", algorithm="dqn", num_episodes=800,
                batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                epsilon_decay=0.995, target_update=100, memory_size=10000,
                learning_rate=1e-3, seed=42):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ====================== NETWORK & TRAINING STEP ======================
    if algorithm == "dqn":
        online_net = DQN(state_dim, action_dim)
        target_net = DQN(state_dim, action_dim)
        training_step = dqn_training_step
        desc = "DQN"

    elif algorithm == "qr":
        online_net = QuantileDQN(state_dim, action_dim)
        target_net = QuantileDQN(state_dim, action_dim)
        training_step = qr_training_step
        desc = "QR-DQN"

    elif algorithm == "iqn":
        online_net = IQN(state_dim, action_dim)
        target_net = IQN(state_dim, action_dim)
        training_step = iqn_training_step
        desc = "IQN"

    else:
        raise ValueError("algorithm must be 'dqn', 'qr' or 'iqn'")

    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer(memory_size)

    returns = []           # episodic returns
    steps_total = 0
    eval_returns = []      # for plotting (every 50 episodes)

    epsilon = epsilon_start
    device = next(online_net.parameters()).device

    for episode in tqdm(range(num_episodes), desc=desc):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        done = False

        while not done:
            # ====================== ACTION SELECTION ======================
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).unsqueeze(0).to(device)

                    if algorithm == "iqn":
                        # IQN needs multiple τ samples to estimate mean Q
                        num_eval_tau = 32
                        tau = torch.rand(num_eval_tau, 1, device=device)
                        state_rep = state_t.repeat(num_eval_tau, 1)
                        q_all = online_net(state_rep, tau)          # (num_eval_tau, action_dim)
                        mean_q = q_all.mean(dim=0)                  # (action_dim,)
                        action = mean_q.argmax().item()

                    elif algorithm == "qr":
                        q = online_net(state_t).mean(dim=2).squeeze(0)
                        action = q.argmax().item()

                    else:  # dqn
                        q = online_net(state_t).squeeze(0)
                        action = q.argmax().item()

            # ====================== STEP ENVIRONMENT ======================
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps_total += 1

            # ====================== TRAINING ======================
            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                batch_states = torch.from_numpy(batch[0]).to(device)
                batch_actions = torch.from_numpy(batch[1]).to(device)
                batch_rewards = torch.from_numpy(batch[2]).to(device)
                batch_next = torch.from_numpy(batch[3]).to(device)
                batch_dones = torch.from_numpy(batch[4]).to(device)

                training_step(online_net, target_net, optimizer,
                              batch_states, batch_actions, batch_rewards,
                              batch_next, batch_dones, gamma)

            if steps_total % target_update == 0:
                target_net.load_state_dict(online_net.state_dict())

        returns.append(episode_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Record for plotting
        if (episode + 1) % 50 == 0:
            avg = np.mean(returns[-50:])
            eval_returns.append(avg)

    env.close()
    return eval_returns

# ================================================================
# 5. Run both and compare
# ================================================================
if __name__ == "__main__":
    print("Starting mini project: DQN vs QR-DQN on CartPole-v1")
    print("This will take ~2-4 minutes on a normal laptop.\n")

    # dqn_returns = train_agent(is_distributional=False, num_episodes=800)
    # qr_returns = train_agent(is_distributional=True, num_episodes=800)

    # DQN settings (original)
    iqn_returns = train_agent(algorithm="iqn", num_episodes=800)

    dqn_returns = train_agent(algorithm="dqn", num_episodes=800)
    qr_returns  = train_agent(algorithm="qr",  num_episodes=800)

    # --------------------- Plot (similar style to your attached image) ---------------------
    plt.figure(figsize=(10, 6))
    x = np.arange(len(dqn_returns)) * 50   # every 50 episodes

    plt.plot(x, dqn_returns, label='DQN', linewidth=2, color='#1f77b4')
    plt.plot(x, qr_returns, label='QR-DQN (Distributional)', linewidth=2, color='#d62728')

    plt.xlabel('Episodes')
    plt.ylabel('Average Return (last 50 episodes)')
    plt.title('DQN vs QR-DQN on CartPole-v1\n( Distributional RL learns faster & more stably )')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dqn_vs_qr_dqn_cartpole.png', dpi=200)
    plt.show()

    # --------------------- Metrics ---------------------
    print("\n=== FINAL METRICS ===")
    print(f"DQN  - Final avg return (last 100 eps): {np.mean(dqn_returns[-2:]):.1f} ± {np.std(dqn_returns[-2:]):.1f}")
    print(f"QR-DQN - Final avg return (last 100 eps): {np.mean(qr_returns[-2:]):.1f} ± {np.std(qr_returns[-2:]):.1f}")
    print(f"QR-DQN advantage: {np.mean(qr_returns[-2:]) - np.mean(dqn_returns[-2:]):+.1f} points")

    solved_dqn = next((i*50 for i, v in enumerate(dqn_returns) if v >= 475), "Not solved")
    solved_qr = next((i*50 for i, v in enumerate(qr_returns) if v >= 475), "Not solved")
    print(f"Episodes to solve (≥475): DQN = {solved_dqn} | QR-DQN = {solved_qr}")