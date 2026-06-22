## The Big Picture

Your project trains RL agents to sell a large block of shares over a fixed time horizon, minimising implementation shortfall (the cost of trading). The key contribution is using **IQN** (a distributional RL algorithm) to control **tail risk** via CVaR — something standard RL methods like DQN cannot do.

The code follows this pipeline:

```
Environment (defines the trading game)
    → Agents (learn how to play)
        → Training (the learning loop)
            → Evaluation (measure performance)
                → Visualisation (figures for the paper)
```

---

## File-by-File Explanation

### Layer 1: The Trading Environment

**`envs/base_env.py`** — The rules of the game

This defines the MDP from Section 2 of your paper. Every trading episode follows the same structure: the agent starts with `q₀` shares, observes a 5D state, picks an action (how much to sell), receives a reward, and repeats for N periods.

Key class: **`BaseExecutionEnv`**
- `reset()` → starts a new trading day, returns initial state
- `step(action)` → executes the trade, returns (next_state, reward, done, info)
- `_compute_reward()` → your paper's Eq. 4: IS contribution minus quadratic trade penalty
- `_build_state()` → constructs the 5D normalised state vector from Table 1 of your paper

Key config: **`EnvConfig`** — holds N, q₀, p₀, η, γ, α (all the MDP parameters)

**`envs/simulated_env.py`** — Two synthetic markets for Phase 1

Inherits from `BaseExecutionEnv` and only overrides how prices move.

Key classes:
- **`AlmgrenChrissEnv`** (Phase 1A) — Gaussian price dynamics. The sanity check: all agents should perform similarly because there are no fat tails to exploit
- **`RegimeSwitchingEnv`** (Phase 1B) — Hidden Markov volatility switching between normal and stress regimes. Creates fat-tailed IS distribution. This is where IQN-CVaR should show its advantage. Links to your paper's main experimental claim

**`envs/lobster_env.py`** — Real market data for Phase 2

Replays historical LOBSTER limit order book data. Each episode = one trading day. The `_execution_price()` override walks the real LOB levels instead of using a linear impact model. Links to your paper's empirical validation section.

---

### Layer 2: The Agents

**`agents/baselines.py`** — All comparison agents

This file contains every agent your paper compares against IQN. They form the ablation ladder in your paper's Table 3:

Key classes (in paper order):
- **`TWAPAgent`** — sells equal amounts each period. The simplest baseline. No learning.
- **`AlmgrenChrissAgent`** — the analytical optimal solution from Almgren & Chriss (2001). Pre-computes the entire schedule at initialisation. The "best you can do with a model."
- **`DQNAgent`** — Mnih et al. (2015). Learns Q(s,a) with a neural network. Has overestimation bias.
- **`DDQNAgent`** — Van Hasselt et al. (2016). Fixes DQN's bias by decoupling action selection from evaluation. This is what Ning et al. (2021) use for execution.
- **`QRDQNAgent`** — Dabney et al. (2017). The bridge between scalar and distributional RL. Learns N=51 fixed quantile values. Shows distributional learning helps even without continuous τ.

The ablation story: TWAP → AC → DQN → DDQN → QR-DQN → IQN. Each step adds one capability.

Shared infrastructure: **`_DeepRLBase`** — all learned agents (DQN, DDQN, QR-DQN) share the same training loop, epsilon schedule, replay buffer, and target network update. This ensures fair comparison.

**`agents/iqn_agent.py`** — Your contribution

The central file of the paper. One class handles both risk-neutral and risk-sensitive policies.

Key class: **`IQNAgent`**
- `select_action()` — the key line is `tau_high=self.cfg.cvar_alpha`. For neutral (α=1.0), τ ~ U[0,1] optimises E[Z]. For CVaR (α=0.95), τ ~ U[0,0.95] optimises CVaR₉₅[Z]. Same weights, different τ range. This is your paper's zero-cost risk control claim.
- `_compute_loss()` — implements the quantile Huber loss from Dabney et al. (2018) Eq. 3. This is where the distributional learning happens.
- `update()` — one gradient step: sample batch → compute loss → backprop → clip gradients → update target network

Key config: **`AgentConfig`** — `cvar_alpha` is the single parameter that controls the risk profile.

---

### Layer 3: The Neural Network

**`networks/iqn_network.py`** — The IQN architecture from Dabney et al. (2018) Section 3.1

Implements `Z_τ(s, a) ≈ f(ψ(s) ⊙ φ(τ))_a`

Key classes:
- **`CosineQuantileEmbedding`** (φ) — maps scalar τ ∈ [0,1] to a 128-dim vector using cosine features. Paper Eq. 4.
- **`StateEncoder`** (ψ) — maps the 5D state to a 128-dim embedding
- **`OutputMLP`** (f) — maps the combined embedding to Q-quantile values per action
- **`IQNNetwork`** — combines all three. The `forward()` method is the complete forward pass. `get_action_values()` averages over τ samples for greedy action selection.

The element-wise product `ψ(s) ⊙ φ(τ)` is what makes IQN different from QR-DQN: τ conditions the entire computation, enabling continuous quantile queries.

---

### Layer 4: Training Infrastructure

**`training/replay_buffer.py`** — Experience replay (Mnih et al. 2015)

Key class: **`ReplayBuffer`**
- `push()` — stores one (s, a, r, s', done) transition
- `sample()` — returns a random minibatch as torch tensors
- `ready` — True once enough transitions are stored to start learning

Pre-allocated numpy arrays for O(1) sampling. ~3MB total memory.

**`training/scheduler.py`** — Epsilon and learning rate schedules

Key classes:
- **`EpsilonScheduler`** — controls exploration. High ε early (random), low ε late (greedy). The `epsilon_decay_steps` parameter is critical — too fast and agents lock into bad policies.
- **`LRScheduler`** — optional learning rate decay. Cosine warmup prevents catastrophic early updates.

**`training/trainer.py`** — The centralised training loop

Key class: **`Trainer`**
- `train()` — runs the full train-eval-checkpoint cycle for any agent+env pair
- Handles early stopping on CVaR₉₅ (your paper's key metric)
- `TrainingLog` — accumulates all data for post-hoc analysis

Key function: **`train_all()`** — trains all learned agents sequentially, skipping rule-based agents (TWAP, AC) and CVaR variants (which copy weights from IQN-neutral).

---

### Layer 5: Evaluation and Visualisation

**`evaluation/metrics.py`** — All risk metrics from your paper's Table 3

Key functions:
- **`cvar_alpha()`** — THE key metric. CVaR₉₅ = average IS in the worst 5% of episodes. This is what IQN-CVaR is designed to minimise.
- **`var_alpha()`** — Value-at-Risk, the threshold above which the worst episodes fall
- **`gain_loss_ratio()`** — upside/downside asymmetry relative to TWAP
- **`mean_is()`**, **`std_is()`**, **`max_is()`** — basic summary statistics

Key class: **`EpisodeTracker`** — accumulates per-step data during evaluation. Called by `run_simulation.py` during the evaluation phase. Produces the IS array that all metric functions consume.

Key class: **`MetricsSuite`** — wraps the full evaluate-and-compute-metrics pipeline into one call.

**`evaluation/visualizer.py`** — All paper figures

Key class: **`Visualizer`** with these methods mapping to paper figures:
- `plot_is_distributions()` → Figure 2: overlaid IS histograms. Visual proof that IQN-CVaR has thinner right tail
- `plot_mean_cvar_frontier()` → Figure 3: the Pareto scatter. IQN-CVaR should dominate lower-left (low mean AND low CVaR)
- `plot_quantile_distribution()` → Figure 6: the learned Z(s,a) curves. Shows the agent knows the full return distribution shape
- `plot_training_curves()` → Figure 4: loss and reward convergence
- `plot_inventory_trajectories()` → Figure 5: how each agent liquidates over time
- `plot_regime_comparison()` → Figure 7: box plots showing CVaR advantage concentrates in stress episodes

---

### Layer 6: Experiment Scripts

**`experiments/run_simulation.py`** — The master script that ties everything together

Key functions:
- **`build_agents()`** — constructs all 8 agents with shared hyperparameters
- **`train_agent()`** — trains one agent, returns a log dict
- **`share_iqn_weights()`** — copies IQN-neutral weights to CVaR variants. Implements the zero-cost risk control claim.
- **`evaluate_all()`** — evaluates all agents on the same episodes for fair comparison
- **`evaluate_by_regime()`** — partitions IS by normal/stress regime for Phase 1B analysis
- **`run_phase()`** — the full pipeline: build → train → share weights → evaluate → generate figures → save

---

### Configs

**`configs/sim_config.yaml`** — all environment parameters (N, q₀, η, γ, σ, regime transition probabilities)

**`configs/agent_config.yaml`** — all agent and training parameters (lr, batch size, epsilon schedule, network size, CVaR alphas)

---

## How It All Connects

```
sim_config.yaml ──→ SimConfig ──→ AlmgrenChrissEnv / RegimeSwitchingEnv
                                         │
agent_config.yaml ──→ AgentConfig ──→ IQNAgent ──→ IQNNetwork
                  ──→ DeepRLConfig ──→ DQN/DDQN/QR-DQN ──→ _QMLP/_QRNetwork
                                         │
                          run_simulation.py
                          │
                          ├─ build_agents()     → all 8 agents
                          ├─ train_agent()      → Trainer + ReplayBuffer + Scheduler
                          ├─ share_iqn_weights() → copy neutral → CVaR
                          ├─ evaluate_all()     → MetricsSuite + EpisodeTracker
                          └─ Visualizer         → all paper figures
```