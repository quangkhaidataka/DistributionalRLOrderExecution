# === QUANTILE METHODS PROJECTION (QR-DQN / IQN) ===
# No heavy projection needed!

if not done_i:
    # 1. Choose next action (greedy on mean or risk-sensitive)
    a_next = argmax over a' of mean_of(Z_target(s_next_i, a'))
    
    # 2. Get the N quantile values from target network
    next_quantiles = Z_target_network(s_next_i, a_next)   # list of N numbers
    
    # 3. Apply distributional Bellman (this is the "project")
    for k = 1 to N:
        target_quantile[k] = r_i + gamma * (1 - done_i) * next_quantiles[k]
        
else:
    # Terminal: all quantiles become exactly r_i
    for k = 1 to N:
        target_quantile[k] = r_i

# 4. Now compute quantile regression loss
sample_loss = 0
for k = 1 to N:
    error = target_quantile[k] - current_quantiles[k]   # current from online network
    if error >= 0:
        sample_loss += tau[k] * error
    else:
        sample_loss += (tau[k] - 1) * error

sample_loss = sample_loss / N


### Demonstration 
current_quantiles_all[0, 0, :]   # ← for action 0
→ [ -12.3,  -5.1,   2.4,   8.7,  15.2 ]

current_quantiles_all[0, 1, :]   # ← for action 1
→ [ -8.9,   -1.2,   4.1,  11.5,  22.0 ]