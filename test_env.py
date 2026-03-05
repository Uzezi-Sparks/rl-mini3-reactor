# test_env.py

import numpy as np
from src.reactor_env import ReactorEnv

# Create environment
env = ReactorEnv(n_bins=20, k=2)

print("=== REACTOR ENVIRONMENT TEST ===\n")
print(f"State space: {env.n_bins} bins")
print(f"Action space: {env.n_actions} actions {env.actions}")
print(f"Sweet spot: [{env.mu_lo}, {env.mu_hi}]")
print(f"Meltdown at: {env.mu_max}\n")

# Run one episode
obs = env.reset()
print(f"Initial obs (bin): {obs}")
print(f"True mu: {env.mu:.2f}\n")

total_reward = 0
for t in range(10):
    # Take random action
    action = np.random.randint(env.n_actions)
    a_value = env.actions[action]
    
    obs_next, reward, done, info = env.step(action)
    
    print(f"Step {t+1}: action={a_value:+d}, mu={info['mu']:.2f}, z={info['z']:.2f}, obs={obs_next}, reward={reward:.2f}")
    
    total_reward += reward
    obs = obs_next
    
    if done:
        if info['meltdown']:
            print("\n💥 MELTDOWN!")
        else:
            print("\n✅ Episode completed safely")
        break

print(f"\nTotal reward: {total_reward:.2f}")