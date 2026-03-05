import numpy as np
import matplotlib.pyplot as plt

# Load all results
sarsa_s2    = np.load('results/sarsa_returns_sigma2.npy')
qlearn_s2   = np.load('results/qlearning_returns_sigma2.npy')
sarsa_s4    = np.load('results/sarsa_returns_sigma4.npy')
qlearn_s4   = np.load('results/qlearning_returns_sigma4.npy')
sarsa_q_s2  = np.load('results/sarsa_q_table.npy')
qlearn_q_s2 = np.load('results/qlearning_q_table.npy')
sarsa_q_s4  = np.load('results/sarsa_q_table_sigma4.npy')
qlearn_q_s4 = np.load('results/qlearning_q_table_sigma4.npy')

def smooth(x, w=100):
    return np.convolve(x, np.ones(w)/w, mode='valid')

# Create main comparison figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Reactor Control: RL Algorithm Comparison', fontsize=16, fontweight='bold')

# Plot 1: Learning curves sigma=2.0
ax = axes[0, 0]
ax.plot(smooth(sarsa_s2),  label='SARSA',      linewidth=2, color='blue')
ax.plot(smooth(qlearn_s2), label='Q-Learning', linewidth=2, color='red')
ax.set_title('Learning Curves — Low Noise (σ=2.0)')
ax.set_xlabel('Episode')
ax.set_ylabel('Avg Return (100-ep window)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Learning curves sigma=4.0
ax = axes[0, 1]
ax.plot(smooth(sarsa_s4),  label='SARSA',      linewidth=2, color='blue')
ax.plot(smooth(qlearn_s4), label='Q-Learning', linewidth=2, color='red')
ax.set_title('Learning Curves — High Noise (σ=4.0)')
ax.set_xlabel('Episode')
ax.set_ylabel('Avg Return (100-ep window)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Q-table heatmap SARSA
ax = axes[1, 0]
im = ax.imshow(sarsa_q_s2, aspect='auto', cmap='viridis')
plt.colorbar(im, ax=ax, label='Q-value')
ax.set_title('SARSA Q-Table (σ=2.0)')
ax.set_xlabel('Action')
ax.set_ylabel('State (bin)')
ax.set_xticks(range(5))
ax.set_xticklabels(['-2', '-1', '0', '+1', '+2'])

# Plot 4: Q-table heatmap Q-Learning
ax = axes[1, 1]
im = ax.imshow(qlearn_q_s2, aspect='auto', cmap='viridis')
plt.colorbar(im, ax=ax, label='Q-value')
ax.set_title('Q-Learning Q-Table (σ=2.0)')
ax.set_xlabel('Action')
ax.set_ylabel('State (bin)')
ax.set_xticks(range(5))
ax.set_xticklabels(['-2', '-1', '0', '+1', '+2'])

plt.tight_layout()
plt.savefig('results/final_comparison.png', dpi=150)
print("✅ Saved: results/final_comparison.png")

# Noise sensitivity bar chart
fig2, ax = plt.subplots(figsize=(7, 5))
algos = ['SARSA', 'Q-Learning']
low  = [np.mean(sarsa_s2[-100:]), np.mean(qlearn_s2[-100:])]
high = [np.mean(sarsa_s4[-100:]), np.mean(qlearn_s4[-100:])]
x = np.arange(2)
bars1 = ax.bar(x - 0.2, low,  0.35, label='σ=2.0', color='steelblue')
bars2 = ax.bar(x + 0.2, high, 0.35, label='σ=4.0', color='coral')
ax.set_title('Noise Sensitivity: Final Performance', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Return (last 100 episodes)')
ax.set_xticks(x)
ax.set_xticklabels(algos, fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for b in list(bars1) + list(bars2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
            f'{b.get_height():.0f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('results/noise_sensitivity.png', dpi=150)
print("✅ Saved: results/noise_sensitivity.png")