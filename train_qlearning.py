# train_qlearning.py

import numpy as np
import matplotlib.pyplot as plt
from src.reactor_env import ReactorEnv
from src.qlearning_agent import QLearningAgent

def train_qlearning(env, agent, n_episodes=2000):
    """Train Q-learning agent"""
    episode_returns = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_return = 0
        timestep = 0
        
        while True:
            # Choose action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update Q-table (no next_action needed!)
            agent.update(state, action, reward, next_state, done)
            
            total_return += reward
            timestep += 1
            
            state = next_state
            
            if done:
                break
        
        episode_returns.append(total_return)
        episode_lengths.append(timestep)
        
        if (episode + 1) % 200 == 0:
            avg_return = np.mean(episode_returns[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode+1}/{n_episodes}: Avg Return = {avg_return:.2f}, Avg Length = {avg_length:.1f}")
    
    return episode_returns, episode_lengths

if __name__ == "__main__":
    env = ReactorEnv(n_bins=20, k=2, sigma=2.0)
    n_states, n_actions = env.get_state_action_space()
    
    agent = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1)
    
    print("=== TRAINING Q-LEARNING ===")
    print(f"States: {n_states}, Actions: {n_actions}")
    print(f"Alpha: {agent.alpha}, Gamma: {agent.gamma}, Epsilon: {agent.epsilon}\n")
    
    returns, lengths = train_qlearning(env, agent, n_episodes=2000)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(returns, alpha=0.3, label='Raw')
    plt.plot(np.convolve(returns, np.ones(100)/100, mode='valid'), label='Moving Avg (100)')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Q-Learning Learning Curve')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(lengths, alpha=0.3)
    plt.plot(np.convolve(lengths, np.ones(100)/100, mode='valid'))
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Length')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/qlearning_learning_curve.png', dpi=150)
    print("\n✅ Saved: results/qlearning_learning_curve.png")
    
    np.save('results/qlearning_q_table.npy', agent.Q)
    print("✅ Saved: results/qlearning_q_table.npy")
    
    # Save returns for comparison
    np.save('results/qlearning_returns_sigma2.npy', returns)
    print("✅ Saved: results/qlearning_returns_sigma2.npy")