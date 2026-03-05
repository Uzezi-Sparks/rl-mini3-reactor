# train_sarsa.py

import numpy as np
import matplotlib.pyplot as plt
from src.reactor_env import ReactorEnv
from src.sarsa_agent import SARSAAgent

def train_sarsa(env, agent, n_episodes=2000):
    """
    Train SARSA agent
    
    Returns:
        episode_returns: List of total returns per episode
        episode_lengths: List of episode lengths
    """
    episode_returns = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        # Reset environment
        state = env.reset()
        
        # Choose initial action
        action = agent.get_action(state)
        
        total_return = 0
        timestep = 0
        
        while True:
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Choose next action (needed for SARSA!)
            next_action = agent.get_action(next_state)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, next_action, done)
            
            total_return += reward
            timestep += 1
            
            # Move to next state
            state = next_state
            action = next_action
            
            if done:
                break
        
        episode_returns.append(total_return)
        episode_lengths.append(timestep)
        
        # Print progress
        if (episode + 1) % 200 == 0:
            avg_return = np.mean(episode_returns[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode+1}/{n_episodes}: Avg Return = {avg_return:.2f}, Avg Length = {avg_length:.1f}")
    
    return episode_returns, episode_lengths

# Run training
if __name__ == "__main__":
    # Create environment
    env = ReactorEnv(n_bins=20, k=2, sigma=2.0)
    n_states, n_actions = env.get_state_action_space()
    
    # Create SARSA agent
    agent = SARSAAgent(n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1)
    
    print("=== TRAINING SARSA ===")
    print(f"States: {n_states}, Actions: {n_actions}")
    print(f"Alpha: {agent.alpha}, Gamma: {agent.gamma}, Epsilon: {agent.epsilon}\n")
    
    # Train
    returns, lengths = train_sarsa(env, agent, n_episodes=2000)
    
    # Plot learning curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(returns, alpha=0.3, label='Raw')
    plt.plot(np.convolve(returns, np.ones(100)/100, mode='valid'), label='Moving Avg (100)')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('SARSA Learning Curve')
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
    plt.savefig('results/sarsa_learning_curve.png', dpi=150)
    print("\n✅ Saved: results/sarsa_learning_curve.png")
    
    # Save Q-table
    np.save('results/sarsa_q_table.npy', agent.Q)
    print("✅ Saved: results/sarsa_q_table.npy")

    # Save returns for comparison
    np.save('results/sarsa_returns_sigma2.npy', returns)
    print("✅ Saved: results/sarsa_returns_sigma2.npy")



