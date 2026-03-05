import numpy as np
from src.reactor_env import ReactorEnv
from src.sarsa_agent import SARSAAgent
from src.qlearning_agent import QLearningAgent

def train_sarsa(env, agent, n_episodes=2000):
    episode_returns = []
    episode_lengths = []
    for episode in range(n_episodes):
        state = env.reset()
        action = agent.get_action(state)
        total_return = 0
        timestep = 0
        while True:
            next_state, reward, done, info = env.step(action)
            next_action = agent.get_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            total_return += reward
            timestep += 1
            state = next_state
            action = next_action
            if done:
                break
        episode_returns.append(total_return)
        episode_lengths.append(timestep)
        if (episode + 1) % 200 == 0:
            avg = np.mean(episode_returns[-100:])
            print(f"Episode {episode+1}: Return={avg:.2f}, Length={np.mean(episode_lengths[-100:]):.1f}")
    return episode_returns

def train_qlearning(env, agent, n_episodes=2000):
    episode_returns = []
    episode_lengths = []
    for episode in range(n_episodes):
        state = env.reset()
        total_return = 0
        timestep = 0
        while True:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_return += reward
            timestep += 1
            state = next_state
            if done:
                break
        episode_returns.append(total_return)
        episode_lengths.append(timestep)
        if (episode + 1) % 200 == 0:
            avg = np.mean(episode_returns[-100:])
            print(f"Episode {episode+1}: Return={avg:.2f}, Length={np.mean(episode_lengths[-100:]):.1f}")
    return episode_returns

if __name__ == "__main__":
    SIGMA = 4.0
    print("="*50)
    print(f"TRAINING WITH HIGH NOISE (sigma={SIGMA})")
    print("="*50)

    env = ReactorEnv(n_bins=20, k=2, sigma=SIGMA)
    n_states, n_actions = env.get_state_action_space()

    print("\n--- SARSA ---")
    sarsa = SARSAAgent(n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1)
    sarsa_returns = train_sarsa(env, sarsa, n_episodes=2000)

    print("\n--- Q-Learning ---")
    qlearn = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1)
    qlearn_returns = train_qlearning(env, qlearn, n_episodes=2000)

    np.save('results/sarsa_returns_sigma4.npy', sarsa_returns)
    np.save('results/qlearning_returns_sigma4.npy', qlearn_returns)
    np.save('results/sarsa_q_table_sigma4.npy', sarsa.Q)
    np.save('results/qlearning_q_table_sigma4.npy', qlearn.Q)

    print("\n" + "="*50)
    print("DONE!")
    print(f"SARSA final avg return:      {np.mean(sarsa_returns[-100:]):.2f}")
    print(f"Q-Learning final avg return: {np.mean(qlearn_returns[-100:]):.2f}")