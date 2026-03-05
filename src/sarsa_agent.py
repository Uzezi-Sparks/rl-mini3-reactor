# src/sarsa_agent.py

import numpy as np

class SARSAAgent:
    """
    SARSA: On-policy TD control
    
    Update rule: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
    """
    
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        """
        Args:
            n_states: Number of discrete states
            n_actions: Number of actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate (ε-greedy)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table to zeros
        self.Q = np.zeros((n_states, n_actions))
        
    def get_action(self, state):
        """
        ε-greedy action selection
        
        Args:
            state: Current state (bin index)
            
        Returns:
            action: Action index
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: greedy action
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (SARSA needs this!)
            done: Whether episode ended
        """
        # Current Q-value
        q_current = self.Q[state, action]
        
        if done:
            # Terminal state: no future value
            q_target = reward
        else:
            # Use next action's Q-value (on-policy)
            q_target = reward + self.gamma * self.Q[next_state, next_action]
        
        # TD error
        td_error = q_target - q_current
        
        # Update Q-value
        self.Q[state, action] += self.alpha * td_error
        
        return td_error
    
    def get_greedy_action(self, state):
        """Get greedy action (for evaluation)"""
        return np.argmax(self.Q[state])