# src/reactor_env.py

import numpy as np

class ReactorEnv:
    """
    Nuclear Reactor Control Environment
    
    The agent controls cadmium rods to manage reactor temperature.
    Challenge: True reactivity (mu) is hidden, only noisy observations available.
    """
    
    def __init__(
        self,
        n_bins=20,          # Number of discrete observation bins
        k=2,                # Max rod movement per step
        mu_min=0.0,         # Min reactivity
        mu_max=100.0,       # Meltdown threshold
        mu_lo=40.0,         # Sweet spot lower
        mu_hi=80.0,         # Sweet spot upper
        mu_hot=60.0,        # Drift activation threshold
        alpha=2.0,          # Rod effectiveness
        delta=0.5,          # Drift rate when hot
        sigma=2.0,          # Sensor noise std dev
        sigma_t=1.0,        # Process noise std dev
        sigma_r=1.0,        # Reward noise std dev
        c=0.1,              # Rod movement cost
        M=1000.0,           # Meltdown penalty
        T=100,              # Episode length
        gamma=0.95          # Discount factor
    ):
        """Initialize reactor parameters"""
        
        # Physical parameters
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_lo = mu_lo
        self.mu_hi = mu_hi
        self.mu_hot = mu_hot
        
        self.alpha = alpha
        self.delta = delta
        self.sigma = sigma
        self.sigma_t = sigma_t
        self.sigma_r = sigma_r
        
        # Reward parameters
        self.c = c
        self.M = M
        
        # Episode parameters
        self.T = T
        self.gamma = gamma
        self.timestep = 0
        
        # Action space: {-k, ..., -1, 0, +1, ..., +k}
        self.k = k
        self.actions = list(range(-k, k+1))
        self.n_actions = len(self.actions)
        
        # State space: discretized observations
        self.n_bins = n_bins
        self.bin_edges = np.linspace(mu_min, mu_max, n_bins + 1)
        
        # Current state (hidden true reactivity)
        self.mu = None
        
    def reset(self):
        """
        Start new episode.
        
        Returns:
            obs: Initial observation (binned noisy sensor reading)
        """
        # Start cold (near mu_min) with small randomness
        self.mu = self.mu_min + np.random.uniform(0, 5)
        self.timestep = 0
        
        # Get noisy observation
        z = self._get_noisy_observation()
        obs = self._discretize_observation(z)
        
        return obs
    
    def step(self, action):
        """
        Take action and advance one timestep.
        
        Args:
            action: Integer index into self.actions
            
        Returns:
            obs: Next observation (binned)
            reward: Noisy reward sample
            done: Whether episode ended
            info: Dict with diagnostics (true mu, observation)
        """
        # Convert action index to rod movement
        a = self.actions[action]
        
        # Check for meltdown BEFORE taking action
        if self.mu >= self.mu_max:
            # Already melted down!
            obs = self.n_bins - 1  # Max bin
            reward = -self.M
            done = True
            info = {'mu': self.mu, 'z': self.mu, 'meltdown': True}
            return obs, reward, done, info
        
        # Update true reactivity using physics
        self.mu = self._update_reactivity(self.mu, a)
        
        # Get noisy observation
        z = self._get_noisy_observation()
        obs = self._discretize_observation(z)
        
        # Compute noisy reward
        reward = self._get_noisy_reward(self.mu, a)
        
        # Check termination
        self.timestep += 1
        done = (self.mu >= self.mu_max) or (self.timestep >= self.T)
        
        info = {
            'mu': self.mu,
            'z': z,
            'meltdown': self.mu >= self.mu_max
        }
        
        return obs, reward, done, info
    
    def _update_reactivity(self, mu, a):
        """
        Update true reactivity based on action and physics.
        
        Formula: mu_{t+1} = clip(mu_t - alpha*a + d(mu_t) + epsilon_t)
        
        Args:
            mu: Current true reactivity
            a: Rod action (positive = insert = cool down)
            
        Returns:
            mu_next: Updated reactivity
        """
        # Action effect: inserting rods (a > 0) decreases mu
        action_effect = -self.alpha * a
        
        # Intrinsic drift: reactor heats up when hot
        drift = self.delta if mu >= self.mu_hot else 0.0
        
        # Process noise
        process_noise = np.random.normal(0, self.sigma_t)
        
        # Update and clip to bounds
        mu_next = mu + action_effect + drift + process_noise
        mu_next = np.clip(mu_next, self.mu_min, self.mu_max)
        
        return mu_next
    
    def _get_noisy_observation(self):
        """
        Sample noisy sensor reading.
        
        Returns:
            z: Noisy observation ~ N(mu, sigma^2)
        """
        z = self.mu + np.random.normal(0, self.sigma)
        return z
    
    def _discretize_observation(self, z):
        """
        Convert continuous observation to discrete bin.
        
        Args:
            z: Continuous observation
            
        Returns:
            obs: Integer bin index in [0, n_bins-1]
        """
        # Find which bin z falls into
        obs = np.digitize(z, self.bin_edges) - 1
        
        # Clip to valid range
        obs = np.clip(obs, 0, self.n_bins - 1)
        
        return obs
    
    def _get_noisy_reward(self, mu, a):
        """
        Sample noisy reward.
        
        Formula:
            R ~ N(R(mu, a), sigma_r^2)
        where
            R(mu, a) = w(mu) - c|a|  if mu in [mu_lo, mu_hi]
                     = -c|a|         if mu < mu_lo
                     = -M            if mu >= mu_max
        
        Args:
            mu: True reactivity
            a: Action taken
            
        Returns:
            reward: Noisy reward sample
        """
        # Compute expected reward
        if mu >= self.mu_max:
            # Meltdown!
            expected_reward = -self.M
        elif mu < self.mu_lo:
            # Too cold, no power
            expected_reward = -self.c * abs(a)
        elif mu <= self.mu_hi:
            # Sweet spot! Generate power
            # w(mu) = mu - mu_lo (linear power function)
            power = mu - self.mu_lo
            expected_reward = power - self.c * abs(a)
        else:
            # Above sweet spot but not meltdown yet
            # No power, just costs
            expected_reward = -self.c * abs(a)
        
        # Add reward noise
        noisy_reward = expected_reward + np.random.normal(0, self.sigma_r)
        
        return noisy_reward
    
    def get_state_action_space(self):
        """Return state and action space sizes"""
        return self.n_bins, self.n_actions