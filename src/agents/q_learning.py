"""
Tabular Q-learning agent for discrete environments.
"""

import numpy as np
from typing import Tuple, Optional


class QLearningAgent:
    """
    Tabular Q-learning agent with epsilon-greedy exploration.
    
    Works only with discrete state and action spaces.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """
        Initialize Q-learning agent.
        
        Args:
            state_size: Number of discrete states
            action_size: Number of discrete actions
            learning_rate: Q-learning learning rate (alpha)
            discount: Discount factor (gamma)
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_end: Final epsilon value
            epsilon_decay: Epsilon decay per episode
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state index
            training: If False, always use greedy policy
        
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_size)
        else:
            # Exploit: greedy action
            return int(np.argmax(self.q_table[state]))
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """
        Update Q-table using Q-learning update rule.
        
        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount * np.max(self.q_table[next_state])
        
        # Q-learning update
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.epsilon_start
    
    def get_q_values(self, state: int) -> np.ndarray:
        """Get Q-values for a given state."""
        return self.q_table[state].copy()
    
    def save(self, filepath: str):
        """Save Q-table to file."""
        np.save(filepath, self.q_table)
    
    def load(self, filepath: str):
        """Load Q-table from file."""
        self.q_table = np.load(filepath)
