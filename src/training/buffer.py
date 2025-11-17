"""
Trajectory buffer for collecting and storing experience.
"""

import numpy as np
from typing import List, Dict, Optional


class TrajectoryBuffer:
    """
    Buffer for storing trajectories (rollouts) for on-policy algorithms.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize trajectory buffer.
        
        Args:
            capacity: Maximum number of steps to store
        """
        self.capacity = capacity
        self.reset()
    
    def reset(self):
        """Reset buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.size = 0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """
        Add a single step to the buffer.
        
        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate
            done: Whether episode terminated
        """
        if self.size >= self.capacity:
            return
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.size += 1
    
    def get_batch(self) -> Dict[str, np.ndarray]:
        """
        Get all stored data as arrays.
        
        Returns:
            Dictionary with keys: states, actions, rewards, log_probs, values, dones
        """
        if self.size == 0:
            return {
                'states': np.array([]),
                'actions': np.array([]),
                'rewards': np.array([]),
                'log_probs': np.array([]),
                'values': np.array([]),
                'dones': np.array([]),
            }
        
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'dones': np.array(self.dones),
        }
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.capacity
    
    def clear(self):
        """Clear buffer (same as reset)."""
        self.reset()
