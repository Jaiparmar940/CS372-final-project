"""
Baseline policies: random and scripted heuristics.
"""

import numpy as np
from typing import Tuple, Optional


class RandomPolicy:
    """Random action policy baseline."""
    
    def __init__(self, action_space):
        """
        Initialize random policy.
        
        Args:
            action_space: Gymnasium action space
        """
        self.action_space = action_space
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Sample a random action.
        
        Args:
            observation: Current observation (unused)
            deterministic: If True, return mean action (for continuous spaces)
        
        Returns:
            Random action
        """
        if hasattr(self.action_space, 'sample'):
            return self.action_space.sample()
        else:
            # Fallback for discrete spaces
            return np.array([self.action_space.sample()])


class ScriptedPolicy:
    """
    Simple scripted policy that moves toward goal.
    
    For continuous environments: computes direction to goal and applies
    acceleration in that direction.
    """
    
    def __init__(self, action_space, max_action: float = 1.0):
        """
        Initialize scripted policy.
        
        Args:
            action_space: Gymnasium action space
            max_action: Maximum action magnitude
        """
        self.action_space = action_space
        self.max_action = max_action
    
    def select_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select action based on simple heuristic.
        
        Observation format: [robot_x, robot_y, robot_vx, robot_vy,
                            goal_x, goal_y, goal_dx, goal_dy, goal_distance, ...]
        
        Args:
            observation: Current observation
            deterministic: Unused (always deterministic)
        
        Returns:
            Action toward goal
        """
        # Extract relative goal position (indices 6, 7)
        goal_dx = observation[6]
        goal_dy = observation[7]
        goal_distance = observation[8]
        
        # Normalize direction to goal
        if goal_distance > 0.1:
            direction_x = goal_dx / goal_distance
            direction_y = goal_dy / goal_distance
        else:
            # Already at goal, stop
            direction_x = 0.0
            direction_y = 0.0
        
        # Apply action in direction of goal
        action = np.array([direction_x, direction_y], dtype=np.float32)
        
        # Clip to action space bounds
        if hasattr(self.action_space, 'low'):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action
