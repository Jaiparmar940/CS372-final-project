"""
Basic tests for environment.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment.robot_env import RobotEnv
from src.environment.discrete_env import DiscreteGridEnv


def test_robot_env():
    """Test robot environment basic functionality."""
    env = RobotEnv(arena_size=10.0, max_steps=100)
    
    # Test reset
    obs, info = env.reset(seed=42)
    assert obs.shape == (12,), f"Expected obs shape (12,), got {obs.shape}"
    assert 'goal_pos' in info
    
    # Test step
    action = np.array([0.5, 0.5])
    obs, reward, terminated, truncated, step_info = env.step(action)
    assert obs.shape == (12,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    
    print("✓ Robot environment tests passed")


def test_discrete_env():
    """Test discrete environment basic functionality."""
    env = DiscreteGridEnv(grid_size=8, max_steps=100)
    
    # Test reset
    state, info = env.reset(seed=42)
    assert isinstance(state, (int, np.integer))
    assert state >= 0 and state < 64  # 8x8 grid
    
    # Test step
    action = 0  # up
    state, reward, terminated, truncated, step_info = env.step(action)
    assert isinstance(state, (int, np.integer))
    assert isinstance(reward, (float, np.floating))
    
    print("✓ Discrete environment tests passed")


if __name__ == '__main__':
    test_robot_env()
    test_discrete_env()
    print("\nAll tests passed!")

