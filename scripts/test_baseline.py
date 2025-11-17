#!/usr/bin/env python3
"""
Test baseline policies to verify environment works correctly.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment.robot_env import RobotEnv
from src.agents.baseline import ScriptedPolicy, RandomPolicy


def test_scripted_policy():
    """Test if scripted policy can reach goals."""
    env = RobotEnv(
        arena_size=10.0,
        max_steps=200,
        obstacle_prob=0.0,  # No obstacles
        goal_range=(-2.0, 2.0),  # Close goals
        goal_threshold=1.0,  # Larger threshold
    )
    
    policy = ScriptedPolicy(env.action_space)
    
    successes = 0
    total_rewards = []
    
    for episode in range(20):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0.0
        done = False
        
        while not done:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            if terminated:
                successes += 1
        
        total_rewards.append(episode_reward)
    
    print(f"Scripted Policy Results:")
    print(f"  Success rate: {successes/20:.2%}")
    print(f"  Mean reward: {np.mean(total_rewards):.2f}")
    print(f"  Episodes reaching goal: {successes}/20")
    
    return successes > 0


if __name__ == '__main__':
    print("Testing baseline policies...")
    print("=" * 50)
    
    if test_scripted_policy():
        print("\n✓ Scripted policy can reach goals - environment is working!")
    else:
        print("\n✗ Scripted policy failed - check environment configuration")

