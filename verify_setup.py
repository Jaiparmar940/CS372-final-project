#!/usr/bin/env python3
"""
Quick verification script to check if all components can be imported.
"""

import sys
import os

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.environment.robot_env import RobotEnv
        from src.environment.discrete_env import DiscreteGridEnv
        print("✓ Environment modules imported")
    except Exception as e:
        print(f"✗ Environment import failed: {e}")
        return False
    
    try:
        from src.agents.baseline import RandomPolicy, ScriptedPolicy
        from src.agents.q_learning import QLearningAgent
        from src.agents.actor_critic import PPOAgent
        print("✓ Agent modules imported")
    except Exception as e:
        print(f"✗ Agent import failed: {e}")
        return False
    
    try:
        from src.networks.actor import ActorNetwork
        from src.networks.critic import CriticNetwork
        print("✓ Network modules imported")
    except Exception as e:
        print(f"✗ Network import failed: {e}")
        return False
    
    try:
        from src.training.trainer import Trainer
        from src.training.buffer import TrajectoryBuffer
        print("✓ Training modules imported")
    except Exception as e:
        print(f"✗ Training import failed: {e}")
        return False
    
    try:
        from src.evaluation.evaluator import Evaluator
        print("✓ Evaluation modules imported")
    except Exception as e:
        print(f"✗ Evaluation import failed: {e}")
        return False
    
    try:
        from src.utils.config import load_config
        from src.utils.logging import Logger
        from src.utils.plotting import plot_learning_curves
        print("✓ Utility modules imported")
    except Exception as e:
        print(f"✗ Utility import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from src.environment.robot_env import RobotEnv
        
        env = RobotEnv(arena_size=10.0, max_steps=100)
        obs, info = env.reset(seed=42)
        action = np.array([0.5, 0.5])
        obs, reward, terminated, truncated, step_info = env.step(action)
        print("✓ Robot environment works")
    except Exception as e:
        print(f"✗ Robot environment test failed: {e}")
        return False
    
    try:
        from src.environment.discrete_env import DiscreteGridEnv
        
        env = DiscreteGridEnv(grid_size=8, max_steps=100)
        state, info = env.reset(seed=42)
        state, reward, terminated, truncated, step_info = env.step(0)
        print("✓ Discrete environment works")
    except Exception as e:
        print(f"✗ Discrete environment test failed: {e}")
        return False
    
    try:
        import torch
        from src.networks.actor import ActorNetwork
        from src.networks.critic import CriticNetwork
        
        actor = ActorNetwork(state_dim=12, action_dim=2, hidden_dims=[64, 64])
        critic = CriticNetwork(state_dim=12, hidden_dims=[64, 64])
        
        state = torch.randn(1, 12)
        action, log_prob = actor.get_action(state)
        value = critic(state)
        print("✓ Networks work")
    except Exception as e:
        print(f"✗ Network test failed: {e}")
        return False
    
    return True


if __name__ == '__main__':
    print("=" * 50)
    print("RL Project Setup Verification")
    print("=" * 50)
    
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    if imports_ok and functionality_ok:
        print("✓ All tests passed! Setup is correct.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

