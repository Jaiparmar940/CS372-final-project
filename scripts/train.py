#!/usr/bin/env python3
"""
Training script for PPO agent.
"""

import sys
import os
import argparse
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment.robot_env import RobotEnv
from src.agents.actor_critic import PPOAgent
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.utils.config import load_config
from src.utils.logging import Logger


def create_eval_env(config):
    """Create evaluation environment with held-out task variations."""
    eval_config = config.env.copy()
    # Use different goal range for evaluation
    eval_config['goal_range'] = [-6.0, 6.0]  # Wider range than training
    
    return RobotEnv(**eval_config)


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create environment
    env = RobotEnv(**config.env)
    
    # Create agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        **config.agent
    )
    
    # Create logger
    logger = Logger(
        log_dir=config.logging['log_dir'],
        log_file=config.logging['log_file'],
        use_csv=True,
    )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        rollout_length=config.training['rollout_length'],
        eval_frequency=config.training['eval_frequency'],
        eval_episodes=config.training['eval_episodes'],
        save_frequency=config.training['save_frequency'],
        log_dir=config.logging['log_dir'],
        logger=logger.log,
    )
    
    # Create evaluation callback
    eval_env = create_eval_env(config)
    evaluator = Evaluator(
        env=eval_env,
        agent=agent,
        num_episodes=config.training['eval_episodes'],
        deterministic=True,
    )
    
    def eval_callback(agent, step):
        """Evaluation callback."""
        metrics = evaluator.evaluate(seed=config.seed)
        metrics['eval_step'] = step
        print(f"Evaluation at step {step}:")
        print(f"  Success rate: {metrics['success_rate']:.3f}")
        print(f"  Mean reward: {metrics['mean_reward']:.2f}")
        return metrics
    
    # Train
    trainer.train(
        total_steps=config.training['total_steps'],
        eval_callback=eval_callback,
    )
    
    # Close logger
    logger.close()
    
    print("Training complete!")


if __name__ == '__main__':
    main()

