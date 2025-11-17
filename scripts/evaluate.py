#!/usr/bin/env python3
"""
Evaluation script for trained agent.
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
from src.evaluation.evaluator import Evaluator, create_evaluation_variations
from src.utils.config import load_config
from src.utils.plotting import plot_evaluation_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of episodes per variation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for evaluation results')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create environment
    env = RobotEnv(**config.env)
    
    # Create agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        **config.agent
    )
    
    # Load trained model
    print(f"Loading model from {args.model}...")
    agent.load(args.model)
    agent.actor.eval()
    agent.critic.eval()
    
    # Create evaluator
    evaluator = Evaluator(
        env=env,
        agent=agent,
        num_episodes=args.num_episodes,
        deterministic=True,
    )
    
    # Evaluate on default configuration
    print("Evaluating on default configuration...")
    default_metrics = evaluator.evaluate(seed=args.seed)
    print(f"Success rate: {default_metrics['success_rate']:.3f}")
    print(f"Mean reward: {default_metrics['mean_reward']:.2f} ± {default_metrics['std_reward']:.2f}")
    print(f"Mean length: {default_metrics['mean_length']:.1f} ± {default_metrics['std_length']:.1f}")
    
    # Evaluate on variations
    print("\nEvaluating on task variations...")
    variations = create_evaluation_variations()
    results = {'default': default_metrics}
    
    for variation in variations:
        name = variation['name']
        var_config = variation['config']
        
        # Create environment with variation config
        var_env = RobotEnv(**{**config.env, **var_config})
        var_evaluator = Evaluator(
            env=var_env,
            agent=agent,
            num_episodes=args.num_episodes,
            deterministic=True,
        )
        
        metrics = var_evaluator.evaluate(seed=args.seed)
        results[name] = metrics
        
        print(f"\n{name}:")
        print(f"  Success rate: {metrics['success_rate']:.3f}")
        print(f"  Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    # Plot results
    if args.output:
        plot_evaluation_results(results, output_file=args.output, metric='success_rate')
    else:
        plot_evaluation_results(results, metric='success_rate')
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

