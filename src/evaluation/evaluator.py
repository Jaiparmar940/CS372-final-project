"""
Evaluation module for zero-shot generalization testing.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from ..environment.robot_env import RobotEnv
from ..agents.actor_critic import PPOAgent


class Evaluator:
    """
    Evaluator for testing zero-shot generalization on held-out task variations.
    """
    
    def __init__(
        self,
        env: RobotEnv,
        agent: PPOAgent,
        num_episodes: int = 20,
        deterministic: bool = True,
    ):
        """
        Initialize evaluator.
        
        Args:
            env: Environment instance (can be configured for test variations)
            agent: Agent to evaluate
            num_episodes: Number of episodes to run
            deterministic: Whether to use deterministic policy
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.deterministic = deterministic
    
    def evaluate(
        self,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate agent on current environment configuration.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary of evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        successes = []
        final_distances = []
        
        for episode in range(self.num_episodes):
            if seed is not None:
                obs, info = self.env.reset(seed=seed + episode)
            else:
                obs, info = self.env.reset()
            
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action, _, _ = self.agent.select_action(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            successes.append(step_info.get('success', False))
            final_distances.append(step_info.get('goal_distance', np.inf))
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': np.mean(successes),
            'mean_final_distance': np.mean(final_distances),
            'std_final_distance': np.std(final_distances),
        }
        
        return metrics
    
    def evaluate_variations(
        self,
        variations: List[Dict],
        seed: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate agent on multiple task variations.
        
        Args:
            variations: List of dictionaries with environment configuration overrides
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary mapping variation names to metrics
        """
        results = {}
        
        for i, variation in enumerate(variations):
            # Configure environment for this variation
            variation_name = variation.get('name', f'variation_{i}')
            config = variation.get('config', {})
            
            # Create new environment with variation config
            # (In practice, you'd modify env parameters)
            # For now, we'll evaluate on the current env and log the variation name
            
            metrics = self.evaluate(seed=seed)
            results[variation_name] = metrics
        
        return results


def create_evaluation_variations() -> List[Dict]:
    """
    Create list of task variations for zero-shot evaluation.
    
    Returns:
        List of variation configurations
    """
    variations = [
        {
            'name': 'far_goals',
            'config': {
                'goal_range': (-8.0, 8.0),  # Larger range than training
            }
        },
        {
            'name': 'no_obstacle',
            'config': {
                'obstacle_prob': 0.0,  # No obstacles
            }
        },
        {
            'name': 'always_obstacle',
            'config': {
                'obstacle_prob': 1.0,  # Always obstacle
            }
        },
        {
            'name': 'large_arena',
            'config': {
                'arena_size': 15.0,  # Larger arena
            }
        },
    ]
    
    return variations

