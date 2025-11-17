"""
Training loop for RL agents.
"""

import numpy as np
import torch
from typing import Dict, Optional, Callable
import os

from ..agents.actor_critic import PPOAgent
from ..environment.robot_env import RobotEnv
from .buffer import TrajectoryBuffer


class Trainer:
    """
    Trainer for PPO agent with task distribution sampling.
    """
    
    def __init__(
        self,
        env: RobotEnv,
        agent: PPOAgent,
        rollout_length: int = 2048,
        eval_frequency: int = 10000,
        eval_episodes: int = 10,
        save_frequency: int = 50000,
        log_dir: str = "results",
        logger: Optional[Callable] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            env: Environment instance
            agent: PPO agent
            rollout_length: Number of steps to collect before update
            eval_frequency: Frequency of evaluation (in steps)
            eval_episodes: Number of episodes for evaluation
            save_frequency: Frequency of checkpoint saving (in steps)
            log_dir: Directory for saving logs and checkpoints
            logger: Optional logging function
        """
        self.env = env
        self.agent = agent
        self.rollout_length = rollout_length
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.save_frequency = save_frequency
        self.log_dir = log_dir
        self.logger = logger
        
        self.buffer = TrajectoryBuffer(capacity=rollout_length)
        
        # Training statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
    
    def collect_rollout(self) -> Dict[str, np.ndarray]:
        """
        Collect a rollout of experiences.
        
        Returns:
            Dictionary with trajectory data
        """
        self.buffer.reset()
        
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        while not self.buffer.is_full():
            # Select action
            action, log_prob, value = self.agent.select_action(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.buffer.add(obs, action, reward, log_prob, value, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            if done:
                # Episode ended, next value is 0
                next_value = 0.0
                
                # Store episode statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_successes.append(step_info.get('success', False))
                self.total_episodes += 1
                
                # Log episode
                if self.logger:
                    self.logger({
                        'episode': self.total_episodes,
                        'reward': episode_reward,
                        'length': episode_length,
                        'success': step_info.get('success', False),
                        'total_steps': self.total_steps,
                    })
                
                # Reset for next episode
                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
        
        # Get final value estimate for bootstrap (if buffer not full, episode may still be ongoing)
        if not self.buffer.is_full():
            _, _, final_value = self.agent.select_action(obs, deterministic=True)
        else:
            final_value = 0.0
        
        return self.buffer.get_batch(), final_value
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step (collect rollout and update).
        
        Returns:
            Dictionary of training metrics
        """
        # Collect rollout
        batch, final_value = self.collect_rollout()
        
        # Compute advantages and returns
        advantages, returns = self.agent.compute_gae(
            rewards=batch['rewards'],
            values=batch['values'],
            dones=batch['dones'],
            last_value=final_value
        )
        
        # Update agent
        metrics = self.agent.update(
            states=batch['states'],
            actions=batch['actions'],
            old_log_probs=batch['log_probs'],
            advantages=advantages,
            returns=returns
        )
        
        # Add training statistics
        if len(self.episode_rewards) > 0:
            metrics['mean_episode_reward'] = np.mean(self.episode_rewards[-100:])
            metrics['mean_episode_length'] = np.mean(self.episode_lengths[-100:])
            # Compute success rate from recent episodes
            if len(self.episode_successes) > 0:
                metrics['success_rate'] = np.mean(self.episode_successes[-100:])
        
        metrics['total_steps'] = self.total_steps
        metrics['total_episodes'] = self.total_episodes
        
        return metrics
    
    def train(
        self,
        total_steps: int,
        eval_callback: Optional[Callable] = None
    ):
        """
        Main training loop.
        
        Args:
            total_steps: Total number of training steps
            eval_callback: Optional callback function for evaluation
        """
        print(f"Starting training for {total_steps} steps...")
        
        while self.total_steps < total_steps:
            # Training step
            metrics = self.train_step()
            
            # Log training metrics
            if self.logger:
                self.logger(metrics)
            
            # Evaluation
            if self.total_steps % self.eval_frequency == 0 and eval_callback:
                print(f"Evaluating at step {self.total_steps}...")
                eval_results = eval_callback(self.agent, self.total_steps)
                if self.logger:
                    self.logger(eval_results)
            
            # Save checkpoint
            if self.total_steps % self.save_frequency == 0:
                checkpoint_path = os.path.join(
                    self.log_dir,
                    f"checkpoint_step_{self.total_steps}.pt"
                )
                self.agent.save(checkpoint_path)
                print(f"Saved checkpoint at step {self.total_steps}")
        
        # Final save
        final_path = os.path.join(self.log_dir, "final_model.pt")
        self.agent.save(final_path)
        print("Training complete!")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics."""
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            'mean_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'std_episode_length': np.std(self.episode_lengths),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
        }

