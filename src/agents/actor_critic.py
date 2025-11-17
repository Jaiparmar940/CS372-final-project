"""
PPO (Proximal Policy Optimization) actor-critic agent.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional

from ..networks.actor import ActorNetwork
from ..networks.critic import CriticNetwork


class PPOAgent:
    """
    PPO agent with actor-critic architecture.
    
    Implements Proximal Policy Optimization with:
    - Clipped objective
    - Generalized Advantage Estimation (GAE)
    - Multiple update epochs per batch
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        actor_hidden_dims: list = [256, 256],
        critic_hidden_dims: list = [256, 256],
        device: str = "cpu",
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of update epochs per batch
            actor_hidden_dims: Actor network hidden dimensions
            critic_hidden_dims: Critic network hidden dimensions
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.device = torch.device(device)
        
        # Create networks
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims
        ).to(self.device)
        
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_dims=critic_hidden_dims
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            deterministic: If True, use deterministic policy
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor, deterministic=deterministic)
            value = self.critic(state_tensor)
        
        action = action.cpu().numpy().flatten()
        log_prob = log_prob.cpu().item()
        value = value.cpu().item()
        
        return action, log_prob, value
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        gae = 0.0
        
        next_value = last_value
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        
        return advantages, returns
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            states: States [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            old_log_probs: Old log probabilities [batch_size]
            advantages: Advantages [batch_size]
            returns: Returns [batch_size]
        
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        # Multiple update epochs
        for epoch in range(self.update_epochs):
            # Get current policy outputs
            log_probs, entropy = self.actor.evaluate_action(states, actions)
            values = self.critic(states).squeeze(-1)
            
            # Compute ratio for PPO clipping
            ratio = torch.exp(log_probs - old_log_probs)
            
            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = nn.functional.mse_loss(values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
        
        # Average over epochs
        n_epochs = self.update_epochs
        metrics = {
            "actor_loss": total_actor_loss / n_epochs,
            "critic_loss": total_critic_loss / n_epochs,
            "entropy": total_entropy / n_epochs,
        }
        
        return metrics
    
    def save(self, filepath: str):
        """Save agent networks."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
