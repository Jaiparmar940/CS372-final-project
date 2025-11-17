"""
Actor network for policy gradient methods.

Outputs action distribution parameters (mean and log_std for continuous actions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ActorNetwork(nn.Module):
    """
    Actor network that outputs action distribution parameters.
    
    For continuous actions: outputs mean and log_std of Gaussian distribution.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output heads for mean and log_std
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            mean: Action mean [batch_size, action_dim]
            log_std: Log standard deviation [batch_size, action_dim]
        """
        x = self.shared_layers(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
            deterministic: If True, return mean action (no sampling)
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            # For deterministic, log_prob is not meaningful, return zeros
            log_prob = torch.zeros_like(mean[:, 0])
        else:
            # Sample from Gaussian distribution
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1)
        
        # Clip action to [-1, 1] (assuming action space is normalized)
        action = torch.tanh(action)
        
        # Adjust log_prob for tanh transformation
        if not deterministic:
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        if state.dim() == 1:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        
        return action, log_prob
    
    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability of given action.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
        
        Returns:
            log_prob: Log probability of action
            entropy: Entropy of action distribution
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        # Adjust for tanh transformation
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        entropy = normal.entropy().sum(dim=-1)
        
        return log_prob, entropy
