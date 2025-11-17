"""
Critic network for value estimation.

Outputs state value V(s) for actor-critic methods.
"""

import torch
import torch.nn as nn
import numpy as np


class CriticNetwork(nn.Module):
    """
    Critic network that estimates state values V(s).
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
    ):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        
        self.state_dim = state_dim
        
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
        
        # Output layer (single value)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
        
        Returns:
            value: State value estimate [batch_size, 1] or [1]
        """
        value = self.network(state)
        return value
