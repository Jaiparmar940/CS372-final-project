"""Neural network architectures for RL agents."""

from .actor import ActorNetwork
from .critic import CriticNetwork

__all__ = ['ActorNetwork', 'CriticNetwork']
