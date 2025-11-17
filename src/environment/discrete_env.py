"""
Discrete grid environment for tabular Q-learning baseline.

Simple grid world where agent navigates to goal position.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class DiscreteGridEnv(gym.Env):
    """
    Discrete grid world environment for Q-learning.
    
    State: grid position (x, y)
    Action: up, down, left, right, stay
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 100,
        goal_pos: Optional[Tuple[int, int]] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize discrete grid environment.
        
        Args:
            grid_size: Size of the square grid
            max_steps: Maximum steps per episode
            goal_pos: Fixed goal position (None = random each episode)
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.fixed_goal_pos = goal_pos
        self.render_mode = render_mode
        
        # State space: grid position (x, y)
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        
        # Action space: 5 actions (up, down, left, right, stay)
        self.action_space = spaces.Discrete(5)
        
        # Action mappings: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.action_to_delta = {
            0: (0, -1),   # up
            1: (0, 1),    # down
            2: (-1, 0),   # left
            3: (1, 0),    # right
            4: (0, 0),    # stay
        }
        
        # Internal state
        self.agent_pos = None
        self.goal_pos = None
        self.step_count = 0
        self.rng = None
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (x, y) position to state index."""
        x, y = pos
        return y * self.grid_size + x
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to (x, y) position."""
        x = state % self.grid_size
        y = state // self.grid_size
        return (x, y)
    
    def _sample_goal(self) -> Tuple[int, int]:
        """Sample a random goal position."""
        goal_x = self.rng.integers(0, self.grid_size)
        goal_y = self.rng.integers(0, self.grid_size)
        # Ensure goal is not at agent starting position
        while (goal_x, goal_y) == (0, 0):
            goal_x = self.rng.integers(0, self.grid_size)
            goal_y = self.rng.integers(0, self.grid_size)
        return (goal_x, goal_y)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        
        # Agent starts at (0, 0)
        self.agent_pos = (0, 0)
        
        # Set goal position
        if self.fixed_goal_pos is not None:
            self.goal_pos = self.fixed_goal_pos
        else:
            self.goal_pos = self._sample_goal()
        
        self.step_count = 0
        
        state = self._pos_to_state(self.agent_pos)
        info = {
            "goal_pos": self.goal_pos,
            "agent_pos": self.agent_pos,
        }
        
        return state, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        # Get movement delta
        dx, dy = self.action_to_delta[action]
        
        # Update position with boundary checking
        new_x = np.clip(self.agent_pos[0] + dx, 0, self.grid_size - 1)
        new_y = np.clip(self.agent_pos[1] + dy, 0, self.grid_size - 1)
        self.agent_pos = (new_x, new_y)
        
        self.step_count += 1
        
        # Compute reward
        if self.agent_pos == self.goal_pos:
            reward = 10.0  # Success reward
            terminated = True
        else:
            # Distance-based reward (negative)
            dist = abs(self.agent_pos[0] - self.goal_pos[0]) + \
                   abs(self.agent_pos[1] - self.goal_pos[1])
            reward = -0.1 * dist - 0.01  # Small time penalty
            terminated = False
        
        truncated = self.step_count >= self.max_steps
        
        state = self._pos_to_state(self.agent_pos)
        info = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "distance": abs(self.agent_pos[0] - self.goal_pos[0]) + 
                       abs(self.agent_pos[1] - self.goal_pos[1]),
        }
        
        return state, reward, terminated, truncated, info
    
    def render(self):
        """Render the grid (simple text output)."""
        if self.render_mode == "human":
            grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            # Mark goal
            grid[self.goal_pos[1]][self.goal_pos[0]] = 'G'
            # Mark agent
            grid[self.agent_pos[1]][self.agent_pos[0]] = 'A'
            
            print("\n" + "=" * (self.grid_size * 2 + 1))
            for row in grid:
                print("|" + " ".join(row) + "|")
            print("=" * (self.grid_size * 2 + 1) + "\n")
