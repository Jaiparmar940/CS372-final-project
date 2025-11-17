"""
2D Robotic Environment with continuous states and actions.

A point robot navigates to goal positions with an optional single obstacle.
Follows Gymnasium API conventions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class RobotEnv(gym.Env):
    """
    2D point robot environment for goal navigation.
    
    State: [robot_x, robot_y, robot_vx, robot_vy, goal_x, goal_y, 
            goal_dx, goal_dy, goal_distance, obstacle_x, obstacle_y, obstacle_radius]
    Action: [acceleration_x, acceleration_y] in [-1, 1]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        arena_size: float = 10.0,
        max_steps: int = 500,
        dt: float = 0.1,
        damping: float = 0.1,
        goal_threshold: float = 0.5,
        obstacle_prob: float = 0.5,
        obstacle_radius_range: Tuple[float, float] = (0.5, 1.5),
        reward_distance_scale: float = 1.0,
        reward_success: float = 100.0,
        reward_collision: float = -50.0,
        reward_time: float = -0.1,
        goal_range: Optional[Tuple[float, float]] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the robot environment.
        
        Args:
            arena_size: Size of the square arena (centered at origin)
            max_steps: Maximum steps per episode
            dt: Time step for dynamics
            damping: Velocity damping coefficient
            goal_threshold: Distance threshold for success
            obstacle_prob: Probability of obstacle being present
            obstacle_radius_range: (min, max) radius for obstacles
            reward_distance_scale: Scale for distance-based reward
            reward_success: Reward for reaching goal
            reward_collision: Penalty for collision
            reward_time: Per-step time penalty
            goal_range: (min, max) range for goal positions (None = full arena)
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.arena_size = arena_size
        self.max_steps = max_steps
        self.dt = dt
        self.damping = damping
        self.goal_threshold = goal_threshold
        self.obstacle_prob = obstacle_prob
        self.obstacle_radius_range = obstacle_radius_range
        self.reward_distance_scale = reward_distance_scale
        self.reward_success = reward_success
        self.reward_collision = reward_collision
        self.reward_time = reward_time
        self.goal_range = goal_range or (-arena_size/2, arena_size/2)
        self.render_mode = render_mode
        
        # State space: [robot_pos(2), robot_vel(2), goal_pos(2), goal_rel(2), 
        #               goal_dist(1), obstacle_pos(2), obstacle_radius(1)]
        # Total: 12 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )
        
        # Action space: acceleration in x and y
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Internal state
        self.robot_pos = None
        self.robot_vel = None
        self.goal_pos = None
        self.obstacle_pos = None
        self.obstacle_radius = None
        self.has_obstacle = False
        self.step_count = 0
        self.rng = None
        
    def _get_observation(self) -> np.ndarray:
        """Compute current observation."""
        goal_dx = self.goal_pos[0] - self.robot_pos[0]
        goal_dy = self.goal_pos[1] - self.robot_pos[1]
        goal_distance = np.sqrt(goal_dx**2 + goal_dy**2)
        
        # If no obstacle, set obstacle features to zeros
        if not self.has_obstacle:
            obs_x, obs_y, obs_r = 0.0, 0.0, 0.0
        else:
            obs_x = self.obstacle_pos[0]
            obs_y = self.obstacle_pos[1]
            obs_r = self.obstacle_radius
        
        obs = np.array([
            self.robot_pos[0],      # robot_x
            self.robot_pos[1],      # robot_y
            self.robot_vel[0],      # robot_vx
            self.robot_vel[1],      # robot_vy
            self.goal_pos[0],       # goal_x
            self.goal_pos[1],       # goal_y
            goal_dx,                # goal_dx
            goal_dy,                # goal_dy
            goal_distance,          # goal_distance
            obs_x,                  # obstacle_x
            obs_y,                  # obstacle_y
            obs_r,                  # obstacle_radius
        ], dtype=np.float32)
        
        return obs
    
    def _sample_goal(self) -> np.ndarray:
        """Sample a random goal position."""
        goal_x = self.rng.uniform(self.goal_range[0], self.goal_range[1])
        goal_y = self.rng.uniform(self.goal_range[0], self.goal_range[1])
        return np.array([goal_x, goal_y], dtype=np.float32)
    
    def _sample_obstacle(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Sample obstacle position and radius if obstacle should be present."""
        if self.rng.random() < self.obstacle_prob:
            # Sample obstacle position (avoid center where robot starts)
            margin = 1.0
            obs_x = self.rng.uniform(
                -self.arena_size/2 + margin,
                self.arena_size/2 - margin
            )
            obs_y = self.rng.uniform(
                -self.arena_size/2 + margin,
                self.arena_size/2 - margin
            )
            obs_radius = self.rng.uniform(
                self.obstacle_radius_range[0],
                self.obstacle_radius_range[1]
            )
            return np.array([obs_x, obs_y], dtype=np.float32), obs_radius
        else:
            return None, None
    
    def _check_collision(self) -> bool:
        """Check if robot collides with obstacle."""
        if not self.has_obstacle:
            return False
        
        dist_to_obstacle = np.sqrt(
            (self.robot_pos[0] - self.obstacle_pos[0])**2 +
            (self.robot_pos[1] - self.obstacle_pos[1])**2
        )
        return dist_to_obstacle < self.obstacle_radius
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward for current state and action."""
        goal_dx = self.goal_pos[0] - self.robot_pos[0]
        goal_dy = self.goal_pos[1] - self.robot_pos[1]
        goal_distance = np.sqrt(goal_dx**2 + goal_dy**2)
        
        # Success reward
        if goal_distance < self.goal_threshold:
            return self.reward_success
        
        # Collision penalty
        if self._check_collision():
            return self.reward_collision
        
        # Distance-based reward (negative, encourages getting closer)
        distance_reward = -self.reward_distance_scale * goal_distance
        
        # Time penalty
        time_penalty = self.reward_time
        
        return distance_reward + time_penalty
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        
        # Reset robot to center
        self.robot_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_vel = np.array([0.0, 0.0], dtype=np.float32)
        
        # Sample goal
        self.goal_pos = self._sample_goal()
        
        # Sample obstacle
        self.obstacle_pos, self.obstacle_radius = self._sample_obstacle()
        self.has_obstacle = self.obstacle_pos is not None
        
        self.step_count = 0
        
        obs = self._get_observation()
        info = {
            "goal_pos": self.goal_pos.copy(),
            "has_obstacle": self.has_obstacle,
            "obstacle_pos": self.obstacle_pos.copy() if self.has_obstacle else None,
            "obstacle_radius": self.obstacle_radius if self.has_obstacle else None,
        }
        
        return obs, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment forward."""
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update velocity with damping and acceleration
        self.robot_vel = self.robot_vel * (1 - self.damping) + action * self.dt
        
        # Update position
        self.robot_pos = self.robot_pos + self.robot_vel * self.dt
        
        # Clip position to arena bounds
        self.robot_pos = np.clip(
            self.robot_pos,
            -self.arena_size/2,
            self.arena_size/2
        )
        
        # Clip velocity to prevent excessive speeds
        max_vel = 5.0
        self.robot_vel = np.clip(self.robot_vel, -max_vel, max_vel)
        
        self.step_count += 1
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check termination conditions
        goal_distance = np.sqrt(
            (self.goal_pos[0] - self.robot_pos[0])**2 +
            (self.goal_pos[1] - self.robot_pos[1])**2
        )
        terminated = bool(goal_distance < self.goal_threshold)
        truncated = bool(self.step_count >= self.max_steps)
        
        obs = self._get_observation()
        info = {
            "goal_distance": goal_distance,
            "collision": self._check_collision(),
            "success": terminated,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (placeholder for visualization)."""
        if self.render_mode == "human":
            # Could implement matplotlib visualization here
            pass
        elif self.render_mode == "rgb_array":
            # Could return RGB array for video recording
            pass
