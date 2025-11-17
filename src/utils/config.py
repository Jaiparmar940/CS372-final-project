"""
Configuration management using YAML files.
"""

import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Configuration class for RL project."""
    
    # Environment config
    env: Dict[str, Any] = field(default_factory=lambda: {
        'arena_size': 10.0,
        'max_steps': 500,
        'dt': 0.1,
        'damping': 0.1,
        'goal_threshold': 0.5,
        'obstacle_prob': 0.5,
        'obstacle_radius_range': [0.5, 1.5],
        'reward_distance_scale': 1.0,
        'reward_success': 100.0,
        'reward_collision': -50.0,
        'reward_time': -0.1,
        'goal_range': [-4.0, 4.0],  # Training goal range
    })
    
    # Agent config
    agent: Dict[str, Any] = field(default_factory=lambda: {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'update_epochs': 10,
        'actor_hidden_dims': [256, 256],
        'critic_hidden_dims': [256, 256],
        'device': 'cpu',
    })
    
    # Training config
    training: Dict[str, Any] = field(default_factory=lambda: {
        'total_steps': 1000000,
        'rollout_length': 2048,
        'eval_frequency': 10000,
        'eval_episodes': 10,
        'save_frequency': 50000,
    })
    
    # Logging config
    logging: Dict[str, Any] = field(default_factory=lambda: {
        'log_dir': 'results',
        'log_file': 'training.log',
    })
    
    # Seed
    seed: int = 42


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Config object
    """
    if config_path is None:
        return Config()
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Config file {config_path} not found, using defaults")
        return Config()
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config with defaults and override with file values
    config = Config()
    
    if 'env' in config_dict:
        config.env.update(config_dict['env'])
    if 'agent' in config_dict:
        config.agent.update(config_dict['agent'])
    if 'training' in config_dict:
        config.training.update(config_dict['training'])
    if 'logging' in config_dict:
        config.logging.update(config_dict['logging'])
    if 'seed' in config_dict:
        config.seed = config_dict['seed']
    
    return config


def save_config(config: Config, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object
        config_path: Path to save config
    """
    config_dict = {
        'env': config.env,
        'agent': config.agent,
        'training': config.training,
        'logging': config.logging,
        'seed': config.seed,
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

