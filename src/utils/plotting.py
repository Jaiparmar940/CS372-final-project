"""
Plotting utilities for visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def plot_learning_curves(
    log_file: str,
    output_file: Optional[str] = None,
    metrics: List[str] = None,
    window_size: int = 100,
):
    """
    Plot learning curves from training log.
    
    Args:
        log_file: Path to CSV log file
        output_file: Path to save plot (None = show)
        metrics: List of metrics to plot (None = all)
        window_size: Window size for moving average
    """
    # Load data
    df = pd.read_csv(log_file)
    
    if metrics is None:
        # Plot all numeric columns except episode/step numbers
        metrics = [col for col in df.columns 
                   if col not in ['episode', 'total_steps', 'total_episodes'] 
                   and df[col].dtype in [np.float64, np.int64]]
    
    # Create subplots
    n_metrics = len(metrics)
    if n_metrics == 0:
        print("No metrics to plot")
        return
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        
        ax = axes[i]
        values = df[metric].values
        
        # Plot raw values
        ax.plot(values, alpha=0.3, label='Raw', color='lightblue')
        
        # Plot moving average
        if len(values) >= window_size:
            window = np.ones(window_size) / window_size
            smoothed = np.convolve(values, window, mode='valid')
            x_smooth = np.arange(window_size - 1, len(values))
            ax.plot(x_smooth, smoothed, label=f'Moving Avg ({window_size})', color='blue', linewidth=2)
        
        ax.set_xlabel('Step/Episode')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_evaluation_results(
    results: Dict[str, Dict[str, float]],
    output_file: Optional[str] = None,
    metric: str = 'success_rate',
):
    """
    Plot evaluation results across task variations.
    
    Args:
        results: Dictionary mapping variation names to metrics
        output_file: Path to save plot (None = show)
        metric: Metric to plot
    """
    variations = list(results.keys())
    values = [results[v].get(metric, 0.0) for v in variations]
    errors = [results[v].get(f'std_{metric}', 0.0) if f'std_{metric}' in results[v] 
              else results[v].get('std_reward', 0.0) for v in variations]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(variations))
    bars = ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7, color='steelblue')
    
    ax.set_xlabel('Task Variation')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Across Task Variations')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variations, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_trajectory(
    states: np.ndarray,
    goal_pos: np.ndarray,
    obstacle_pos: Optional[np.ndarray] = None,
    obstacle_radius: Optional[float] = None,
    output_file: Optional[str] = None,
):
    """
    Plot robot trajectory.
    
    Args:
        states: Array of states [T, state_dim] (robot positions)
        goal_pos: Goal position [2]
        obstacle_pos: Obstacle position [2] (optional)
        obstacle_radius: Obstacle radius (optional)
        output_file: Path to save plot (None = show)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract robot positions
    robot_positions = states[:, :2]  # First two dimensions are x, y
    
    # Plot trajectory
    ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', alpha=0.5, label='Trajectory')
    ax.scatter(robot_positions[0, 0], robot_positions[0, 1], 
              color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(robot_positions[-1, 0], robot_positions[-1, 1], 
              color='red', s=100, marker='s', label='End', zorder=5)
    
    # Plot goal
    ax.scatter(goal_pos[0], goal_pos[1], 
              color='gold', s=200, marker='*', label='Goal', zorder=5)
    
    # Plot obstacle
    if obstacle_pos is not None and obstacle_radius is not None:
        circle = plt.Circle(obstacle_pos, obstacle_radius, 
                          color='gray', alpha=0.5, label='Obstacle')
        ax.add_patch(circle)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

