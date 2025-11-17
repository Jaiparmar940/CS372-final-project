# Reinforcement Learning Project: Goal Navigation with Generalization

## What it Does

This project implements a complete reinforcement learning framework for training a robotic agent to navigate to goal positions in a 2D continuous environment. The agent learns using Proximal Policy Optimization (PPO), an actor-critic algorithm, and is evaluated on its ability to generalize to novel task variations (zero-shot generalization). The project includes a custom 2D robotic environment, multiple baseline agents (random, scripted, and tabular Q-learning), and a full training/evaluation pipeline with comprehensive logging and visualization.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train an agent:**
   ```bash
   python scripts/train.py --config configs/default_config.yaml
   ```

3. **Evaluate a trained model:**
   ```bash
   python scripts/evaluate.py --model results/final_model.pt
   ```

4. **Plot training curves:**
   ```bash
   python scripts/plot_results.py --log_file results/training.csv
   ```

## Video Links

*Demo Video:* [Link to demo video - to be added]
*Technical Walkthrough:* [Link to technical walkthrough - to be added]

## Evaluation

### Training Performance

The agent is trained on a distribution of tasks where:
- Goal positions are sampled from a limited range (default: [-4, 4] in both x and y)
- Obstacles are randomly placed with 50% probability
- Arena size is fixed at 10x10 units

### Zero-Shot Generalization

The agent is evaluated on held-out task variations:
- **Far goals**: Goals outside the training range ([-8, 8])
- **No obstacle**: Environments without obstacles
- **Always obstacle**: Environments with guaranteed obstacles
- **Large arena**: Larger arena size (15x15 units)

### Metrics

- **Success rate**: Percentage of episodes where agent reaches goal (within 0.5 units)
- **Mean episode reward**: Average cumulative reward per episode
- **Mean episode length**: Average number of steps per episode
- **Final distance**: Distance to goal at episode termination

## Project Structure

```
rl_project/
├── src/
│   ├── environment/      # Custom environments
│   ├── agents/           # RL agents (baselines, Q-learning, PPO)
│   ├── networks/         # Neural network architectures
│   ├── training/         # Training loop and buffers
│   ├── evaluation/       # Evaluation and generalization testing
│   └── utils/            # Config, logging, plotting utilities
├── configs/              # Configuration files
├── scripts/              # Training, evaluation, plotting scripts
├── tests/                # Unit tests
└── results/              # Output directory (logs, models, plots)
```

## MDP Structure

### State Space (Continuous, 12 dimensions)
- Robot position (x, y)
- Robot velocity (vx, vy)
- Goal position (gx, gy)
- Relative goal position (dx, dy)
- Distance to goal
- Obstacle position and radius (if present)

### Action Space (Continuous, 2 dimensions)
- Acceleration commands (ax, ay) ∈ [-1, 1]²

### Dynamics
- Point mass model: `v_{t+1} = v_t * (1 - damping) + a_t * dt`
- Position update: `p_{t+1} = p_t + v_t * dt`
- Velocity clipping to prevent excessive speeds

### Reward Function
- Distance-based reward: `-distance_scale * ||robot - goal||`
- Success bonus: +100 when within goal threshold (0.5 units)
- Collision penalty: -50 if robot collides with obstacle
- Time penalty: -0.1 per step

### Termination Conditions
- Success: Robot reaches goal (distance < 0.5)
- Timeout: Maximum steps (500) reached

## Training Procedure

1. **Task Distribution**: Each episode samples a new goal position and obstacle configuration
2. **Rollout Collection**: Agent collects trajectories of fixed length (2048 steps)
3. **Advantage Estimation**: Generalized Advantage Estimation (GAE) computes advantages
4. **Policy Update**: PPO clipped objective updates actor and critic networks
5. **Evaluation**: Periodic evaluation on held-out task variations

## Individual Contributions

[For group projects: describe individual contributions here]

