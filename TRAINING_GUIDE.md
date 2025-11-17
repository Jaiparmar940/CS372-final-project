# Training Guide

## Current Issue
The current model (`results/final_model.pt`) was trained with the old harsh reward configuration and shows 0% success rate. The environment works correctly (scripted policy achieves 100% success), but the RL agent needs better training.

## Solution: Retrain with Improved Configuration

### Step 1: Remove old model
```bash
rm results/final_model.pt
rm results/training.csv  # Optional: remove old training logs
```

### Step 2: Train with easy configuration (recommended for initial learning)
```bash
python scripts/train.py --config configs/easy_config.yaml
```

This uses:
- **No obstacles** (obstacle_prob: 0.0)
- **Very close goals** (goal_range: [-1.5, 1.5])
- **Large success threshold** (1.0 units)
- **High success reward** (500.0)
- **Small distance penalty** (0.01)
- **Shorter episodes** (200 steps max)
- **Higher learning rate** (5e-4)
- **More exploration** (entropy_coef: 0.1)

### Step 3: Train with improved configuration (after easy config works)
Once the easy config shows learning, you can use the improved config:
```bash
python scripts/train.py --config configs/improved_config.yaml
```

### Step 4: Evaluate
```bash
python scripts/evaluate.py --model results/final_model.pt
```

## Key Improvements Made

1. **Action Scaling**: Actions are now scaled by 2.0x for more responsive movement
2. **Progress Reward**: Agent gets rewarded for moving toward goal (0.5x velocity toward goal)
3. **Better Reward Shaping**: 
   - Smaller distance penalties
   - Larger success rewards
   - Minimal time penalties
4. **Curriculum Learning**: Start with easier tasks (no obstacles, close goals)
5. **Better Hyperparameters**: Higher learning rates, more exploration

## Expected Results After Retraining

With the easy config, you should see:
- **Success rate > 0%** (ideally > 50% after some training)
- **Episode lengths < 200** (not always hitting max)
- **Rewards improving** (less negative, moving toward positive)
- **Learning curves showing improvement**

## Troubleshooting

If the agent still doesn't learn:
1. Check that you're using the updated `robot_env.py` with action scaling
2. Verify the config file is being loaded correctly
3. Try even easier settings (goal_range: [-1, 1], goal_threshold: 1.5)
4. Increase training steps
5. Check that observations are reasonable (not NaN or inf)

