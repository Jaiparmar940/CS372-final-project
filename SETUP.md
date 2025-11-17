# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. **Clone or download the project:**
   ```bash
   cd rl_project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

The project requires the following packages:
- `numpy>=1.21.0`: Numerical computations and environment simulation
- `torch>=1.12.0`: PyTorch for neural networks and GPU support
- `gymnasium>=0.28.0`: Environment API (Gymnasium standard)
- `matplotlib>=3.5.0`: Plotting and visualization
- `pyyaml>=6.0`: Configuration file parsing
- `pandas>=1.3.0`: Data handling for plotting

## GPU Support (Optional)

If you have a CUDA-compatible GPU and want to use it for training:

1. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Update the config file to use GPU:
   ```yaml
   agent:
     device: "cuda"
   ```

## Verification

Run the basic environment test to verify installation:
```bash
python tests/test_environment.py
```

You should see:
```
✓ Robot environment tests passed
✓ Discrete environment tests passed

All tests passed!
```

## Project Structure

After installation, the project structure should be:
```
rl_project/
├── src/              # Source code
├── configs/          # Configuration files
├── scripts/          # Executable scripts
├── tests/            # Unit tests
├── results/          # Output directory (created automatically)
└── requirements.txt  # Dependencies
```

## Running the Project

See `README.md` for instructions on training, evaluation, and visualization.

