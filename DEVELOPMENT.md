# LiteTorch Development Guide

This document outlines the project structure, planned implementations, and development guidelines.

## Project Structure

```
litetorch/
├── litetorch/          # Main package
│   ├── autograd/       # Automatic differentiation
│   ├── nn/             # Neural network modules
│   ├── optim/          # Optimizers
│   └── rl/             # Reinforcement learning algorithms
├── tests/              # Test files
└── benchmarks/         # Benchmark scripts
```

## TODO List - Implementations

### 1. Core Tensor Operations (litetorch/autograd/)
- [ ] **Tensor class** - Basic tensor implementation with NumPy backend
- [ ] **Autograd engine** - Automatic differentiation with computational graph
- [ ] **Backward propagation** - Gradient computation through the graph
- [ ] **Common operations** - add, sub, mul, div, matmul, pow
- [ ] **Activation functions gradients** - ReLU, Sigmoid, Tanh
- [ ] **Reduction operations** - sum, mean, max, min

### 2. Neural Network Layers (litetorch/nn/)
- [ ] **Linear** - Fully connected layer
- [ ] **Conv2d** - 2D Convolutional layer
- [ ] **MaxPool2d** - 2D Max pooling
- [ ] **Dropout** - Dropout regularization
- [ ] **BatchNorm2d** - Batch normalization
- [ ] **ReLU** - Rectified Linear Unit activation
- [ ] **Sigmoid** - Sigmoid activation
- [ ] **Tanh** - Hyperbolic tangent activation
- [ ] **Softmax** - Softmax activation
- [ ] **Module** - Base class for all neural network modules
- [ ] **Sequential** - Sequential container for layers
- [ ] **MSELoss** - Mean Squared Error loss
- [ ] **CrossEntropyLoss** - Cross-entropy loss

### 3. Optimizers (litetorch/optim/)
- [ ] **SGD** - Stochastic Gradient Descent
- [ ] **SGD with Momentum** - SGD with momentum
- [ ] **Adam** - Adaptive Moment Estimation
- [ ] **RMSprop** - Root Mean Square Propagation
- [ ] **AdaGrad** - Adaptive Gradient Algorithm

### 4. Reinforcement Learning Algorithms (litetorch/rl/)
- [ ] **DQN** - Deep Q-Network
- [ ] **Double DQN** - Double Deep Q-Network
- [ ] **A2C** - Advantage Actor-Critic
- [ ] **PPO** - Proximal Policy Optimization
- [ ] **SAC** - Soft Actor-Critic
- [ ] **TD3** - Twin Delayed Deep Deterministic Policy Gradient
- [ ] **REINFORCE** - Policy Gradient algorithm
- [ ] **Replay Buffer** - Experience replay buffer

## Testing

Each implementation includes test files in the `tests/` directory:
- `tests/test_tensor.py` - Tests for Tensor operations
- `tests/test_autograd.py` - Tests for automatic differentiation
- `tests/test_nn_layers.py` - Tests for neural network layers
- `tests/test_optimizers.py` - Tests for optimizers
- `tests/test_rl_algorithms.py` - Tests for RL algorithms

### Running Tests

Using unittest (built-in):
```bash
python -m unittest discover tests/ -v
```

Using pytest (if installed):
```bash
pip install pytest
pytest tests/ -v
```

## Benchmarks

Each implementation includes benchmark scripts in the `benchmarks/` directory comparing performance against PyTorch or Stable Baselines 3:
- `benchmarks/bench_tensor_ops.py` - Benchmark tensor operations vs PyTorch
- `benchmarks/bench_autograd.py` - Benchmark autograd vs PyTorch
- `benchmarks/bench_nn_layers.py` - Benchmark layers vs PyTorch
- `benchmarks/bench_optimizers.py` - Benchmark optimizers vs PyTorch
- `benchmarks/bench_rl_algorithms.py` - Benchmark RL algorithms vs Stable Baselines 3

### Running Benchmarks

```bash
# For PyTorch benchmarks
pip install torch

# For RL benchmarks
pip install stable-baselines3 gymnasium

# Run individual benchmarks
python benchmarks/bench_tensor_ops.py
python benchmarks/bench_autograd.py
python benchmarks/bench_nn_layers.py
python benchmarks/bench_optimizers.py
python benchmarks/bench_rl_algorithms.py
```

## Installation

```bash
pip install -e .
```

For benchmark dependencies:
```bash
pip install -e .[benchmark]
```

## Usage

```python
import litetorch as lt

# Example usage will be added as implementations are completed
```
