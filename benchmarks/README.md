# LiteTorch Benchmarks

This directory contains benchmark scripts to compare LiteTorch performance against PyTorch and Stable Baselines 3.

## Running Benchmarks

### Prerequisites
```bash
# For PyTorch benchmarks
pip install torch

# For RL benchmarks
pip install stable-baselines3 gymnasium
```

### Running Individual Benchmarks
```bash
# Tensor operations
python benchmarks/bench_tensor_ops.py

# Autograd
python benchmarks/bench_autograd.py

# Neural network layers
python benchmarks/bench_nn_layers.py

# Optimizers
python benchmarks/bench_optimizers.py

# RL algorithms
python benchmarks/bench_rl_algorithms.py
```

## Benchmark Structure

- `bench_tensor_ops.py` - Benchmark basic tensor operations vs PyTorch
- `bench_autograd.py` - Benchmark automatic differentiation vs PyTorch
- `bench_nn_layers.py` - Benchmark neural network layers vs PyTorch
- `bench_optimizers.py` - Benchmark optimization algorithms vs PyTorch
- `bench_rl_algorithms.py` - Benchmark RL algorithms vs Stable Baselines 3

## What's Benchmarked

Each benchmark measures:
1. **Execution time**: How long operations take
2. **Performance comparison**: Speed relative to reference implementation
3. **Correctness**: Verify results match (when applicable)
4. **Convergence**: For RL algorithms, compare learning curves

## Notes

- Benchmarks are implemented as placeholder scripts with TODO comments
- Actual benchmark code will be added as components are implemented
- PyTorch and SB3 are optional dependencies for benchmarking only
- LiteTorch is designed for education, not to match production performance
