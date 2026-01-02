# LiteTorch Tests

This directory contains test files for all LiteTorch components.

## Running Tests

### Using unittest (built-in)
```bash
python -m unittest discover tests/ -v
```

### Using pytest (if installed)
```bash
pip install pytest
pytest tests/ -v
```

## Test Structure

- `test_tensor.py` - Tests for basic Tensor operations and functionality
- `test_autograd.py` - Tests for automatic differentiation engine
- `test_nn_layers.py` - Tests for neural network layers and modules
- `test_optimizers.py` - Tests for optimization algorithms
- `test_rl_algorithms.py` - Tests for reinforcement learning algorithms

## Test Coverage

Each test file contains placeholder test cases that should be implemented as the corresponding components are developed. The tests are structured to verify:

1. **Correctness**: Results match expected values
2. **Shape compatibility**: Tensors have correct shapes
3. **Gradient computation**: Backpropagation works correctly
4. **Edge cases**: Proper handling of boundary conditions

## Contributing Tests

When implementing a new component:
1. Update the corresponding test file with actual test implementations
2. Ensure tests cover both forward and backward passes (where applicable)
3. Test edge cases and error handling
4. Compare results with PyTorch when possible for validation
