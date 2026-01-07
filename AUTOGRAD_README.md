# Autograd Tests and Benchmarks

This document provides detailed information about the comprehensive tests and benchmarks for automatic differentiation (autograd) functionality in LiteTorch.

## Overview

The autograd module is responsible for automatic differentiation - computing gradients of operations with respect to their inputs. This is the foundation of neural network training.

## Test Suite

### Test Coverage (27 tests)

The test suite in `tests/test_autograd.py` includes:

#### 1. Basic Backward Pass (6 tests)
- `test_simple_backward_scalar` - Simple scalar operation backward pass
- `test_backward_addition` - Gradient for addition operation
- `test_backward_subtraction` - Gradient for subtraction operation
- `test_backward_multiplication` - Gradient for element-wise multiplication
- `test_backward_division` - Gradient for division operation
- `test_backward_power` - Gradient for power operation

#### 2. Chain Rule (3 tests)
- `test_chain_rule_simple` - Simple chain of operations
- `test_chain_rule_complex` - Complex chain with multiple operations
- `test_chain_rule_branching` - Branching computational graph

#### 3. Matrix Operations (2 tests)
- `test_matmul_gradient` - Gradient for matrix multiplication
- `test_matmul_chain` - Chain of matrix multiplications

#### 4. Activation Functions (4 tests)
- `test_relu_gradient` - ReLU activation gradient
- `test_sigmoid_gradient` - Sigmoid activation gradient
- `test_tanh_gradient` - Tanh activation gradient
- `test_softmax_gradient` - Softmax activation gradient

#### 5. Gradient Accumulation (2 tests)
- `test_gradient_accumulation_simple` - Basic gradient accumulation
- `test_zero_grad` - Gradient zeroing functionality

#### 6. Edge Cases (5 tests)
- `test_no_grad_context` - Computation without gradient tracking
- `test_detach` - Detaching tensor from computational graph
- `test_backward_on_non_scalar` - Backward on non-scalar tensors
- `test_in_place_operations` - In-place operations handling
- `test_retain_graph` - Retaining graph for multiple backward passes

#### 7. Computational Graphs (3 tests)
- `test_diamond_graph` - Diamond-shaped computational graph
- `test_multiple_outputs` - Tensor used in multiple operations
- `test_nested_operations` - Deeply nested operations

#### 8. Neural Networks (2 tests)
- `test_simple_linear_layer` - Single linear layer backward pass
- `test_two_layer_network` - Two-layer network backward pass

### Running Tests

```bash
# Run all autograd tests
python -m pytest tests/test_autograd.py -v

# Run specific test class
python -m pytest tests/test_autograd.py::TestAutogradBasicBackward -v

# Run specific test
python -m pytest tests/test_autograd.py::TestAutogradBasicBackward::test_backward_addition -v

# Run with detailed output
python -m pytest tests/test_autograd.py -v -s
```

### Test Status

⚠️ **Note:** All tests are currently marked as skipped with `@unittest.skip("Autograd engine not yet implemented")` because the autograd functionality needs to be implemented in the Tensor class.

These tests define the expected API and behavior for the autograd engine:

```python
# Expected API
x = Tensor(shape=(2,), requires_grad=True)
x.data = [1.0, 2.0]

y = x * 2.0  # Forward pass builds computational graph
loss = y.sum()

loss.backward()  # Backward pass computes gradients

print(x.grad.data)  # Access gradients
x.zero_grad()  # Reset gradients
```

## Benchmark Suite

### Benchmark Coverage

The benchmark suite in `benchmarks/bench_autograd.py` includes:

#### 1. Simple Backward Pass
- Element-wise multiplication followed by sum
- Measures basic backward pass performance

#### 2. Chain Backward Pass
- Chain of operations: `((x * 2 + 3) * 4 + 5) * 6`
- Tests gradient propagation through multiple operations

#### 3. Matrix Multiplication Backward
- Matrix multiplication with gradient computation
- Tests performance of matmul backward pass

#### 4. Activation Functions Backward
- ReLU backward pass
- Sigmoid backward pass
- Compares activation function gradient performance

#### 5. Neural Network Backward
- Two-layer neural network (input → hidden → output)
- Tests realistic neural network training scenario

#### 6. Memory Usage
- Measures peak memory usage during backward pass
- Helps identify memory inefficiencies

#### 7. Gradient Accumulation
- Multiple backward passes without zeroing gradients
- Tests gradient accumulation performance

### Running Benchmarks

```bash
# Run all autograd benchmarks
python benchmarks/bench_autograd.py

# Run in quick mode (smaller sizes, fewer iterations)
QUICK=1 python benchmarks/bench_autograd.py

# Run with custom iteration count
ITERATIONS=500 python benchmarks/bench_autograd.py

# Run without PyTorch comparison
SKIP_PYTORCH=1 python benchmarks/bench_autograd.py
```

### Environment Variables

- `QUICK=1` - Run with smaller tensor sizes and fewer iterations
- `ITERATIONS=N` - Set number of iterations per benchmark (default: 1000, quick: 100)
- `SKIP_PYTORCH=1` - Skip PyTorch comparison benchmarks

### Expected Performance

Based on similar operations in other parts of LiteTorch:

- **LiteTorch**: Pure Python implementation, typically 10-1000x slower than PyTorch
- **PyTorch**: Highly optimized C++/CUDA backend with graph optimizations
- **Purpose**: LiteTorch prioritizes educational clarity over performance

Example output:
```
============================================================
Benchmark: Simple Backward Pass
============================================================
  LiteTorch: 1.53 µs
  PyTorch:   0.05 µs
  Ratio:     30.6x slower

============================================================
Benchmark: Neural Network Backward Pass
============================================================
  LiteTorch: 150.2 µs
  PyTorch:   2.1 µs
  Ratio:     71.5x slower
```

## Implementation Requirements

To enable these tests and benchmarks, the following needs to be implemented in the Tensor class:

### 1. Tensor Attributes
```python
class Tensor:
    def __init__(self, shape, requires_grad=False):
        self.requires_grad = requires_grad
        self.grad = None  # Stores gradients
        self.grad_fn = None  # Function to compute gradients
        # ... existing attributes ...
```

### 2. Backward Pass
```python
def backward(self, gradient=None, retain_graph=False):
    """
    Compute gradients using reverse-mode automatic differentiation.
    
    Args:
        gradient: Optional gradient tensor (required for non-scalar tensors)
        retain_graph: Whether to keep the computational graph after backward
    """
    # Build reverse graph and compute gradients
    pass
```

### 3. Gradient Management
```python
def zero_grad(self):
    """Reset gradients to zero."""
    if self.grad is not None:
        self.grad.data = [0.0] * len(self.grad.data)

def detach(self):
    """Detach tensor from computational graph."""
    result = self.clone()
    result.requires_grad = False
    result.grad_fn = None
    return result
```

### 4. Operation Tracking
Each operation (add, mul, matmul, etc.) needs to:
1. Create output tensor with `requires_grad=True` if any input requires gradients
2. Store a backward function (`grad_fn`) that computes gradients
3. Track dependencies for the computational graph

### 5. Context Managers
```python
class no_grad:
    """Context manager to disable gradient tracking."""
    def __enter__(self):
        # Disable gradient tracking
        pass
    
    def __exit__(self, *args):
        # Re-enable gradient tracking
        pass
```

## Educational Value

These tests and benchmarks provide:

1. **Clear API Definition**: Shows how autograd should work
2. **Comprehensive Coverage**: Tests all major operations and edge cases
3. **Performance Baseline**: Establishes performance expectations
4. **Learning Tool**: Demonstrates automatic differentiation concepts

## Comparison with PyTorch

The API is designed to be similar to PyTorch for familiarity:

| Feature | LiteTorch | PyTorch |
|---------|-----------|---------|
| Enable gradients | `requires_grad=True` | `requires_grad=True` |
| Compute gradients | `tensor.backward()` | `tensor.backward()` |
| Access gradients | `tensor.grad` | `tensor.grad` |
| Zero gradients | `tensor.zero_grad()` | `tensor.zero_grad()` |
| Disable tracking | `with no_grad():` | `with torch.no_grad():` |
| Detach tensor | `tensor.detach()` | `tensor.detach()` |

## Future Enhancements

Potential improvements for the test suite:

1. **Higher-order gradients** - Test gradient of gradients
2. **Sparse gradients** - Test gradient computation for sparse tensors
3. **Gradient checking** - Numerical gradient verification
4. **Custom autograd functions** - Extend autograd with custom operations
5. **Graph visualization** - Visualize computational graphs

## Contributing

When adding new autograd features:

1. Add corresponding tests to `tests/test_autograd.py`
2. Add performance benchmarks to `benchmarks/bench_autograd.py`
3. Update this README with new test/benchmark descriptions
4. Ensure tests are well-documented with clear docstrings
5. Mark tests with `@unittest.skip()` if implementation is pending

## References

- [PyTorch Autograd](https://pytorch.org/docs/stable/autograd.html)
- [Automatic Differentiation in Machine Learning](https://arxiv.org/abs/1502.05767)
- [Reverse-mode Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation)

## Summary

This comprehensive test and benchmark suite provides:
- ✅ 27 tests covering all aspects of automatic differentiation
- ✅ Performance benchmarks comparing with PyTorch
- ✅ Clear API definition for autograd implementation
- ✅ Educational examples of gradient computation
- ✅ Support for quick/full benchmark modes
- ✅ Memory usage tracking

The tests currently skip since the autograd engine is not yet implemented, but they provide a clear roadmap for implementation and ensure correctness once the feature is added.
