# Reshape and Sum Tests & Benchmarks

This directory contains comprehensive tests and benchmarks for the Tensor `reshape` and `sum` operations in LiteTorch.

## Overview

The test suite includes:
- **40 comprehensive tests** covering reshape and sum operations
- Tests for basic operations, edge cases, and invalid inputs
- Benchmarks comparing LiteTorch against NumPy and PyTorch
- Memory usage benchmarks

## Quick Start

### Running Tests

```bash
# Using Python runner (cross-platform)
python run_tests_benchmarks.py test

# Or using bash script (Linux/Mac)
./run_tests_benchmarks.sh test
```

### Running Benchmarks

```bash
# Full benchmarks (takes several minutes)
python run_tests_benchmarks.py benchmark

# Quick benchmarks (fewer iterations)
python run_tests_benchmarks.py benchmark-quick
```

### Running Everything

```bash
python run_tests_benchmarks.py all
```

## Available Commands

| Command | Description |
|---------|-------------|
| `test` | Run reshape & sum tests only |
| `test-all` | Run all tensor tests |
| `benchmark` | Run full benchmarks (~5 minutes) |
| `benchmark-quick` | Run quick benchmarks (~30 seconds) |
| `benchmark-no-pytorch` | Run benchmarks without PyTorch |
| `benchmark-no-numpy` | Run benchmarks without NumPy |
| `benchmark-litetorch-only` | Run benchmarks for LiteTorch only |
| `all` | Run all tests and benchmarks |
| `install` | Install dependencies |
| `help` | Show help message |

## Test Coverage

### Reshape Tests (27 tests)

#### Basic Operations
- 1D to 2D, 2D to 1D, 2D to different 2D
- 1D to 3D, 3D to 1D, 3D to 2D
- Data order preservation
- Reshape to same shape

#### Edge Cases
- Single element tensors
- Tensors with dimensions of size 1
- Large tensors (1000+ elements)
- 4D and 5D tensors
- Multi-dimensional transformations

#### Invalid Operations
- Incompatible sizes
- Too large/small dimensions
- Zero dimensions

#### Data Integrity
- Original tensor unchanged
- Independent copies created
- Multiple consecutive reshapes

### Sum Tests (13 tests)

#### Basic Operations
- Sum without axis (1D, 2D, 3D)
- Sum with all zeros
- Sum with negative values
- Sum with mixed positive/negative

#### Edge Cases
- Single element tensors
- Large tensors
- Float values
- Original tensor preservation

#### Axis Operations
- Sum along specific axes
- Sum along multiple axes (tuple)

## Benchmark Results

The benchmarks compare:
- **Time performance**: Iterations per second for various sizes
- **Memory usage**: Peak and current memory consumption

### Expected Results

LiteTorch is a pure Python implementation and is significantly slower than optimized libraries:

- **NumPy**: 10-1000x faster (C backend)
- **PyTorch**: 10-1000x faster (C++/CUDA backend)

However, LiteTorch provides:
- ✓ Clear, readable implementation for learning
- ✓ No complex dependencies
- ✓ Easy to understand and modify

### Sample Benchmark Output

```
================================================================================
  Reshape: Small Tensors (10x10 -> 100)
================================================================================

Library         Total Time      Avg Time        Ops/sec         Speedup   
--------------------------------------------------------------------------------
LiteTorch       39.72 ms        3.97 µs         251734          1.00x (baseline)
NumPy           2.00 ms         199.87 ns       5003295         19.88x    
PyTorch         1.50 ms         150.00 ns       6666667         26.47x    
```

## Environment Variables

You can customize the benchmarks using environment variables:

```bash
# Set number of iterations
ITERATIONS=500 python run_tests_benchmarks.py benchmark

# Skip specific libraries
SKIP_PYTORCH=1 python run_tests_benchmarks.py benchmark
SKIP_NUMPY=1 python run_tests_benchmarks.py benchmark
```

## Requirements

### Required
- Python 3.6+
- pytest (for tests)
- numpy (already required by litetorch)

### Optional (for benchmarks)
- PyTorch (for PyTorch comparison)
- numpy (for NumPy comparison)

Install optional dependencies:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Test Details

### Reshape Operation

The `reshape` operation changes the shape of a tensor while preserving its data. It must satisfy:
- Total number of elements remains the same
- Data order is preserved (row-major / C-style)
- Original tensor is not modified (creates a copy)

Example:
```python
from litetorch.tensor import Tensor

t = Tensor(shape=(2, 3))
t.data = [1, 2, 3, 4, 5, 6]

reshaped = t.reshape((3, 2))
# reshaped.shape = (3, 2)
# reshaped.data = [1, 2, 3, 4, 5, 6]
```

### Sum Operation

The `sum` operation computes the sum of all elements or along specific axes.

Example:
```python
from litetorch.tensor import Tensor

t = Tensor(shape=(2, 3))
t.data = [1, 2, 3, 4, 5, 6]

# Sum all elements
total = t.sum()  # Returns 21

# Sum along axis (not fully implemented)
# result = t.sum(axis=0)
```

## Known Issues

1. **Sum with tuple axis**: Currently has a bug (tuple modification issue)
   - Status: Documented in tests
   - Workaround: Use sum without axis for now

2. **Sum with single axis**: Returns None (not implemented)
   - Status: Placeholder in original code
   - Tests document expected behavior

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Use descriptive test names
3. Add docstrings explaining what is tested
4. Group related tests in classes
5. Test both valid and invalid inputs

When adding benchmarks:
1. Include all three libraries (LiteTorch, NumPy, PyTorch)
2. Use appropriate iteration counts for the operation size
3. Include both time and memory benchmarks
4. Document expected performance characteristics

## Files

- `tests/test_reshape_sum.py` - Comprehensive test suite
- `benchmarks/bench_reshape_sum.py` - Benchmark suite
- `run_tests_benchmarks.py` - Python runner script (cross-platform)
- `run_tests_benchmarks.sh` - Bash runner script (Linux/Mac)
- `RESHAPE_SUM_README.md` - This file

## License

This is part of the LiteTorch project - an educational reimplementation of PyTorch.
