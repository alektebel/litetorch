"""
Comprehensive benchmarks for Tensor reshape and sum operations.

This benchmark suite compares:
- LiteTorch tensor operations against PyTorch and NumPy
- Time performance for various operation sizes
- Memory usage for different tensor sizes
- Edge cases and special scenarios

Usage:
    python benchmarks/bench_reshape_sum.py

Optional environment variables:
    ITERATIONS=1000  # Number of iterations for each benchmark
    SKIP_PYTORCH=1   # Skip PyTorch benchmarks
    SKIP_NUMPY=1     # Skip NumPy benchmarks
"""
import time
import sys
import os
import tracemalloc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litetorch.tensor import Tensor

# Check for optional dependencies
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Configuration
SKIP_PYTORCH = os.environ.get('SKIP_PYTORCH', '0') == '1'
SKIP_NUMPY = os.environ.get('SKIP_NUMPY', '0') == '1'
DEFAULT_ITERATIONS = int(os.environ.get('ITERATIONS', '1000'))


def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def format_memory(bytes_val):
    """Format memory in appropriate units."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


def benchmark_time(name, func, iterations=DEFAULT_ITERATIONS, warmup=10):
    """Benchmark execution time of a function."""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Actual benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()
    
    total_time = end - start
    avg_time = total_time / iterations
    
    return {
        'name': name,
        'total_time': total_time,
        'avg_time': avg_time,
        'iterations': iterations,
        'ops_per_sec': iterations / total_time if total_time > 0 else float('inf')
    }


def benchmark_memory(name, func):
    """Benchmark memory usage of a function."""
    tracemalloc.start()
    result = func()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'name': name,
        'current': current,
        'peak': peak,
        'result': result
    }


def print_comparison_header(operation):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {operation}")
    print("=" * 80)


def print_time_results(results_dict):
    """Print time benchmark results in a table."""
    if not results_dict:
        return
    
    # Print header
    print(f"\n{'Library':<15} {'Total Time':<15} {'Avg Time':<15} {'Ops/sec':<15} {'Speedup':<10}")
    print("-" * 80)
    
    # Get baseline (LiteTorch)
    baseline = None
    if 'LiteTorch' in results_dict:
        baseline = results_dict['LiteTorch']['avg_time']
    
    # Print results
    for lib_name in ['LiteTorch', 'NumPy', 'PyTorch']:
        if lib_name not in results_dict:
            continue
        
        result = results_dict[lib_name]
        total_time = format_time(result['total_time'])
        avg_time = format_time(result['avg_time'])
        ops_per_sec = f"{result['ops_per_sec']:.0f}"
        
        if baseline and lib_name != 'LiteTorch':
            speedup = baseline / result['avg_time']
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "1.00x (baseline)"
        
        print(f"{lib_name:<15} {total_time:<15} {avg_time:<15} {ops_per_sec:<15} {speedup_str:<10}")


def print_memory_results(results_dict):
    """Print memory benchmark results in a table."""
    if not results_dict:
        return
    
    print(f"\n{'Library':<15} {'Current':<15} {'Peak':<15} {'Efficiency':<10}")
    print("-" * 60)
    
    # Get baseline (LiteTorch)
    baseline = None
    if 'LiteTorch' in results_dict:
        baseline = results_dict['LiteTorch']['peak']
    
    # Print results
    for lib_name in ['LiteTorch', 'NumPy', 'PyTorch']:
        if lib_name not in results_dict:
            continue
        
        result = results_dict[lib_name]
        current = format_memory(result['current'])
        peak = format_memory(result['peak'])
        
        if baseline and lib_name != 'LiteTorch':
            efficiency = baseline / result['peak']
            efficiency_str = f"{efficiency:.2f}x"
        else:
            efficiency_str = "1.00x (baseline)"
        
        print(f"{lib_name:<15} {current:<15} {peak:<15} {efficiency_str:<10}")


# ============================================================================
# RESHAPE BENCHMARKS
# ============================================================================

def benchmark_reshape_small():
    """Benchmark reshape on small tensors (10x10 -> 100)."""
    print_comparison_header("Reshape: Small Tensors (10x10 -> 100)")
    
    results = {}
    
    # LiteTorch
    t = Tensor(shape=(10, 10))
    t.data = list(range(100))
    results['LiteTorch'] = benchmark_time(
        "LiteTorch", 
        lambda: t.reshape((100,)),
        iterations=10000
    )
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.arange(100).reshape(10, 10)
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.reshape(100),
            iterations=10000
        )
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.arange(100).reshape(10, 10)
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.reshape(100),
            iterations=10000
        )
    
    print_time_results(results)


def benchmark_reshape_medium():
    """Benchmark reshape on medium tensors (100x100 -> 10000)."""
    print_comparison_header("Reshape: Medium Tensors (100x100 -> 10000)")
    
    results = {}
    
    # LiteTorch
    t = Tensor(shape=(100, 100))
    t.data = list(range(10000))
    results['LiteTorch'] = benchmark_time(
        "LiteTorch",
        lambda: t.reshape((10000,)),
        iterations=1000
    )
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.arange(10000).reshape(100, 100)
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.reshape(10000),
            iterations=1000
        )
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.arange(10000).reshape(100, 100)
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.reshape(10000),
            iterations=1000
        )
    
    print_time_results(results)


def benchmark_reshape_large():
    """Benchmark reshape on large tensors (1000x1000 -> 1000000)."""
    print_comparison_header("Reshape: Large Tensors (1000x1000 -> 1000000)")
    
    results = {}
    
    # LiteTorch
    t = Tensor(shape=(1000, 1000))
    t.data = list(range(1000000))
    results['LiteTorch'] = benchmark_time(
        "LiteTorch",
        lambda: t.reshape((1000000,)),
        iterations=100
    )
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.arange(1000000).reshape(1000, 1000)
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.reshape(1000000),
            iterations=100
        )
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.arange(1000000).reshape(1000, 1000)
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.reshape(1000000),
            iterations=100
        )
    
    print_time_results(results)


def benchmark_reshape_3d():
    """Benchmark reshape on 3D tensors (10x10x10 -> 1000)."""
    print_comparison_header("Reshape: 3D Tensors (10x10x10 -> 1000)")
    
    results = {}
    
    # LiteTorch
    t = Tensor(shape=(10, 10, 10))
    t.data = list(range(1000))
    results['LiteTorch'] = benchmark_time(
        "LiteTorch",
        lambda: t.reshape((1000,)),
        iterations=5000
    )
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.arange(1000).reshape(10, 10, 10)
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.reshape(1000),
            iterations=5000
        )
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.arange(1000).reshape(10, 10, 10)
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.reshape(1000),
            iterations=5000
        )
    
    print_time_results(results)


def benchmark_reshape_memory():
    """Benchmark memory usage for reshape operations."""
    print_comparison_header("Reshape: Memory Usage (1000x1000)")
    
    results = {}
    
    # LiteTorch
    def litetorch_reshape_mem():
        t = Tensor(shape=(1000, 1000))
        t.data = list(range(1000000))
        return t.reshape((1000000,))
    results['LiteTorch'] = benchmark_memory("LiteTorch", litetorch_reshape_mem)
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        def numpy_reshape_mem():
            arr = np.arange(1000000).reshape(1000, 1000)
            return arr.reshape(1000000)
        results['NumPy'] = benchmark_memory("NumPy", numpy_reshape_mem)
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        def pytorch_reshape_mem():
            tensor = torch.arange(1000000).reshape(1000, 1000)
            return tensor.reshape(1000000)
        results['PyTorch'] = benchmark_memory("PyTorch", pytorch_reshape_mem)
    
    print_memory_results(results)


# ============================================================================
# SUM BENCHMARKS
# ============================================================================

def benchmark_sum_small():
    """Benchmark sum on small tensors (100 elements)."""
    print_comparison_header("Sum: Small Tensors (100 elements)")
    
    results = {}
    
    # LiteTorch
    t = Tensor(shape=(10, 10))
    t.data = list(range(100))
    results['LiteTorch'] = benchmark_time(
        "LiteTorch",
        lambda: t.sum(),
        iterations=10000
    )
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.arange(100).reshape(10, 10)
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.sum(),
            iterations=10000
        )
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.arange(100).reshape(10, 10)
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.sum(),
            iterations=10000
        )
    
    print_time_results(results)


def benchmark_sum_medium():
    """Benchmark sum on medium tensors (10000 elements)."""
    print_comparison_header("Sum: Medium Tensors (10000 elements)")
    
    results = {}
    
    # LiteTorch
    t = Tensor(shape=(100, 100))
    t.data = list(range(10000))
    results['LiteTorch'] = benchmark_time(
        "LiteTorch",
        lambda: t.sum(),
        iterations=1000
    )
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.arange(10000).reshape(100, 100)
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.sum(),
            iterations=1000
        )
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.arange(10000).reshape(100, 100)
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.sum(),
            iterations=1000
        )
    
    print_time_results(results)


def benchmark_sum_large():
    """Benchmark sum on large tensors (1000000 elements)."""
    print_comparison_header("Sum: Large Tensors (1000000 elements)")
    
    results = {}
    
    # LiteTorch
    t = Tensor(shape=(1000, 1000))
    t.data = list(range(1000000))
    results['LiteTorch'] = benchmark_time(
        "LiteTorch",
        lambda: t.sum(),
        iterations=100
    )
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.arange(1000000).reshape(1000, 1000)
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.sum(),
            iterations=100
        )
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.arange(1000000).reshape(1000, 1000)
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.sum(),
            iterations=100
        )
    
    print_time_results(results)


def benchmark_sum_1d():
    """Benchmark sum on 1D tensors."""
    print_comparison_header("Sum: 1D Tensors (10000 elements)")
    
    results = {}
    
    # LiteTorch
    t = Tensor(shape=(10000,))
    t.data = list(range(10000))
    results['LiteTorch'] = benchmark_time(
        "LiteTorch",
        lambda: t.sum(),
        iterations=1000
    )
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.arange(10000)
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.sum(),
            iterations=1000
        )
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.arange(10000)
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.sum(),
            iterations=1000
        )
    
    print_time_results(results)


def benchmark_sum_3d():
    """Benchmark sum on 3D tensors."""
    print_comparison_header("Sum: 3D Tensors (10x10x10)")
    
    results = {}
    
    # LiteTorch
    t = Tensor(shape=(10, 10, 10))
    t.data = list(range(1000))
    results['LiteTorch'] = benchmark_time(
        "LiteTorch",
        lambda: t.sum(),
        iterations=5000
    )
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.arange(1000).reshape(10, 10, 10)
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.sum(),
            iterations=5000
        )
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.arange(1000).reshape(10, 10, 10)
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.sum(),
            iterations=5000
        )
    
    print_time_results(results)


def benchmark_sum_memory():
    """Benchmark memory usage for sum operations."""
    print_comparison_header("Sum: Memory Usage (1000x1000)")
    
    results = {}
    
    # LiteTorch
    def litetorch_sum_mem():
        t = Tensor(shape=(1000, 1000))
        t.data = list(range(1000000))
        return t.sum()
    results['LiteTorch'] = benchmark_memory("LiteTorch", litetorch_sum_mem)
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        def numpy_sum_mem():
            arr = np.arange(1000000).reshape(1000, 1000)
            return arr.sum()
        results['NumPy'] = benchmark_memory("NumPy", numpy_sum_mem)
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        def pytorch_sum_mem():
            tensor = torch.arange(1000000).reshape(1000, 1000)
            return tensor.sum()
        results['PyTorch'] = benchmark_memory("PyTorch", pytorch_sum_mem)
    
    print_memory_results(results)


# ============================================================================
# EDGE CASES BENCHMARKS
# ============================================================================

def benchmark_edge_cases():
    """Benchmark edge cases."""
    print_comparison_header("Edge Cases: Single Element")
    
    results = {}
    
    # LiteTorch - single element
    t = Tensor(shape=(1,))
    t.data = [42]
    results['LiteTorch'] = benchmark_time(
        "LiteTorch",
        lambda: t.sum(),
        iterations=10000
    )
    
    # NumPy - single element
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.array([42])
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.sum(),
            iterations=10000
        )
    
    # PyTorch - single element
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.tensor([42])
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.sum(),
            iterations=10000
        )
    
    print_time_results(results)


def benchmark_reshape_edge_cases():
    """Benchmark reshape edge cases."""
    print_comparison_header("Edge Cases: Reshape with Singleton Dimensions")
    
    results = {}
    
    # LiteTorch - reshape with singleton dimensions
    t = Tensor(shape=(100,))
    t.data = list(range(100))
    results['LiteTorch'] = benchmark_time(
        "LiteTorch",
        lambda: t.reshape((1, 100, 1)),
        iterations=5000
    )
    
    # NumPy
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        arr = np.arange(100)
        results['NumPy'] = benchmark_time(
            "NumPy",
            lambda: arr.reshape(1, 100, 1),
            iterations=5000
        )
    
    # PyTorch
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        tensor = torch.arange(100)
        results['PyTorch'] = benchmark_time(
            "PyTorch",
            lambda: tensor.reshape(1, 100, 1),
            iterations=5000
        )
    
    print_time_results(results)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def print_system_info():
    """Print system information."""
    print("\n" + "=" * 80)
    print("  BENCHMARK: Tensor Reshape and Sum Operations")
    print("=" * 80)
    print(f"\nPython version: {sys.version}")
    print(f"NumPy available: {NUMPY_AVAILABLE and not SKIP_NUMPY}")
    if NUMPY_AVAILABLE and not SKIP_NUMPY:
        print(f"  NumPy version: {np.__version__}")
    print(f"PyTorch available: {PYTORCH_AVAILABLE and not SKIP_PYTORCH}")
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        print(f"  PyTorch version: {torch.__version__}")
    print(f"\nDefault iterations: {DEFAULT_ITERATIONS}")
    print()


def print_summary():
    """Print benchmark summary."""
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print("\nLiteTorch is a pure Python implementation for educational purposes.")
    print("As expected, it is significantly slower than optimized libraries like")
    print("NumPy (C backend) and PyTorch (C++/CUDA backend).")
    print("\nHowever, LiteTorch provides:")
    print("  - Clear, readable implementation for learning")
    print("  - No external dependencies (except numpy for math utilities)")
    print("  - Easy to understand and modify")
    print("\nFor production use, always prefer NumPy or PyTorch.")
    print()


def main():
    """Run all benchmarks."""
    print_system_info()
    
    # Reshape benchmarks
    benchmark_reshape_small()
    benchmark_reshape_medium()
    benchmark_reshape_large()
    benchmark_reshape_3d()
    benchmark_reshape_memory()
    
    # Sum benchmarks
    benchmark_sum_small()
    benchmark_sum_medium()
    benchmark_sum_large()
    benchmark_sum_1d()
    benchmark_sum_3d()
    benchmark_sum_memory()
    
    # Edge cases
    benchmark_edge_cases()
    benchmark_reshape_edge_cases()
    
    # Summary
    print_summary()


if __name__ == "__main__":
    main()
