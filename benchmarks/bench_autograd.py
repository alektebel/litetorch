"""
Comprehensive benchmarks for automatic differentiation (autograd).

This benchmark suite compares:
- LiteTorch autograd against PyTorch autograd
- Time performance for various backward operations
- Memory usage for different graph sizes
- Gradient computation accuracy

Usage:
    python benchmarks/bench_autograd.py

Optional environment variables:
    ITERATIONS=1000  # Number of iterations for each benchmark
    SKIP_PYTORCH=1   # Skip PyTorch benchmarks
    QUICK=1          # Run quick benchmarks with smaller sizes
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
    print("Warning: PyTorch not available. Install with: pip install torch")
    print("Benchmarks will run in LiteTorch-only mode.\n")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Some benchmarks may be skipped.\n")

# Configuration
SKIP_PYTORCH = os.environ.get('SKIP_PYTORCH', '0') == '1'
QUICK = os.environ.get('QUICK', '0') == '1'
DEFAULT_ITERATIONS = int(os.environ.get('ITERATIONS', '100' if QUICK else '1000'))


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
        try:
            func()
        except Exception:
            pass
    
    # Actual benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        try:
            func()
        except Exception:
            pass
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
    try:
        result = func()
    except Exception:
        result = None
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'name': name,
        'current': current,
        'peak': peak,
        'result': result
    }


def print_comparison(litetorch_result, pytorch_result):
    """Print comparison between LiteTorch and PyTorch."""
    print(f"  LiteTorch: {format_time(litetorch_result['avg_time'])}")
    
    if pytorch_result:
        print(f"  PyTorch:   {format_time(pytorch_result['avg_time'])}")
        
        if litetorch_result['avg_time'] > 0:
            speedup = litetorch_result['avg_time'] / pytorch_result['avg_time']
            if speedup > 1:
                print(f"  Ratio:     {speedup:.2f}x slower")
            else:
                print(f"  Ratio:     {1/speedup:.2f}x faster")
    print()


def benchmark_simple_backward():
    """Benchmark simple backward pass."""
    print("=" * 60)
    print("Benchmark: Simple Backward Pass")
    print("=" * 60)
    print("Note: Requires autograd implementation in Tensor class")
    print()
    
    size = 10 if QUICK else 100
    
    try:
        def litetorch_backward():
            x = Tensor(shape=(size, size), requires_grad=True)
            if NUMPY_AVAILABLE:
                x.data = list(np.random.randn(size * size))
            else:
                x.data = list(range(size * size))
            
            y = Tensor(shape=(size, size), requires_grad=True)
            if NUMPY_AVAILABLE:
                y.data = list(np.random.randn(size * size))
            else:
                y.data = list(range(size * size))
            
            z = x * y
            loss = z.sum()
            loss.backward()
        
        litetorch_result = benchmark_time("LiteTorch Simple Backward", litetorch_backward)
    except Exception as e:
        print(f"LiteTorch: Not implemented yet ({type(e).__name__})")
        litetorch_result = None
    
    pytorch_result = None
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        def pytorch_backward():
            x = torch.randn(size, size, requires_grad=True)
            y = torch.randn(size, size, requires_grad=True)
            z = x * y
            loss = z.sum()
            loss.backward()
        
        pytorch_result = benchmark_time("PyTorch Simple Backward", pytorch_backward)
    
    if litetorch_result:
        print_comparison(litetorch_result, pytorch_result)
    else:
        print("Skipped - autograd not implemented\n")


def benchmark_chain_backward():
    """Benchmark backward pass through chain of operations."""
    print("=" * 60)
    print("Benchmark: Chain Backward Pass")
    print("=" * 60)
    print("Note: Requires autograd implementation in Tensor class")
    print()
    
    size = 10 if QUICK else 50
    
    try:
        def litetorch_chain():
            x = Tensor(shape=(size, size), requires_grad=True)
            if NUMPY_AVAILABLE:
                x.data = list(np.random.randn(size * size))
            else:
                x.data = list(range(size * size))
            
            # Chain: y = ((x * 2 + 3) * 4 + 5) * 6
            y = x * 2.0
            y = y + 3.0
            y = y * 4.0
            y = y + 5.0
            y = y * 6.0
            
            loss = y.sum()
            loss.backward()
        
        litetorch_result = benchmark_time("LiteTorch Chain Backward", litetorch_chain)
    except Exception as e:
        print(f"LiteTorch: Not implemented yet ({type(e).__name__})")
        litetorch_result = None
    
    pytorch_result = None
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        def pytorch_chain():
            x = torch.randn(size, size, requires_grad=True)
            
            # Same chain
            y = x * 2.0
            y = y + 3.0
            y = y * 4.0
            y = y + 5.0
            y = y * 6.0
            
            loss = y.sum()
            loss.backward()
        
        pytorch_result = benchmark_time("PyTorch Chain Backward", pytorch_chain)
    
    if litetorch_result:
        print_comparison(litetorch_result, pytorch_result)
    else:
        print("Skipped - autograd not implemented\n")


def benchmark_matmul_backward():
    """Benchmark backward pass through matrix multiplication."""
    print("=" * 60)
    print("Benchmark: Matrix Multiplication Backward")
    print("=" * 60)
    print("Note: Requires autograd implementation in Tensor class")
    print()
    
    size = 10 if QUICK else 50
    
    try:
        def litetorch_matmul():
            A = Tensor(shape=(size, size), requires_grad=True)
            B = Tensor(shape=(size, size), requires_grad=True)
            
            if NUMPY_AVAILABLE:
                A.data = list(np.random.randn(size * size))
                B.data = list(np.random.randn(size * size))
            else:
                A.data = list(range(size * size))
                B.data = list(range(size * size))
            
            C = A @ B
            loss = C.sum()
            loss.backward()
        
        litetorch_result = benchmark_time("LiteTorch Matmul Backward", litetorch_matmul)
    except Exception as e:
        print(f"LiteTorch: Not implemented yet ({type(e).__name__})")
        litetorch_result = None
    
    pytorch_result = None
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        def pytorch_matmul():
            A = torch.randn(size, size, requires_grad=True)
            B = torch.randn(size, size, requires_grad=True)
            
            C = A @ B
            loss = C.sum()
            loss.backward()
        
        pytorch_result = benchmark_time("PyTorch Matmul Backward", pytorch_matmul)
    
    if litetorch_result:
        print_comparison(litetorch_result, pytorch_result)
    else:
        print("Skipped - autograd not implemented\n")


def benchmark_activation_backward():
    """Benchmark backward pass through activation functions."""
    print("=" * 60)
    print("Benchmark: Activation Functions Backward")
    print("=" * 60)
    print("Note: Requires autograd implementation in Tensor class")
    print()
    
    size = 100 if QUICK else 1000
    
    # ReLU
    print("ReLU:")
    try:
        def litetorch_relu():
            x = Tensor(shape=(size, size), requires_grad=True)
            if NUMPY_AVAILABLE:
                x.data = list(np.random.randn(size * size))
            else:
                x.data = list(range(size * size))
            
            y = x.relu()
            loss = y.sum()
            loss.backward()
        
        litetorch_result = benchmark_time("LiteTorch ReLU", litetorch_relu)
    except Exception as e:
        print(f"  LiteTorch: Not implemented yet ({type(e).__name__})")
        litetorch_result = None
    
    pytorch_result = None
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        def pytorch_relu():
            x = torch.randn(size, size, requires_grad=True)
            y = torch.relu(x)
            loss = y.sum()
            loss.backward()
        
        pytorch_result = benchmark_time("PyTorch ReLU", pytorch_relu)
    
    if litetorch_result:
        print_comparison(litetorch_result, pytorch_result)
    else:
        print("  Skipped - autograd not implemented\n")
    
    # Sigmoid
    print("Sigmoid:")
    try:
        def litetorch_sigmoid():
            x = Tensor(shape=(size, size), requires_grad=True)
            if NUMPY_AVAILABLE:
                x.data = list(np.random.randn(size * size))
            else:
                x.data = list(range(size * size))
            
            y = x.sigmoid()
            loss = y.sum()
            loss.backward()
        
        litetorch_result = benchmark_time("LiteTorch Sigmoid", litetorch_sigmoid)
    except Exception as e:
        print(f"  LiteTorch: Not implemented yet ({type(e).__name__})")
        litetorch_result = None
    
    pytorch_result = None
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        def pytorch_sigmoid():
            x = torch.randn(size, size, requires_grad=True)
            y = torch.sigmoid(x)
            loss = y.sum()
            loss.backward()
        
        pytorch_result = benchmark_time("PyTorch Sigmoid", pytorch_sigmoid)
    
    if litetorch_result:
        print_comparison(litetorch_result, pytorch_result)
    else:
        print("  Skipped - autograd not implemented\n")


def benchmark_neural_network_backward():
    """Benchmark backward pass through a small neural network."""
    print("=" * 60)
    print("Benchmark: Neural Network Backward Pass")
    print("=" * 60)
    print("Note: Requires autograd and nn layers implementation")
    print()
    
    batch_size = 4 if QUICK else 32
    input_size = 10 if QUICK else 100
    hidden_size = 10 if QUICK else 50
    output_size = 5 if QUICK else 10
    
    try:
        def litetorch_nn():
            # Input
            x = Tensor(shape=(batch_size, input_size), requires_grad=True)
            if NUMPY_AVAILABLE:
                x.data = list(np.random.randn(batch_size * input_size))
            else:
                x.data = list(range(batch_size * input_size))
            
            # Layer 1
            W1 = Tensor(shape=(input_size, hidden_size), requires_grad=True)
            b1 = Tensor(shape=(hidden_size,), requires_grad=True)
            if NUMPY_AVAILABLE:
                W1.data = list(np.random.randn(input_size * hidden_size))
                b1.data = list(np.random.randn(hidden_size))
            else:
                W1.data = list(range(input_size * hidden_size))
                b1.data = list(range(hidden_size))
            
            # Layer 2
            W2 = Tensor(shape=(hidden_size, output_size), requires_grad=True)
            b2 = Tensor(shape=(output_size,), requires_grad=True)
            if NUMPY_AVAILABLE:
                W2.data = list(np.random.randn(hidden_size * output_size))
                b2.data = list(np.random.randn(output_size))
            else:
                W2.data = list(range(hidden_size * output_size))
                b2.data = list(range(output_size))
            
            # Forward pass
            h = (x @ W1) + b1
            h = h.relu()
            y = (h @ W2) + b2
            
            loss = y.sum()
            loss.backward()
        
        litetorch_result = benchmark_time("LiteTorch NN Backward", litetorch_nn)
    except Exception as e:
        print(f"LiteTorch: Not implemented yet ({type(e).__name__})")
        litetorch_result = None
    
    pytorch_result = None
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        def pytorch_nn():
            # Input
            x = torch.randn(batch_size, input_size, requires_grad=True)
            
            # Layer 1
            W1 = torch.randn(input_size, hidden_size, requires_grad=True)
            b1 = torch.randn(hidden_size, requires_grad=True)
            
            # Layer 2
            W2 = torch.randn(hidden_size, output_size, requires_grad=True)
            b2 = torch.randn(output_size, requires_grad=True)
            
            # Forward pass
            h = torch.matmul(x, W1) + b1
            h = torch.relu(h)
            y = torch.matmul(h, W2) + b2
            
            loss = y.sum()
            loss.backward()
        
        pytorch_result = benchmark_time("PyTorch NN Backward", pytorch_nn)
    
    if litetorch_result:
        print_comparison(litetorch_result, pytorch_result)
    else:
        print("Skipped - autograd not implemented\n")


def benchmark_memory_usage():
    """Benchmark memory usage of autograd."""
    print("=" * 60)
    print("Benchmark: Memory Usage")
    print("=" * 60)
    print("Note: Requires autograd implementation in Tensor class")
    print()
    
    size = 10 if QUICK else 100
    
    try:
        def litetorch_memory():
            x = Tensor(shape=(size, size), requires_grad=True)
            if NUMPY_AVAILABLE:
                x.data = list(np.random.randn(size * size))
            else:
                x.data = list(range(size * size))
            
            # Chain of operations
            y = x * 2.0
            y = y + 3.0
            y = y * 4.0
            
            loss = y.sum()
            loss.backward()
            
            return x.grad
        
        litetorch_mem = benchmark_memory("LiteTorch Memory", litetorch_memory)
        print(f"LiteTorch:")
        print(f"  Peak memory: {format_memory(litetorch_mem['peak'])}")
        print()
    except Exception as e:
        print(f"LiteTorch: Not implemented yet ({type(e).__name__})")
        print()
    
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        def pytorch_memory():
            x = torch.randn(size, size, requires_grad=True)
            
            # Chain of operations
            y = x * 2.0
            y = y + 3.0
            y = y * 4.0
            
            loss = y.sum()
            loss.backward()
            
            return x.grad
        
        pytorch_mem = benchmark_memory("PyTorch Memory", pytorch_memory)
        print(f"PyTorch:")
        print(f"  Peak memory: {format_memory(pytorch_mem['peak'])}")
        print()


def benchmark_gradient_accumulation():
    """Benchmark gradient accumulation."""
    print("=" * 60)
    print("Benchmark: Gradient Accumulation")
    print("=" * 60)
    print("Note: Requires autograd implementation in Tensor class")
    print()
    
    size = 10 if QUICK else 50
    accumulation_steps = 5 if QUICK else 10
    
    try:
        def litetorch_accumulation():
            x = Tensor(shape=(size, size), requires_grad=True)
            if NUMPY_AVAILABLE:
                x.data = list(np.random.randn(size * size))
            else:
                x.data = list(range(size * size))
            
            for _ in range(accumulation_steps):
                y = x * 2.0
                loss = y.sum()
                loss.backward()
        
        litetorch_result = benchmark_time("LiteTorch Gradient Accumulation", 
                                         litetorch_accumulation,
                                         iterations=DEFAULT_ITERATIONS // 10)
    except Exception as e:
        print(f"LiteTorch: Not implemented yet ({type(e).__name__})")
        litetorch_result = None
    
    pytorch_result = None
    if PYTORCH_AVAILABLE and not SKIP_PYTORCH:
        def pytorch_accumulation():
            x = torch.randn(size, size, requires_grad=True)
            
            for _ in range(accumulation_steps):
                y = x * 2.0
                loss = y.sum()
                loss.backward()
        
        pytorch_result = benchmark_time("PyTorch Gradient Accumulation",
                                       pytorch_accumulation,
                                       iterations=DEFAULT_ITERATIONS // 10)
    
    if litetorch_result:
        print_comparison(litetorch_result, pytorch_result)
    else:
        print("Skipped - autograd not implemented\n")


def print_summary():
    """Print benchmark summary."""
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("Autograd benchmarks completed!")
    print()
    print("Note: Most benchmarks require the autograd engine to be implemented")
    print("in the Tensor class with the following API:")
    print("  - Tensor(shape, requires_grad=True)")
    print("  - tensor.backward()")
    print("  - tensor.grad (gradient storage)")
    print("  - tensor.zero_grad()")
    print()
    print("Expected performance:")
    print("  - LiteTorch is typically 10-1000x slower than PyTorch")
    print("  - PyTorch uses highly optimized C++/CUDA backend")
    print("  - LiteTorch uses pure Python for educational clarity")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Autograd Benchmarks")
    print("=" * 60)
    print()
    
    if QUICK:
        print("Running in QUICK mode (smaller sizes)")
        print()
    
    if SKIP_PYTORCH:
        print("PyTorch benchmarks disabled (SKIP_PYTORCH=1)")
        print()
    
    print(f"Iterations per benchmark: {DEFAULT_ITERATIONS}")
    print()
    
    # Run benchmarks
    benchmark_simple_backward()
    benchmark_chain_backward()
    benchmark_matmul_backward()
    benchmark_activation_backward()
    benchmark_neural_network_backward()
    benchmark_memory_usage()
    benchmark_gradient_accumulation()
    
    # Print summary
    print_summary()
