"""
Benchmark automatic differentiation against PyTorch.

This script compares the performance and correctness of automatic
differentiation between LiteTorch and PyTorch.
"""
import time
import numpy as np

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


def benchmark_backward_pass(name, litetorch_fn, pytorch_fn, iterations=1000):
    """Benchmark backward pass computation."""
    # Benchmark LiteTorch
    start = time.time()
    for _ in range(iterations):
        litetorch_fn()
    litetorch_time = time.time() - start
    
    # Benchmark PyTorch
    if PYTORCH_AVAILABLE:
        start = time.time()
        for _ in range(iterations):
            pytorch_fn()
        pytorch_time = time.time() - start
        
        speedup = pytorch_time / litetorch_time if litetorch_time > 0 else float('inf')
        print(f"{name}:")
        print(f"  LiteTorch: {litetorch_time:.4f}s")
        print(f"  PyTorch:   {pytorch_time:.4f}s")
        print(f"  Speedup:   {speedup:.2f}x")
    else:
        print(f"{name}:")
        print(f"  LiteTorch: {litetorch_time:.4f}s")
    print()


def benchmark_simple_backward():
    """Benchmark simple backward pass."""
    # TODO: Implement after autograd engine is created
    # Example structure:
    # import litetorch as lt
    # 
    # def litetorch_backward():
    #     x = lt.Tensor(np.random.randn(100, 100), requires_grad=True)
    #     y = lt.Tensor(np.random.randn(100, 100), requires_grad=True)
    #     z = x * y
    #     z.backward()
    # 
    # def pytorch_backward():
    #     x = torch.randn(100, 100, requires_grad=True)
    #     y = torch.randn(100, 100, requires_grad=True)
    #     z = x * y
    #     z.sum().backward()
    # 
    # benchmark_backward_pass("Simple Backward", litetorch_backward, pytorch_backward)
    print("Simple backward pass benchmark - TODO: Implement after autograd engine")


def benchmark_chain_backward():
    """Benchmark backward pass through chain of operations."""
    # TODO: Implement after autograd engine is created
    print("Chain backward pass benchmark - TODO: Implement after autograd engine")


def benchmark_neural_network_backward():
    """Benchmark backward pass through a small neural network."""
    # TODO: Implement after autograd engine and nn layers are created
    print("Neural network backward benchmark - TODO: Implement after nn layers")


if __name__ == "__main__":
    print("=" * 50)
    print("Autograd Benchmark")
    print("=" * 50)
    print()
    
    benchmark_simple_backward()
    benchmark_chain_backward()
    benchmark_neural_network_backward()
    
    print("Benchmarks complete!")
    print("\nNote: Benchmarks will be fully implemented after autograd engine is created.")
