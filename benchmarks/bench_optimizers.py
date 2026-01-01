"""
Benchmark optimizers against PyTorch.

This script compares the performance of optimization algorithms
between LiteTorch and PyTorch.
"""
import time
import numpy as np

try:
    import torch
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


def benchmark_optimizer(name, litetorch_fn, pytorch_fn, iterations=1000):
    """Benchmark an optimizer."""
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


def benchmark_sgd():
    """Benchmark SGD optimizer."""
    # TODO: Implement after SGD optimizer is created
    # Example structure:
    # import litetorch as lt
    # 
    # def litetorch_step():
    #     model = lt.nn.Linear(100, 10)
    #     optimizer = lt.optim.SGD(model.parameters(), lr=0.01)
    #     x = lt.Tensor(np.random.randn(32, 100))
    #     y = model(x)
    #     loss = y.sum()
    #     loss.backward()
    #     optimizer.step()
    # 
    # def pytorch_step():
    #     model = torch.nn.Linear(100, 10)
    #     optimizer = optim.SGD(model.parameters(), lr=0.01)
    #     x = torch.randn(32, 100)
    #     y = model(x)
    #     loss = y.sum()
    #     loss.backward()
    #     optimizer.step()
    # 
    # benchmark_optimizer("SGD", litetorch_step, pytorch_step)
    print("SGD optimizer benchmark - TODO: Implement after SGD optimizer")


def benchmark_adam():
    """Benchmark Adam optimizer."""
    # TODO: Implement after Adam optimizer is created
    print("Adam optimizer benchmark - TODO: Implement after Adam optimizer")


def benchmark_rmsprop():
    """Benchmark RMSprop optimizer."""
    # TODO: Implement after RMSprop optimizer is created
    print("RMSprop optimizer benchmark - TODO: Implement after RMSprop optimizer")


def benchmark_training_convergence():
    """Benchmark optimizer convergence on a simple problem."""
    # TODO: Implement after optimizers are created
    print("Training convergence benchmark - TODO: Implement after optimizers")


if __name__ == "__main__":
    print("=" * 50)
    print("Optimizers Benchmark")
    print("=" * 50)
    print()
    
    benchmark_sgd()
    benchmark_adam()
    benchmark_rmsprop()
    benchmark_training_convergence()
    
    print("Benchmarks complete!")
    print("\nNote: Benchmarks will be fully implemented after optimizers are created.")
