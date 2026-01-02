"""
Benchmark tensor operations against PyTorch.

This script compares the performance of basic tensor operations
between LiteTorch and PyTorch.
"""
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litetorch.autograd.tensor import Tensor

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


def benchmark_operation(name, litetorch_fn, pytorch_fn, iterations=1000):
    """Benchmark a single operation."""
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
        slower_factor = litetorch_time / pytorch_time if pytorch_time > 0 else float('inf')
        
        print(f"{name}:")
        print(f"  LiteTorch: {litetorch_time:.4f}s ({iterations} iterations)")
        print(f"  PyTorch:   {pytorch_time:.4f}s ({iterations} iterations)")
        if slower_factor >= 1:
            print(f"  LiteTorch is {slower_factor:.2f}x slower than PyTorch")
        else:
            print(f"  LiteTorch is {speedup:.2f}x faster than PyTorch")
        print(f"  Per-op LiteTorch: {litetorch_time/iterations*1000:.4f}ms")
        print(f"  Per-op PyTorch:   {pytorch_time/iterations*1000:.4f}ms")
    else:
        print(f"{name}:")
        print(f"  LiteTorch: {litetorch_time:.4f}s ({iterations} iterations)")
        print(f"  Per-op:    {litetorch_time/iterations*1000:.4f}ms")
    print()


def benchmark_tensor_creation():
    """Benchmark tensor creation."""
    print("=" * 60)
    print("Benchmark: Tensor Creation")
    print("=" * 60)
    
    # Small tensors (2x2)
    def litetorch_create_small():
        t = Tensor(shape=(2, 2))
        t.data = [1, 2, 3, 4]
    
    def pytorch_create_small():
        t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    
    benchmark_operation("Small Tensor (2x2)", litetorch_create_small, pytorch_create_small, iterations=10000)
    
    # Medium tensors (10x10)
    def litetorch_create_medium():
        t = Tensor(shape=(10, 10))
    
    def pytorch_create_medium():
        t = torch.zeros(10, 10, dtype=torch.float32)
    
    benchmark_operation("Medium Tensor (10x10)", litetorch_create_medium, pytorch_create_medium, iterations=5000)
    
    # Large tensors (100x100)
    def litetorch_create_large():
        t = Tensor(shape=(100, 100))
    
    def pytorch_create_large():
        t = torch.zeros(100, 100, dtype=torch.float32)
    
    benchmark_operation("Large Tensor (100x100)", litetorch_create_large, pytorch_create_large, iterations=1000)
    
    # Very large tensors (1000x1000)
    def litetorch_create_xlarge():
        t = Tensor(shape=(1000, 1000))
    
    def pytorch_create_xlarge():
        t = torch.zeros(1000, 1000, dtype=torch.float32)
    
    benchmark_operation("XLarge Tensor (1000x1000)", litetorch_create_xlarge, pytorch_create_xlarge, iterations=100)


def benchmark_tensor_addition():
    """Benchmark tensor addition."""
    print("=" * 60)
    print("Benchmark: Tensor Addition")
    print("=" * 60)
    
    # Small tensors (10x10)
    t1_small = Tensor(shape=(10, 10))
    t2_small = Tensor(shape=(10, 10))
    t1_small.data = [i for i in range(100)]
    t2_small.data = [i*2 for i in range(100)]
    
    if PYTORCH_AVAILABLE:
        pt1_small = torch.randn(10, 10)
        pt2_small = torch.randn(10, 10)
    
    def litetorch_add_small():
        result = t1_small + t2_small
    
    def pytorch_add_small():
        c = pt1_small + pt2_small
    
    benchmark_operation("Small Addition (10x10)", litetorch_add_small, pytorch_add_small, iterations=5000)
    
    # Medium tensors (50x50)
    t1_med = Tensor(shape=(50, 50))
    t2_med = Tensor(shape=(50, 50))
    t1_med.data = [i for i in range(2500)]
    t2_med.data = [i*2 for i in range(2500)]
    
    if PYTORCH_AVAILABLE:
        pt1_med = torch.randn(50, 50)
        pt2_med = torch.randn(50, 50)
    
    def litetorch_add_medium():
        result = t1_med + t2_med
    
    def pytorch_add_medium():
        c = pt1_med + pt2_med
    
    benchmark_operation("Medium Addition (50x50)", litetorch_add_medium, pytorch_add_medium, iterations=1000)
    
    # Large tensors (100x100)
    t1_large = Tensor(shape=(100, 100))
    t2_large = Tensor(shape=(100, 100))
    t1_large.data = [i for i in range(10000)]
    t2_large.data = [i*2 for i in range(10000)]
    
    if PYTORCH_AVAILABLE:
        pt1_large = torch.randn(100, 100)
        pt2_large = torch.randn(100, 100)
    
    def litetorch_add_large():
        result = t1_large + t2_large
    
    def pytorch_add_large():
        c = pt1_large + pt2_large
    
    benchmark_operation("Large Addition (100x100)", litetorch_add_large, pytorch_add_large, iterations=500)


def benchmark_tensor_operations():
    """Benchmark various tensor operations."""
    print("=" * 60)
    print("Benchmark: Various Tensor Operations")
    print("=" * 60)
    
    # Test data assignment
    t = Tensor(shape=(50, 50))
    
    def litetorch_data_assignment():
        t.data = [i for i in range(2500)]
    
    def pytorch_data_assignment():
        t = torch.tensor([i for i in range(2500)], dtype=torch.float32).reshape(50, 50)
    
    benchmark_operation("Data Assignment (50x50)", litetorch_data_assignment, pytorch_data_assignment, iterations=1000)
    
    # Test shape access
    t = Tensor(shape=(100, 100))
    
    def litetorch_shape_access():
        s = t.shape
    
    def pytorch_shape_access():
        pt = torch.zeros(100, 100)
        s = pt.shape
    
    benchmark_operation("Shape Access", litetorch_shape_access, pytorch_shape_access, iterations=100000)
    
    # Test data access
    t = Tensor(shape=(100, 100))
    t.data = [i for i in range(10000)]
    
    def litetorch_data_iteration():
        total = sum(t.data)
    
    def pytorch_data_iteration():
        pt = torch.randn(100, 100)
        total = pt.sum().item()
    
    benchmark_operation("Data Iteration/Sum (100x100)", litetorch_data_iteration, pytorch_data_iteration, iterations=1000)


def benchmark_memory_efficiency():
    """Benchmark memory footprint of tensors."""
    print("=" * 60)
    print("Benchmark: Memory Efficiency Comparison")
    print("=" * 60)
    
    import sys
    
    # Test memory for small tensor
    t_small = Tensor(shape=(10, 10))
    litetorch_small_size = sys.getsizeof(t_small.data) + sys.getsizeof(t_small.shape)
    
    if PYTORCH_AVAILABLE:
        pt_small = torch.zeros(10, 10)
        pytorch_small_size = pt_small.element_size() * pt_small.nelement()
        
        print(f"Small Tensor (10x10) Memory:")
        print(f"  LiteTorch: ~{litetorch_small_size} bytes")
        print(f"  PyTorch:   ~{pytorch_small_size} bytes")
        print()
    
    # Test memory for large tensor
    t_large = Tensor(shape=(100, 100))
    litetorch_large_size = sys.getsizeof(t_large.data) + sys.getsizeof(t_large.shape)
    
    if PYTORCH_AVAILABLE:
        pt_large = torch.zeros(100, 100)
        pytorch_large_size = pt_large.element_size() * pt_large.nelement()
        
        print(f"Large Tensor (100x100) Memory:")
        print(f"  LiteTorch: ~{litetorch_large_size} bytes")
        print(f"  PyTorch:   ~{pytorch_large_size} bytes")
        print()


def benchmark_edge_cases():
    """Benchmark edge case scenarios."""
    print("=" * 60)
    print("Benchmark: Edge Cases")
    print("=" * 60)
    
    # Single element tensor
    def litetorch_single():
        t = Tensor(shape=(1,))
        t.data = [5.0]
    
    def pytorch_single():
        t = torch.tensor([5.0])
    
    benchmark_operation("Single Element Tensor", litetorch_single, pytorch_single, iterations=10000)
    
    # Scalar tensor
    def litetorch_scalar():
        t = Tensor(shape=())
        t.data = [5.0]
    
    def pytorch_scalar():
        t = torch.tensor(5.0)
    
    benchmark_operation("Scalar Tensor", litetorch_scalar, pytorch_scalar, iterations=10000)
    
    # 1D tensor
    def litetorch_1d():
        t = Tensor(shape=(100,))
    
    def pytorch_1d():
        t = torch.zeros(100)
    
    benchmark_operation("1D Tensor (100)", litetorch_1d, pytorch_1d, iterations=5000)
    
    # 3D tensor
    def litetorch_3d():
        t = Tensor(shape=(10, 10, 10))
    
    def pytorch_3d():
        t = torch.zeros(10, 10, 10)
    
    benchmark_operation("3D Tensor (10x10x10)", litetorch_3d, pytorch_3d, iterations=1000)
    
    # Multiple sequential additions
    t1 = Tensor(shape=(20, 20))
    t2 = Tensor(shape=(20, 20))
    t1.data = [i for i in range(400)]
    t2.data = [i*2 for i in range(400)]
    
    if PYTORCH_AVAILABLE:
        pt_a = torch.randn(20, 20)
        pt_b = torch.randn(20, 20)
    
    def litetorch_sequential_ops():
        r1 = t1 + t2
        r2 = r1 + t1
        r3 = r2 + t2
    
    def pytorch_sequential_ops():
        r1 = pt_a + pt_b
        r2 = r1 + pt_a
        r3 = r2 + pt_b
    
    benchmark_operation("Sequential Additions (20x20)", litetorch_sequential_ops, pytorch_sequential_ops, iterations=1000)


if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("      TENSOR OPERATIONS BENCHMARK: LiteTorch vs PyTorch")
    print("=" * 60)
    print()
    
    if not PYTORCH_AVAILABLE:
        print("WARNING: PyTorch not available. Only LiteTorch will be benchmarked.")
        print("Install PyTorch with: pip install torch")
        print()
    
    benchmark_tensor_creation()
    benchmark_tensor_addition()
    benchmark_tensor_operations()
    benchmark_memory_efficiency()
    benchmark_edge_cases()
    
    print()
    print("=" * 60)
    print("                 Benchmarks Complete!")
    print("=" * 60)
    print()
    print("Summary:")
    print("- LiteTorch is a pure Python implementation focused on education")
    print("- PyTorch uses optimized C++/CUDA backend for production performance")
    print("- LiteTorch will be slower but offers clearer educational value")
    print("- For production use cases, always prefer PyTorch or similar frameworks")
    print()
