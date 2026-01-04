"""
Example usage of reshape and sum operations in LiteTorch.

This script demonstrates the basic usage of reshape and sum operations
on Tensor objects, showcasing various scenarios and edge cases.
"""

from litetorch.tensor import Tensor


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def example_reshape_basic():
    """Demonstrate basic reshape operations."""
    print_section("Basic Reshape Operations")
    
    # 1D to 2D
    print("\n1. Reshape 1D to 2D:")
    t = Tensor(shape=(6,))
    t.data = [1, 2, 3, 4, 5, 6]
    print(f"   Original: shape={t.shape}, data={t.data}")
    
    reshaped = t.reshape((2, 3))
    print(f"   Reshaped: shape={reshaped.shape}, data={reshaped.data}")
    
    # 2D to 1D
    print("\n2. Reshape 2D to 1D:")
    t2 = Tensor(shape=(2, 3))
    t2.data = [1, 2, 3, 4, 5, 6]
    print(f"   Original: shape={t2.shape}, data={t2.data}")
    
    reshaped2 = t2.reshape((6,))
    print(f"   Reshaped: shape={reshaped2.shape}, data={reshaped2.data}")
    
    # 3D reshaping
    print("\n3. Reshape 1D to 3D:")
    t3 = Tensor(shape=(24,))
    t3.data = list(range(24))
    print(f"   Original: shape={t3.shape}, data={t3.data[:8]}...")
    
    reshaped3 = t3.reshape((2, 3, 4))
    print(f"   Reshaped: shape={reshaped3.shape}, data={reshaped3.data[:8]}...")


def example_reshape_advanced():
    """Demonstrate advanced reshape scenarios."""
    print_section("Advanced Reshape Operations")
    
    # Flatten
    print("\n1. Flattening a multi-dimensional tensor:")
    t = Tensor(shape=(2, 3, 4))
    t.data = list(range(24))
    print(f"   Original: shape={t.shape}")
    
    flat = t.reshape((24,))
    print(f"   Flattened: shape={flat.shape}")
    
    # Add singleton dimensions
    print("\n2. Adding singleton dimensions:")
    t2 = Tensor(shape=(6,))
    t2.data = [1, 2, 3, 4, 5, 6]
    print(f"   Original: shape={t2.shape}")
    
    reshaped = t2.reshape((1, 6, 1))
    print(f"   Reshaped: shape={reshaped.shape}")
    
    # Multiple consecutive reshapes
    print("\n3. Multiple consecutive reshapes:")
    t3 = Tensor(shape=(12,))
    t3.data = list(range(12))
    print(f"   Original: shape={t3.shape}")
    
    r1 = t3.reshape((3, 4))
    print(f"   Step 1: {r1.shape}")
    
    r2 = r1.reshape((2, 6))
    print(f"   Step 2: {r2.shape}")
    
    r3 = r2.reshape((4, 3))
    print(f"   Step 3: {r3.shape}")


def example_sum_basic():
    """Demonstrate basic sum operations."""
    print_section("Basic Sum Operations")
    
    # 1D sum
    print("\n1. Sum of 1D tensor:")
    t = Tensor(shape=(5,))
    t.data = [1, 2, 3, 4, 5]
    print(f"   Tensor: {t.data}")
    print(f"   Sum: {t.sum()}")
    
    # 2D sum
    print("\n2. Sum of 2D tensor:")
    t2 = Tensor(shape=(2, 3))
    t2.data = [1, 2, 3, 4, 5, 6]
    print(f"   Tensor shape: {t2.shape}")
    print(f"   Tensor data: {t2.data}")
    print(f"   Sum: {t2.sum()}")
    
    # 3D sum
    print("\n3. Sum of 3D tensor:")
    t3 = Tensor(shape=(2, 2, 2))
    t3.data = [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"   Tensor shape: {t3.shape}")
    print(f"   Tensor data: {t3.data}")
    print(f"   Sum: {t3.sum()}")


def example_sum_edge_cases():
    """Demonstrate edge cases for sum operations."""
    print_section("Sum Edge Cases")
    
    # Single element
    print("\n1. Sum of single element:")
    t = Tensor(shape=(1,))
    t.data = [42]
    print(f"   Tensor: {t.data}")
    print(f"   Sum: {t.sum()}")
    
    # Negative values
    print("\n2. Sum with negative values:")
    t2 = Tensor(shape=(4,))
    t2.data = [-1, -2, -3, -4]
    print(f"   Tensor: {t2.data}")
    print(f"   Sum: {t2.sum()}")
    
    # Mixed positive/negative
    print("\n3. Sum with mixed values:")
    t3 = Tensor(shape=(6,))
    t3.data = [1, -2, 3, -4, 5, -6]
    print(f"   Tensor: {t3.data}")
    print(f"   Sum: {t3.sum()}")
    
    # Float values
    print("\n4. Sum with float values:")
    t4 = Tensor(shape=(4,))
    t4.data = [1.5, 2.5, 3.5, 4.5]
    print(f"   Tensor: {t4.data}")
    print(f"   Sum: {t4.sum()}")


def example_combined():
    """Demonstrate combining reshape and sum operations."""
    print_section("Combining Reshape and Sum")
    
    print("\n1. Reshape then sum:")
    t = Tensor(shape=(2, 3))
    t.data = [1, 2, 3, 4, 5, 6]
    print(f"   Original shape: {t.shape}, data: {t.data}")
    
    reshaped = t.reshape((3, 2))
    print(f"   Reshaped: {reshaped.shape}, data: {reshaped.data}")
    
    total = reshaped.sum()
    print(f"   Sum: {total}")
    
    print("\n2. Flatten then sum:")
    t2 = Tensor(shape=(2, 2, 2))
    t2.data = list(range(1, 9))
    print(f"   Original shape: {t2.shape}, data: {t2.data}")
    
    flat = t2.reshape((8,))
    print(f"   Flattened: {flat.shape}, data: {flat.data}")
    
    total2 = flat.sum()
    print(f"   Sum: {total2}")


def example_practical():
    """Demonstrate practical use cases."""
    print_section("Practical Examples")
    
    print("\n1. Batch of images (simulated):")
    # Simulate batch of 4 images, each 3x3 with 1 channel
    t = Tensor(shape=(4, 3, 3, 1))
    t.data = list(range(36))
    print(f"   Batch shape: {t.shape} (batch, height, width, channels)")
    
    # Flatten for processing
    flat = t.reshape((36,))
    print(f"   Flattened: {flat.shape}")
    
    # Calculate average pixel value
    avg = flat.sum() / len(flat.data)
    print(f"   Average pixel value: {avg:.2f}")
    
    print("\n2. Time series data:")
    # 10 time steps, 3 features
    t2 = Tensor(shape=(10, 3))
    t2.data = [i % 10 for i in range(30)]
    print(f"   Time series shape: {t2.shape} (timesteps, features)")
    print(f"   Data: {t2.data}")
    
    # Reshape to process all features together
    reshaped = t2.reshape((30,))
    print(f"   Flattened: {reshaped.shape}")
    print(f"   Total: {reshaped.sum()}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("  LiteTorch: Reshape and Sum Examples")
    print("=" * 60)
    print("\nThis script demonstrates the usage of reshape and sum")
    print("operations on Tensor objects in LiteTorch.")
    
    example_reshape_basic()
    example_reshape_advanced()
    example_sum_basic()
    example_sum_edge_cases()
    example_combined()
    example_practical()
    
    print("\n" + "=" * 60)
    print("  Examples Complete!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - tests/test_reshape_sum.py (comprehensive tests)")
    print("  - benchmarks/bench_reshape_sum.py (benchmarks)")
    print("  - RESHAPE_SUM_README.md (documentation)")
    print()


if __name__ == "__main__":
    main()
