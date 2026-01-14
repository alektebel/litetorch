"""
Benchmarks for LLM implementation.

Compares performance of attention mechanisms, transformer blocks, and GPT model
against reference implementations.
"""
import time
import numpy as np
import os

# Check if we should run quick benchmarks
QUICK_MODE = os.getenv('QUICK', '0') == '1'

# Try to import PyTorch for comparison
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")
    print("Running benchmarks without PyTorch comparison.")


def benchmark_function(name, fn, iterations=10, warmup=2):
    """
    Benchmark a function.
    
    Args:
        name: Name of the benchmark
        fn: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations
    
    Returns:
        Average time per iteration in seconds
    """
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        fn()
    elapsed = time.time() - start
    
    avg_time = elapsed / iterations
    return avg_time


def benchmark_attention():
    """Benchmark attention mechanisms."""
    print("\n" + "="*60)
    print("ATTENTION MECHANISMS BENCHMARK")
    print("="*60)
    
    from litetorch.nn.attention import ScaledDotProductAttention, MultiHeadAttention
    
    # Test configurations
    configs = [
        ("Small", 2, 32, 64),
        ("Medium", 4, 64, 128),
    ]
    
    if not QUICK_MODE:
        configs.append(("Large", 8, 128, 256))
    
    iterations = 5 if QUICK_MODE else 10
    
    for config_name, batch_size, seq_len, d_model in configs:
        print(f"\n{config_name} Configuration: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
        print("-" * 60)
        
        # LiteTorch Scaled Dot-Product Attention
        attention = ScaledDotProductAttention(dropout=0.0)
        query = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        key = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        value = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        
        def litetorch_sdpa():
            attention.forward(query, key, value)
        
        lt_time = benchmark_function("LiteTorch SDPA", litetorch_sdpa, iterations)
        print(f"  LiteTorch SDPA:  {lt_time*1000:.2f} ms")
        
        # PyTorch Scaled Dot-Product Attention
        if PYTORCH_AVAILABLE:
            query_pt = torch.from_numpy(query)
            key_pt = torch.from_numpy(key)
            value_pt = torch.from_numpy(value)
            
            def pytorch_sdpa():
                # Manual implementation since torch.nn.functional.scaled_dot_product_attention
                # might not be available in all versions
                scores = torch.matmul(query_pt, key_pt.transpose(-2, -1)) / np.sqrt(d_model)
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, value_pt)
            
            pt_time = benchmark_function("PyTorch SDPA", pytorch_sdpa, iterations)
            print(f"  PyTorch SDPA:    {pt_time*1000:.2f} ms")
            print(f"  Speedup:         {lt_time/pt_time:.2f}x (PyTorch is faster)")
        
        # Multi-Head Attention
        print(f"\n  Multi-Head Attention (4 heads):")
        mha = MultiHeadAttention(d_model, num_heads=4, dropout=0.0)
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        
        def litetorch_mha():
            mha.forward(x, x, x)
        
        lt_mha_time = benchmark_function("LiteTorch MHA", litetorch_mha, iterations)
        print(f"    LiteTorch MHA:   {lt_mha_time*1000:.2f} ms")
        
        if PYTORCH_AVAILABLE:
            mha_pt = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            x_pt = torch.from_numpy(x)
            
            def pytorch_mha():
                with torch.no_grad():
                    mha_pt(x_pt, x_pt, x_pt)
            
            pt_mha_time = benchmark_function("PyTorch MHA", pytorch_mha, iterations)
            print(f"    PyTorch MHA:     {pt_mha_time*1000:.2f} ms")
            print(f"    Speedup:         {lt_mha_time/pt_mha_time:.2f}x (PyTorch is faster)")


def benchmark_transformer_block():
    """Benchmark transformer block."""
    print("\n" + "="*60)
    print("TRANSFORMER BLOCK BENCHMARK")
    print("="*60)
    
    from litetorch.nn.layers import TransformerBlock
    
    configs = [
        ("Small", 2, 32, 64, 256),
        ("Medium", 4, 64, 128, 512),
    ]
    
    if not QUICK_MODE:
        configs.append(("Large", 8, 128, 256, 1024))
    
    iterations = 5 if QUICK_MODE else 10
    
    for config_name, batch_size, seq_len, hidden_dim, ffn_dim in configs:
        print(f"\n{config_name} Configuration: batch={batch_size}, seq_len={seq_len}, "
              f"hidden_dim={hidden_dim}, ffn_dim={ffn_dim}")
        print("-" * 60)
        
        # LiteTorch
        block = TransformerBlock(hidden_dim, num_heads=4, ffn_dim=ffn_dim, dropout=0.0)
        x = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        
        def litetorch_block():
            block.forward(x)
        
        lt_time = benchmark_function("LiteTorch Block", litetorch_block, iterations)
        print(f"  LiteTorch:       {lt_time*1000:.2f} ms")
        
        # PyTorch
        if PYTORCH_AVAILABLE:
            # PyTorch transformer encoder layer
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=ffn_dim,
                dropout=0.0,
                batch_first=True
            )
            x_pt = torch.from_numpy(x)
            
            def pytorch_block():
                with torch.no_grad():
                    layer(x_pt)
            
            pt_time = benchmark_function("PyTorch Block", pytorch_block, iterations)
            print(f"  PyTorch:         {pt_time*1000:.2f} ms")
            print(f"  Speedup:         {lt_time/pt_time:.2f}x (PyTorch is faster)")


def benchmark_gpt_forward():
    """Benchmark GPT forward pass."""
    print("\n" + "="*60)
    print("GPT FORWARD PASS BENCHMARK")
    print("="*60)
    
    from litetorch.nn.gpt import GPTModel
    
    # Model configurations
    configs = [
        ("Tiny", 100, 32, 2, 2, 32, 64),
        ("Small", 500, 64, 4, 4, 64, 256),
    ]
    
    if not QUICK_MODE:
        configs.append(("Medium", 1000, 128, 6, 6, 128, 512))
    
    iterations = 3 if QUICK_MODE else 5
    
    for config_name, vocab_size, max_seq_len, num_layers, num_heads, hidden_dim, ffn_dim in configs:
        print(f"\n{config_name} Model:")
        print(f"  vocab_size={vocab_size}, max_seq_len={max_seq_len}, "
              f"num_layers={num_layers}, hidden_dim={hidden_dim}")
        print("-" * 60)
        
        # LiteTorch GPT
        model = GPTModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim
        )
        
        batch_size = 2
        seq_len = max_seq_len // 2
        input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        
        def litetorch_forward():
            model.forward(input_ids)
        
        lt_time = benchmark_function("LiteTorch GPT", litetorch_forward, iterations)
        
        num_params = model.get_num_parameters()
        print(f"  Parameters:      {num_params:,}")
        print(f"  Forward pass:    {lt_time*1000:.2f} ms")
        print(f"  Throughput:      {batch_size * seq_len / lt_time:.0f} tokens/sec")


def benchmark_gpt_generation():
    """Benchmark GPT text generation."""
    print("\n" + "="*60)
    print("GPT GENERATION BENCHMARK")
    print("="*60)
    
    from litetorch.nn.gpt import GPTModel
    
    # Use small model for generation benchmark
    model = GPTModel(
        vocab_size=500,
        max_seq_len=128,
        num_layers=4,
        num_heads=4,
        hidden_dim=64,
        ffn_dim=256
    )
    
    prompt = np.array([1, 2, 3, 4, 5])
    
    generation_lengths = [10, 20] if QUICK_MODE else [10, 20, 50]
    
    for max_new_tokens in generation_lengths:
        print(f"\nGenerating {max_new_tokens} tokens:")
        print("-" * 60)
        
        def generate():
            model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)
        
        gen_time = benchmark_function(f"Generate {max_new_tokens}", generate, iterations=3)
        
        print(f"  Total time:      {gen_time*1000:.2f} ms")
        print(f"  Time per token:  {gen_time*1000/max_new_tokens:.2f} ms/token")
        print(f"  Tokens/sec:      {max_new_tokens/gen_time:.1f}")


def benchmark_memory_usage():
    """Benchmark memory usage of different model sizes."""
    print("\n" + "="*60)
    print("MEMORY USAGE BENCHMARK")
    print("="*60)
    
    from litetorch.nn.gpt import GPTModel
    
    configs = [
        ("Tiny", 100, 32, 2, 2, 32, 64),
        ("Small", 500, 64, 4, 4, 64, 256),
        ("Medium", 1000, 128, 6, 6, 192, 768),
    ]
    
    if not QUICK_MODE:
        configs.append(("Large", 5000, 256, 12, 8, 256, 1024))
    
    print("\nModel Size Comparison:")
    print("-" * 60)
    print(f"{'Model':<10} {'Vocab':<8} {'Layers':<8} {'Hidden':<8} {'Params':<15} {'Est. Memory'}")
    print("-" * 60)
    
    for config_name, vocab_size, max_seq_len, num_layers, num_heads, hidden_dim, ffn_dim in configs:
        model = GPTModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim
        )
        
        num_params = model.get_num_parameters()
        # Estimate memory (float32 = 4 bytes)
        memory_mb = (num_params * 4) / (1024 * 1024)
        
        print(f"{config_name:<10} {vocab_size:<8} {num_layers:<8} {hidden_dim:<8} "
              f"{num_params:<15,} {memory_mb:.2f} MB")


def benchmark_training_step():
    """Benchmark training step (forward + loss computation)."""
    print("\n" + "="*60)
    print("TRAINING STEP BENCHMARK")
    print("="*60)
    
    from litetorch.nn.gpt import GPTModel
    
    model = GPTModel(
        vocab_size=500,
        max_seq_len=128,
        num_layers=4,
        num_heads=4,
        hidden_dim=64,
        ffn_dim=256
    )
    
    batch_sizes = [1, 2, 4] if QUICK_MODE else [1, 2, 4, 8]
    seq_len = 64
    
    print(f"\nTraining step with seq_len={seq_len}:")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        input_ids = np.random.randint(0, 500, size=(batch_size, seq_len))
        target_ids = np.random.randint(0, 500, size=(batch_size, seq_len))
        
        def train_step():
            model.compute_loss(input_ids, target_ids)
        
        step_time = benchmark_function(f"Batch {batch_size}", train_step, iterations=3)
        
        print(f"  Batch size {batch_size}:  {step_time*1000:.2f} ms "
              f"({batch_size*seq_len/step_time:.0f} tokens/sec)")


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("LLM IMPLEMENTATION BENCHMARKS")
    print("="*60)
    
    if QUICK_MODE:
        print("\nRunning in QUICK mode (fewer iterations, smaller models)")
        print("Set QUICK=0 for full benchmarks")
    else:
        print("\nRunning FULL benchmarks")
        print("Set QUICK=1 for faster benchmarks")
    
    print(f"\nPyTorch available: {PYTORCH_AVAILABLE}")
    
    start_time = time.time()
    
    # Run benchmarks
    benchmark_attention()
    benchmark_transformer_block()
    benchmark_gpt_forward()
    benchmark_gpt_generation()
    benchmark_memory_usage()
    benchmark_training_step()
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"Total benchmark time: {total_time:.2f} seconds")
    print("="*60)


if __name__ == '__main__':
    run_all_benchmarks()
