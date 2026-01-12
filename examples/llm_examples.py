"""
Examples demonstrating LLM training and text generation.

This script shows how to use the LiteTorch LLM implementation for:
1. Building a GPT model
2. Training on a simple dataset
3. Generating text
4. Understanding model internals
"""
import numpy as np
from litetorch.nn.gpt import GPTModel


def example_1_basic_model_creation():
    """Example 1: Create and inspect a GPT model."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Model Creation")
    print("="*60)
    
    # Create a small GPT model
    model = GPTModel(
        vocab_size=1000,      # Vocabulary size
        max_seq_len=128,      # Maximum sequence length (context window)
        num_layers=4,         # Number of transformer blocks
        num_heads=4,          # Number of attention heads per block
        hidden_dim=128,       # Hidden dimension (embedding size)
        ffn_dim=512,          # Feed-forward network dimension (4x hidden_dim)
        dropout=0.1           # Dropout probability
    )
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary size:     {model.vocab_size}")
    print(f"  Maximum seq length:  {model.max_seq_len}")
    print(f"  Number of layers:    {model.num_layers}")
    print(f"  Attention heads:     {model.num_heads}")
    print(f"  Hidden dimension:    {model.hidden_dim}")
    print(f"  FFN dimension:       {model.ffn_dim}")
    
    # Get parameter count
    num_params = model.get_num_parameters()
    print(f"\nTotal parameters:      {num_params:,}")
    print(f"Estimated memory:      {num_params * 4 / (1024**2):.2f} MB (float32)")
    
    print("\n✓ Model created successfully!")


def example_2_forward_pass():
    """Example 2: Perform a forward pass through the model."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Forward Pass")
    print("="*60)
    
    # Create model
    model = GPTModel(
        vocab_size=500,
        max_seq_len=64,
        num_layers=2,
        num_heads=4,
        hidden_dim=64
    )
    
    # Create random input tokens
    # In practice, these would be real token IDs from a tokenizer
    batch_size = 2
    seq_len = 10
    input_ids = np.random.randint(0, 500, size=(batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input tokens (first sequence): {input_ids[0]}")
    
    # Forward pass
    logits = model.forward(input_ids)
    
    print(f"\nOutput shape: {logits.shape}")
    print(f"Output is logits over vocabulary for each position")
    
    # Get predictions (most likely next token at each position)
    predictions = np.argmax(logits, axis=-1)
    print(f"\nPredicted tokens (first sequence): {predictions[0]}")
    
    print("\n✓ Forward pass completed!")


def example_3_text_generation():
    """Example 3: Generate text autoregressively."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Text Generation")
    print("="*60)
    
    # Create model
    model = GPTModel(
        vocab_size=500,
        max_seq_len=128,
        num_layers=4,
        num_heads=4,
        hidden_dim=64
    )
    
    # Starting prompt (in practice, this would be tokenized text)
    prompt = np.array([1, 2, 3, 4, 5])
    print(f"\nPrompt tokens: {prompt}")
    
    # Generate with different settings
    print("\n--- Generation with temperature=1.0 (balanced) ---")
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"Generated tokens: {generated[0]}")
    
    print("\n--- Generation with temperature=0.5 (more focused) ---")
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.5)
    print(f"Generated tokens: {generated[0]}")
    
    print("\n--- Generation with temperature=2.0 (more random) ---")
    generated = model.generate(prompt, max_new_tokens=10, temperature=2.0)
    print(f"Generated tokens: {generated[0]}")
    
    print("\n✓ Text generation completed!")


def example_4_training_step():
    """Example 4: Perform a training step."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Training Step")
    print("="*60)
    
    # Create model
    model = GPTModel(
        vocab_size=500,
        max_seq_len=64,
        num_layers=2,
        num_heads=4,
        hidden_dim=64
    )
    
    # Create training data
    # In language modeling, targets are typically inputs shifted by 1 position
    batch_size = 4
    seq_len = 20
    input_ids = np.random.randint(0, 500, size=(batch_size, seq_len))
    
    # Targets are next tokens (shifted input)
    target_ids = np.roll(input_ids, -1, axis=1)
    # Set last token target to a padding token (not used in loss)
    target_ids[:, -1] = 0
    
    print(f"\nTraining batch shape: {input_ids.shape}")
    print(f"First sequence input:  {input_ids[0, :10]}")
    print(f"First sequence target: {target_ids[0, :10]}")
    
    # Compute loss
    loss = model.compute_loss(input_ids, target_ids)
    
    print(f"\nCross-entropy loss: {loss:.4f}")
    print(f"Perplexity: {np.exp(loss):.2f}")
    
    print("\n✓ Training step completed!")
    print("  (In a real training loop, you would now compute gradients and update weights)")


def example_5_simple_training_loop():
    """Example 5: Simple training loop simulation."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Simple Training Loop")
    print("="*60)
    
    # Create model
    model = GPTModel(
        vocab_size=100,
        max_seq_len=32,
        num_layers=2,
        num_heads=2,
        hidden_dim=32
    )
    
    print("\nSimulating training for 10 steps...")
    print("(Note: This is just forward pass + loss, no actual weight updates)")
    
    losses = []
    for step in range(10):
        # Generate random training batch
        batch_size = 4
        seq_len = 16
        input_ids = np.random.randint(0, 100, size=(batch_size, seq_len))
        target_ids = np.roll(input_ids, -1, axis=1)
        
        # Compute loss
        loss = model.compute_loss(input_ids, target_ids)
        losses.append(loss)
        
        if step % 2 == 0:
            print(f"  Step {step:2d}: loss = {loss:.4f}")
    
    print(f"\nAverage loss: {np.mean(losses):.4f}")
    print("\n✓ Training loop completed!")


def example_6_attention_visualization():
    """Example 6: Understanding attention patterns."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Attention Patterns")
    print("="*60)
    
    from litetorch.nn.attention import ScaledDotProductAttention, create_causal_mask
    
    # Create attention mechanism
    attention = ScaledDotProductAttention(dropout=0.0)
    
    # Small sequence for visualization
    seq_len = 5
    d_k = 8
    
    query = np.random.randn(1, seq_len, d_k)
    key = np.random.randn(1, seq_len, d_k)
    value = np.random.randn(1, seq_len, d_k)
    
    # Without mask (bidirectional attention)
    print("\n--- Without causal mask (bidirectional) ---")
    output, attn_weights = attention.forward(query, key, value)
    print("Attention weights (each row sums to 1.0):")
    print(attn_weights[0].round(3))
    
    # With causal mask (autoregressive)
    print("\n--- With causal mask (autoregressive) ---")
    mask = create_causal_mask(seq_len)
    output, attn_weights = attention.forward(query, key, value, mask=mask)
    print("Attention weights (position i can only attend to j <= i):")
    print(attn_weights[0].round(3))
    print("\nNotice: Upper triangular values are ~0 (masked out)")
    
    print("\n✓ Attention visualization completed!")


def example_7_model_sizes():
    """Example 7: Compare different model sizes."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Model Size Comparison")
    print("="*60)
    
    configs = [
        ("Tiny", 500, 64, 2, 2, 32, 128),
        ("Small", 1000, 128, 4, 4, 64, 256),
        ("Medium", 5000, 256, 6, 6, 192, 768),
        ("Large", 10000, 512, 12, 8, 256, 1024),
    ]
    
    print("\nModel size comparison:")
    print("-" * 80)
    print(f"{'Name':<10} {'Vocab':<8} {'Layers':<8} {'Heads':<8} {'Hidden':<8} {'Params':<15} {'Memory'}")
    print("-" * 80)
    
    for name, vocab, max_len, layers, heads, hidden, ffn in configs:
        model = GPTModel(
            vocab_size=vocab,
            max_seq_len=max_len,
            num_layers=layers,
            num_heads=heads,
            hidden_dim=hidden,
            ffn_dim=ffn
        )
        
        num_params = model.get_num_parameters()
        memory_mb = num_params * 4 / (1024**2)
        
        print(f"{name:<10} {vocab:<8} {layers:<8} {heads:<8} {hidden:<8} "
              f"{num_params:<15,} {memory_mb:.1f} MB")
    
    print("\n✓ Model size comparison completed!")


def example_8_batch_generation():
    """Example 8: Generate multiple sequences in parallel."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Batch Generation")
    print("="*60)
    
    # Create model
    model = GPTModel(
        vocab_size=500,
        max_seq_len=128,
        num_layers=2,
        num_heads=4,
        hidden_dim=64
    )
    
    # Multiple prompts
    prompts = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    print(f"\nBatch of {len(prompts)} prompts:")
    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}: {prompt}")
    
    # Generate for all prompts at once
    generated = model.generate(prompts, max_new_tokens=8, temperature=1.0)
    
    print(f"\nGenerated sequences:")
    for i, seq in enumerate(generated):
        print(f"  Sequence {i+1}: {seq}")
    
    print("\n✓ Batch generation completed!")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*15 + "LLM TRAINING & GENERATION EXAMPLES")
    print("="*70)
    
    print("\nThese examples demonstrate the LiteTorch LLM implementation.")
    print("Each example shows a different aspect of using the GPT model.")
    
    examples = [
        example_1_basic_model_creation,
        example_2_forward_pass,
        example_3_text_generation,
        example_4_training_step,
        example_5_simple_training_loop,
        example_6_attention_visualization,
        example_7_model_sizes,
        example_8_batch_generation,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n✗ Error in {example.__name__}: {e}")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run all examples
    run_all_examples()
