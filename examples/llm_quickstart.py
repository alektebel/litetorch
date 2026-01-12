"""
Quick start example for LiteTorch LLM implementation.

This example shows how to quickly get started with the GPT model.
"""
import numpy as np
from litetorch.nn import GPTModel


def main():
    print("="*60)
    print("LiteTorch LLM - Quick Start Example")
    print("="*60)
    
    # 1. Create a small GPT model
    print("\n1. Creating GPT model...")
    model = GPTModel(
        vocab_size=1000,
        max_seq_len=128,
        num_layers=4,
        num_heads=4,
        hidden_dim=128,
        ffn_dim=512
    )
    print(f"   ✓ Model created with {model.get_num_parameters():,} parameters")
    
    # 2. Forward pass
    print("\n2. Running forward pass...")
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model.forward(input_ids)
    print(f"   ✓ Input shape: {input_ids.shape}")
    print(f"   ✓ Output shape: {logits.shape}")
    
    # 3. Generate text
    print("\n3. Generating text...")
    prompt = np.array([1, 2, 3, 4, 5])
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"   ✓ Prompt: {prompt}")
    print(f"   ✓ Generated: {generated[0]}")
    
    # 4. Compute loss
    print("\n4. Computing loss...")
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    target_ids = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    loss = model.compute_loss(input_ids, target_ids)
    print(f"   ✓ Loss: {loss:.4f}")
    print(f"   ✓ Perplexity: {np.exp(loss):.2f}")
    
    print("\n" + "="*60)
    print("Quick start completed! See LLM_README.md for more details.")
    print("="*60)


if __name__ == '__main__':
    np.random.seed(42)
    main()
