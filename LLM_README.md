# LLM Implementation - Complete Guide

This guide provides comprehensive documentation for the LiteTorch LLM (Large Language Model) implementation, including the GPT architecture built from scratch.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Usage](#usage)
5. [Training](#training)
6. [Generation](#generation)
7. [Examples](#examples)
8. [Benchmarks](#benchmarks)
9. [API Reference](#api-reference)

---

## Overview

This implementation provides a complete, educational implementation of GPT (Generative Pre-trained Transformer) architecture. It includes:

- ✅ **Attention Mechanisms**: Scaled dot-product attention and multi-head attention
- ✅ **Transformer Blocks**: Complete decoder blocks with self-attention and feed-forward networks
- ✅ **Position Encodings**: Both learned and sinusoidal positional embeddings
- ✅ **GPT Model**: Full autoregressive language model
- ✅ **Text Generation**: Autoregressive sampling with temperature and top-k/top-p
- ✅ **Comprehensive Tests**: 24 unit tests covering all components
- ✅ **Benchmarks**: Performance measurements and comparisons
- ✅ **Examples**: 8 detailed examples showing different use cases

### Key Features

- **Educational Focus**: Clear, well-commented code designed for learning
- **Pure NumPy**: Built from scratch using only NumPy (no deep learning frameworks)
- **Complete Implementation**: All core components fully implemented
- **Production Patterns**: Follows best practices from modern LLM implementations
- **Extensible**: Easy to modify and experiment with

---

## Architecture

### GPT Architecture Overview

```
Input Token IDs
      ↓
[Token Embeddings] + [Position Embeddings]
      ↓
[Transformer Block 1]
      ↓
[Transformer Block 2]
      ↓
      ...
      ↓
[Transformer Block N]
      ↓
[Layer Normalization]
      ↓
[Language Modeling Head]
      ↓
Output Logits
```

### Transformer Block

Each transformer block consists of:

```
Input
  ↓
[Layer Norm] → [Multi-Head Attention] → [Residual Add]
  ↓
[Layer Norm] → [Feed-Forward Network] → [Residual Add]
  ↓
Output
```

### Multi-Head Attention

```
Query, Key, Value
       ↓
[Linear Projections] (Q, K, V)
       ↓
[Split into H heads]
       ↓
[Scaled Dot-Product Attention] (per head)
       ↓
[Concatenate heads]
       ↓
[Linear Projection] (output)
       ↓
Output
```

---

## Components

### 1. Attention Mechanisms (`litetorch/nn/attention.py`)

#### ScaledDotProductAttention

Implements the fundamental attention mechanism:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

**Features:**
- Scaled dot-product computation
- Optional attention masking (causal, padding)
- Dropout for regularization

#### MultiHeadAttention

Implements multi-head attention with parallel attention heads:

**Features:**
- Splits hidden dimension across multiple heads
- Parallel attention computation
- Learned projections for Q, K, V
- Output projection

#### Helper Functions

- `create_causal_mask(seq_len)`: Creates autoregressive mask
- `create_padding_mask(seq_lengths, max_len)`: Creates padding mask

### 2. Neural Network Layers (`litetorch/nn/layers.py`)

#### LayerNorm

Layer normalization for stable training:

**Features:**
- Normalizes across feature dimension
- Learnable scale (gamma) and shift (beta) parameters

#### FeedForward

Position-wise feed-forward network (MLP):

**Features:**
- Two linear transformations with non-linearity
- GELU or ReLU activation
- Dropout for regularization

#### Embedding

Token embedding layer:

**Features:**
- Maps token IDs to dense vectors
- Xavier initialization

#### PositionalEncoding

Fixed sinusoidal position encodings:

**Features:**
- Pre-computed position embeddings
- Based on sine and cosine functions

#### LearnedPositionalEncoding

Learned position embeddings (used in GPT):

**Features:**
- Trainable position embeddings
- One embedding per position

#### TransformerBlock

Complete transformer decoder block:

**Features:**
- Self-attention with residual connection
- Feed-forward network with residual connection
- Layer normalization (pre-norm architecture)

### 3. GPT Model (`litetorch/nn/gpt.py`)

Complete GPT implementation:

**Features:**
- Token and position embeddings
- Stack of transformer blocks
- Language modeling head
- Autoregressive text generation
- Loss computation

---

## Usage

### Basic Model Creation

```python
from litetorch.nn.gpt import GPTModel

# Create a GPT model
model = GPTModel(
    vocab_size=50257,      # GPT-2 vocabulary size
    max_seq_len=1024,      # Context window
    num_layers=12,         # Transformer blocks
    num_heads=12,          # Attention heads per block
    hidden_dim=768,        # Model dimension
    ffn_dim=3072,          # FFN dimension (4x hidden_dim)
    dropout=0.1            # Dropout probability
)

# Get model info
num_params = model.get_num_parameters()
print(f"Model has {num_params:,} parameters")
```

### Forward Pass

```python
import numpy as np

# Create input token IDs
batch_size = 4
seq_len = 128
input_ids = np.random.randint(0, 50257, size=(batch_size, seq_len))

# Forward pass
logits = model.forward(input_ids)
# Output shape: (batch_size, seq_len, vocab_size)
```

### Text Generation

```python
# Starting prompt
prompt = np.array([1, 2, 3, 4, 5])  # Token IDs

# Generate text
generated = model.generate(
    prompt,
    max_new_tokens=50,     # Generate 50 tokens
    temperature=1.0,       # Sampling temperature
    top_k=50,              # Top-k filtering
    top_p=0.95             # Nucleus sampling
)

print(f"Generated: {generated}")
```

### Training

```python
# Create training data
input_ids = np.random.randint(0, 50257, size=(batch_size, seq_len))
target_ids = np.roll(input_ids, -1, axis=1)  # Shift by 1

# Compute loss
loss = model.compute_loss(input_ids, target_ids)
print(f"Loss: {loss:.4f}")
print(f"Perplexity: {np.exp(loss):.2f}")
```

---

## Training

### Training Loop Structure

```python
# Training hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 3e-4

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Get batch
        input_ids, target_ids = batch
        
        # Forward pass
        loss = model.compute_loss(input_ids, target_ids)
        
        # Backward pass (not implemented in this educational version)
        # In practice, you would compute gradients here
        
        # Update weights (not implemented)
        # optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### Loss Function

The model uses cross-entropy loss for next-token prediction:

```
Loss = -mean(log P(target | input))
```

Where P(target | input) is the predicted probability of the target token.

---

## Generation

### Sampling Strategies

#### 1. Greedy Sampling

Always select the most likely token:

```python
# Set temperature to very low value
generated = model.generate(prompt, temperature=0.1)
```

#### 2. Temperature Sampling

Control randomness with temperature:

```python
# Low temperature (0.5) - more focused
generated = model.generate(prompt, temperature=0.5)

# High temperature (1.5) - more diverse
generated = model.generate(prompt, temperature=1.5)
```

#### 3. Top-k Sampling

Sample from top k most likely tokens:

```python
generated = model.generate(prompt, top_k=50)
```

#### 4. Top-p (Nucleus) Sampling

Sample from smallest set of tokens with cumulative probability >= p:

```python
generated = model.generate(prompt, top_p=0.95)
```

#### 5. Combined

Combine multiple strategies:

```python
generated = model.generate(
    prompt,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
```

---

## Examples

Run the examples:

```bash
cd /home/runner/work/litetorch/litetorch
python examples/llm_examples.py
```

### Example 1: Basic Model Creation
Creates a GPT model and inspects its configuration and parameter count.

### Example 2: Forward Pass
Performs a forward pass and examines the output.

### Example 3: Text Generation
Demonstrates text generation with different temperature settings.

### Example 4: Training Step
Shows how to compute loss for language modeling.

### Example 5: Simple Training Loop
Simulates a training loop with multiple steps.

### Example 6: Attention Visualization
Visualizes attention patterns with and without causal masking.

### Example 7: Model Size Comparison
Compares different model sizes and their memory requirements.

### Example 8: Batch Generation
Generates multiple sequences in parallel.

---

## Benchmarks

Run benchmarks:

```bash
# Quick benchmarks (~1 minute)
QUICK=1 python benchmarks/bench_llm.py

# Full benchmarks (~5 minutes)
python benchmarks/bench_llm.py
```

### Performance Results (Quick Mode)

**Attention Mechanisms:**
- SDPA (small): ~0.09 ms
- MHA (small): ~0.30 ms

**Transformer Block:**
- Small: ~1.35 ms
- Medium: ~10.88 ms

**GPT Forward Pass:**
- Tiny (32 hidden): ~1 ms, 31k tokens/sec
- Small (64 hidden): ~8 ms, 7.7k tokens/sec

**Text Generation:**
- 10 tokens: ~20 ms (490 tokens/sec)
- 20 tokens: ~38 ms (522 tokens/sec)

---

## API Reference

### GPTModel

```python
class GPTModel:
    def __init__(self, vocab_size, max_seq_len, num_layers, 
                 num_heads, hidden_dim, ffn_dim, dropout)
    
    def forward(self, input_ids, return_hidden_states=False)
    def generate(self, prompt_ids, max_new_tokens, temperature, top_k, top_p)
    def compute_loss(self, input_ids, target_ids)
    def get_num_parameters(self)
```

### MultiHeadAttention

```python
class MultiHeadAttention:
    def __init__(self, hidden_dim, num_heads, dropout)
    def forward(self, query, key, value, mask=None)
```

### TransformerBlock

```python
class TransformerBlock:
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout)
    def forward(self, x, mask=None)
```

---

## Testing

Run tests:

```bash
# Run all LLM tests
python -m pytest tests/test_llm.py -v

# Run specific test class
python -m pytest tests/test_llm.py::TestGPTModel -v

# Run specific test
python -m pytest tests/test_llm.py::TestGPTModel::test_gpt_generation -v
```

### Test Coverage

- **24 comprehensive tests**
- Attention mechanisms (7 tests)
- Neural network layers (7 tests)
- GPT model (8 tests)
- End-to-end scenarios (2 tests)

---

## Model Sizes

### Predefined Configurations

| Name | Vocab | Layers | Heads | Hidden | FFN | Params | Memory |
|------|-------|--------|-------|--------|-----|--------|--------|
| Tiny | 500 | 2 | 2 | 32 | 128 | 59K | 0.2 MB |
| Small | 1,000 | 4 | 4 | 64 | 256 | 335K | 1.3 MB |
| Medium | 5,000 | 6 | 6 | 192 | 768 | 3M | 11.7 MB |
| Large | 10,000 | 12 | 8 | 256 | 1024 | ~10M | ~40 MB |

---

## References

### Papers

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original transformer architecture
   - Introduced scaled dot-product attention and multi-head attention

2. **Improving Language Understanding by Generative Pre-Training** (Radford et al., 2018)
   - GPT architecture
   - Decoder-only transformer for language modeling

3. **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019)
   - GPT-2
   - Scaling transformer language models

4. **Language Models are Few-Shot Learners** (Brown et al., 2020)
   - GPT-3
   - In-context learning with large language models

### Implementation Notes

- Uses **pre-norm** architecture (LayerNorm before sub-layers) as in GPT-2
- Uses **learned position embeddings** instead of sinusoidal
- Uses **GELU activation** in feed-forward networks
- Implements **causal masking** for autoregressive generation

---

## Future Enhancements

Potential extensions to this implementation:

1. **Gradient computation and backpropagation**
2. **Optimizer implementations** (Adam, AdamW)
3. **Training utilities** (learning rate scheduling, gradient clipping)
4. **More efficient attention** (Flash Attention, sparse attention)
5. **Model parallelism** for larger models
6. **Quantization** for inference optimization
7. **KV cache** for faster generation
8. **Beam search** decoding

---

## Conclusion

This LiteTorch LLM implementation provides a complete, educational foundation for understanding how modern language models work. All core components are implemented from scratch using only NumPy, making it easy to understand and modify.

For questions or contributions, please refer to the main LiteTorch repository.
