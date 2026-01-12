"""
Neural Network modules and layers.

This module contains neural network components including:
- Attention mechanisms (scaled dot-product, multi-head attention)
- Core layers (LayerNorm, FeedForward, Embeddings, etc.)
- Transformer blocks
- Complete GPT model implementation
"""

# Import attention mechanisms
from litetorch.nn.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask,
)

# Import core layers
from litetorch.nn.layers import (
    LayerNorm,
    FeedForward,
    Embedding,
    PositionalEncoding,
    LearnedPositionalEncoding,
    TransformerBlock,
)

# Import complete models
from litetorch.nn.gpt import GPTModel

__all__ = [
    # Attention
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'create_causal_mask',
    'create_padding_mask',
    # Layers
    'LayerNorm',
    'FeedForward',
    'Embedding',
    'PositionalEncoding',
    'LearnedPositionalEncoding',
    'TransformerBlock',
    # Models
    'GPTModel',
]
