"""
Neural Network modules and layers.

This module contains neural network components including:
- Attention mechanisms (scaled dot-product, multi-head attention)
- Core layers (LayerNorm, FeedForward, Embeddings, etc.)
- Transformer blocks
- Complete GPT model implementation
- CNN layers and architectures (Conv2d, ResNet, VGG, etc.)
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

# Import CNN layers and architectures
from litetorch.nn.cnn import (
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    BatchNorm2d,
    Dropout2d,
    VGGBlock,
    ResidualBlock,
    BottleneckBlock,
    InceptionModule,
    DepthwiseSeparableConv2d,
    LeNet5,
    AlexNet,
    VGG,
    ResNet,
)

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
    # CNN Layers
    'Conv2d',
    'MaxPool2d',
    'AvgPool2d',
    'BatchNorm2d',
    'Dropout2d',
    # CNN Blocks
    'VGGBlock',
    'ResidualBlock',
    'BottleneckBlock',
    'InceptionModule',
    'DepthwiseSeparableConv2d',
    # CNN Architectures
    'LeNet5',
    'AlexNet',
    'VGG',
    'ResNet',
]
