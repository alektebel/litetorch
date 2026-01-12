"""
Core neural network layers for building LLMs.

This module implements essential layers needed for transformer-based models,
including feed-forward networks, layer normalization, and embeddings.
"""
import numpy as np


class LayerNorm:
    """
    Layer Normalization.
    
    Normalizes inputs across the feature dimension, which helps stabilize
    training in deep networks. Unlike batch normalization, layer norm
    computes statistics independently for each example, making it suitable
    for sequential models where batch size can vary.
    
    References:
        - Ba et al. (2016): "Layer Normalization"
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize layer normalization.
        
        Args:
            normalized_shape: Shape of the input features to normalize
            eps: Small constant for numerical stability
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(normalized_shape)  # Scale parameter
        self.beta = np.zeros(normalized_shape)  # Shift parameter
    
    def forward(self, x):
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
        
        Returns:
            Normalized tensor with same shape as input
        """
        # Compute mean and variance across the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta


class FeedForward:
    """
    Position-wise Feed-Forward Network (MLP).
    
    Applies two linear transformations with a non-linearity in between:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    This is applied to each position separately and identically in the sequence.
    The hidden dimension is typically 4x the model dimension.
    
    References:
        - Vaswani et al. (2017): "Attention is All You Need"
    """
    
    def __init__(self, hidden_dim, ffn_dim, dropout=0.1, activation='relu'):
        """
        Initialize feed-forward network.
        
        Args:
            hidden_dim: Input/output dimension (d_model)
            ffn_dim: Hidden dimension (typically 4 * hidden_dim)
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
        """
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.activation = activation
        
        # Linear layers
        self.W1 = self._init_weights((hidden_dim, ffn_dim))
        self.b1 = np.zeros(ffn_dim)
        self.W2 = self._init_weights((ffn_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
    
    def _init_weights(self, shape):
        """Initialize weights using Xavier initialization."""
        fan_in = shape[0]
        scale = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * scale
    
    def forward(self, x):
        """
        Forward pass through FFN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # First linear transformation
        hidden = np.matmul(x, self.W1) + self.b1
        
        # Apply activation
        if self.activation == 'relu':
            hidden = np.maximum(0, hidden)
        elif self.activation == 'gelu':
            hidden = self._gelu(hidden)
        
        # Apply dropout
        if self.dropout > 0:
            hidden = self._apply_dropout(hidden, self.dropout)
        
        # Second linear transformation
        output = np.matmul(hidden, self.W2) + self.b2
        
        # Apply dropout
        if self.dropout > 0:
            output = self._apply_dropout(output, self.dropout)
        
        return output
    
    def _gelu(self, x):
        """
        Gaussian Error Linear Unit (GELU) activation.
        
        GELU(x) = x * Φ(x) where Φ(x) is the standard Gaussian CDF.
        Approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
        
        References:
            - Hendrycks & Gimpel (2016): "Gaussian Error Linear Units (GELUs)"
        """
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _apply_dropout(self, x, dropout_rate):
        """Apply dropout during training."""
        mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape)
        return x * mask / (1 - dropout_rate)


class Embedding:
    """
    Token embedding layer.
    
    Maps discrete token IDs to continuous dense vectors.
    """
    
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embedding matrix
        # Shape: (vocab_size, embedding_dim)
        self.weight = self._init_weights((vocab_size, embedding_dim))
    
    def _init_weights(self, shape):
        """Initialize weights with small random values."""
        return np.random.randn(*shape) * 0.01
    
    def forward(self, token_ids):
        """
        Look up embeddings for token IDs.
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
        
        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        # Simple lookup in embedding matrix
        return self.weight[token_ids]


class PositionalEncoding:
    """
    Absolute positional encoding using sinusoidal functions.
    
    Since transformers have no inherent notion of position, we add positional
    encodings to give the model information about token positions.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    References:
        - Vaswani et al. (2017): "Attention is All You Need"
    """
    
    def __init__(self, max_seq_len, embedding_dim):
        """
        Initialize positional encoding.
        
        Args:
            max_seq_len: Maximum sequence length
            embedding_dim: Dimension of embeddings
        """
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        # Pre-compute positional encodings
        self.encoding = self._create_positional_encoding()
    
    def _create_positional_encoding(self):
        """
        Create sinusoidal positional encoding matrix.
        
        Returns:
            Encoding matrix of shape (max_seq_len, embedding_dim)
        """
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * 
                         -(np.log(10000.0) / self.embedding_dim))
        
        encoding = np.zeros((self.max_seq_len, self.embedding_dim))
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)
        
        return encoding
    
    def forward(self, seq_len):
        """
        Get positional encodings for a sequence.
        
        Args:
            seq_len: Length of sequence
        
        Returns:
            Positional encodings of shape (seq_len, embedding_dim)
        """
        return self.encoding[:seq_len, :]


class LearnedPositionalEncoding:
    """
    Learned positional embeddings.
    
    Instead of fixed sinusoidal encodings, this uses learnable embeddings
    for each position. Used in models like GPT.
    """
    
    def __init__(self, max_seq_len, embedding_dim):
        """
        Initialize learned positional encoding.
        
        Args:
            max_seq_len: Maximum sequence length
            embedding_dim: Dimension of embeddings
        """
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        # Initialize position embeddings
        self.position_embeddings = self._init_weights((max_seq_len, embedding_dim))
    
    def _init_weights(self, shape):
        """Initialize weights with small random values."""
        return np.random.randn(*shape) * 0.01
    
    def forward(self, seq_len):
        """
        Get positional embeddings for a sequence.
        
        Args:
            seq_len: Length of sequence
        
        Returns:
            Positional embeddings of shape (seq_len, embedding_dim)
        """
        return self.position_embeddings[:seq_len, :]


class TransformerBlock:
    """
    Transformer decoder block for GPT-style models.
    
    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> Residual
          -> LayerNorm -> FeedForward -> Residual
    
    This is the core building block of decoder-only transformers like GPT.
    """
    
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        """
        Initialize transformer block.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            dropout: Dropout probability
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        
        # Import here to avoid circular dependency
        from litetorch.nn.attention import MultiHeadAttention
        
        # Components
        self.ln1 = LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ln2 = LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout, activation='gelu')
    
    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Self-attention with residual connection
        attn_input = self.ln1.forward(x)
        attn_output = self.attention.forward(attn_input, attn_input, attn_input, mask)
        x = x + attn_output
        
        # Feed-forward with residual connection
        ffn_input = self.ln2.forward(x)
        ffn_output = self.ffn.forward(ffn_input)
        x = x + ffn_output
        
        return x
