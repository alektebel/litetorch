"""
Attention mechanisms for transformers and LLMs.

This module implements the core attention mechanisms used in modern LLMs,
including scaled dot-product attention and multi-head attention.
"""
import numpy as np


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention mechanism.
    
    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    This is the fundamental building block of transformer architectures.
    The scaling by sqrt(d_k) prevents the dot products from growing too large,
    which would push the softmax into regions with extremely small gradients.
    
    References:
        - Vaswani et al. (2017): "Attention is All You Need"
    """
    
    def __init__(self, dropout=0.1):
        """
        Initialize scaled dot-product attention.
        
        Args:
            dropout: Dropout probability for attention weights
        """
        self.dropout = dropout
        self.attention_weights = None  # Store for visualization
    
    def forward(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_k)
            key: Key tensor of shape (batch_size, seq_len_k, d_k)
            value: Value tensor of shape (batch_size, seq_len_v, d_v)
            mask: Optional mask of shape (batch_size, seq_len_q, seq_len_k)
                  or (seq_len_q, seq_len_k). Values should be 0 (attend) or 
                  -inf (don't attend)
        
        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_v)
            attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
        """
        # Get the dimension of key vectors for scaling
        d_k = query.shape[-1]
        
        # Compute attention scores: QK^T / sqrt(d_k)
        # Shape: (batch_size, seq_len_q, seq_len_k)
        scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # If mask is 2D, expand to 3D
            if mask.ndim == 2:
                mask = mask[np.newaxis, :, :]
            scores = scores + mask
        
        # Apply softmax to get attention weights
        # Softmax is applied over the last dimension (seq_len_k)
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply dropout to attention weights
        if self.dropout > 0:
            attention_weights = self._apply_dropout(attention_weights, self.dropout)
        
        # Store attention weights for visualization
        self.attention_weights = attention_weights
        
        # Compute weighted sum of values
        # Shape: (batch_size, seq_len_q, d_v)
        output = np.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def _softmax(self, x, axis=-1):
        """
        Numerically stable softmax implementation.
        
        Args:
            x: Input array
            axis: Axis along which to compute softmax
        
        Returns:
            Softmax output
        """
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _apply_dropout(self, x, dropout_rate):
        """
        Apply dropout during training.
        
        Args:
            x: Input array
            dropout_rate: Dropout probability
        
        Returns:
            Array with dropout applied
        """
        # Handle edge cases
        if dropout_rate >= 1.0:
            # If dropout rate is 1.0, all values are dropped
            return np.zeros_like(x)
        if dropout_rate <= 0.0:
            # No dropout
            return x
        
        # Note: In a real implementation, this should only apply during training
        mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape)
        return x * mask / (1 - dropout_rate)


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    
    Instead of performing a single attention function, multi-head attention
    projects the queries, keys and values h times with different, learned
    linear projections. This allows the model to jointly attend to information
    from different representation subspaces at different positions.
    
    The h attention outputs are concatenated and once again projected,
    resulting in the final values.
    
    References:
        - Vaswani et al. (2017): "Attention is All You Need"
    """
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_dim: Hidden dimension size (d_model)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        # Linear projections for Q, K, V
        # In a real implementation, these would be proper learnable layers
        self.W_q = self._init_weights((hidden_dim, hidden_dim))
        self.W_k = self._init_weights((hidden_dim, hidden_dim))
        self.W_v = self._init_weights((hidden_dim, hidden_dim))
        
        # Output projection
        self.W_o = self._init_weights((hidden_dim, hidden_dim))
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
    
    def _init_weights(self, shape):
        """
        Initialize weights using Xavier/Glorot initialization.
        
        Args:
            shape: Shape of weight matrix
        
        Returns:
            Initialized weight matrix
        """
        # Xavier initialization: scale by sqrt(1 / fan_in)
        fan_in = shape[0]
        scale = np.sqrt(1.0 / fan_in)
        return np.random.randn(*shape) * scale
    
    def forward(self, query, key, value, mask=None):
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, hidden_dim)
            key: Key tensor of shape (batch_size, seq_len_k, hidden_dim)
            value: Value tensor of shape (batch_size, seq_len_v, hidden_dim)
            mask: Optional mask of shape (seq_len_q, seq_len_k)
        
        Returns:
            output: Attention output of shape (batch_size, seq_len_q, hidden_dim)
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]
        
        # Linear projections
        # Shape: (batch_size, seq_len, hidden_dim)
        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)
        
        # Split into multiple heads and reshape
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        Q = self._split_heads(Q, batch_size)
        K = self._split_heads(K, batch_size)
        V = self._split_heads(V, batch_size)
        
        # Compute attention for each head
        # Reshape for attention computation
        # Shape: (batch_size * num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size * self.num_heads, -1, self.head_dim)
        K = K.reshape(batch_size * self.num_heads, -1, self.head_dim)
        V = V.reshape(batch_size * self.num_heads, -1, self.head_dim)
        
        # Apply attention
        attn_output, _ = self.attention.forward(Q, K, V, mask=mask)
        
        # Reshape back
        # Shape: (batch_size, num_heads, seq_len_q, head_dim)
        attn_output = attn_output.reshape(batch_size, self.num_heads, seq_len_q, self.head_dim)
        
        # Concatenate heads
        # Shape: (batch_size, seq_len_q, hidden_dim)
        attn_output = self._combine_heads(attn_output, batch_size)
        
        # Final linear projection
        # Shape: (batch_size, seq_len_q, hidden_dim)
        output = np.matmul(attn_output, self.W_o)
        
        return output
    
    def _split_heads(self, x, batch_size):
        """
        Split hidden_dim into num_heads and head_dim.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, hidden_dim)
            batch_size: Batch size
        
        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        seq_len = x.shape[1]
        # Reshape: (batch_size, seq_len, num_heads, head_dim)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose: (batch_size, num_heads, seq_len, head_dim)
        return x.transpose(0, 2, 1, 3)
    
    def _combine_heads(self, x, batch_size):
        """
        Combine num_heads and head_dim back into hidden_dim.
        
        Args:
            x: Tensor of shape (batch_size, num_heads, seq_len, head_dim)
            batch_size: Batch size
        
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim)
        """
        seq_len = x.shape[2]
        # Transpose: (batch_size, seq_len, num_heads, head_dim)
        x = x.transpose(0, 2, 1, 3)
        # Reshape: (batch_size, seq_len, hidden_dim)
        return x.reshape(batch_size, seq_len, self.hidden_dim)


def create_causal_mask(seq_len):
    """
    Create a causal (autoregressive) attention mask.
    
    The mask ensures that position i can only attend to positions j <= i.
    This is essential for autoregressive models like GPT.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Causal mask of shape (seq_len, seq_len)
        Values are 0.0 for positions that can be attended to,
        and -inf for positions that should be masked out.
    """
    # Create upper triangular matrix with ones
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    # Convert to mask format: 0 for attend, -inf for mask
    mask = mask * -1e9
    return mask


def create_padding_mask(seq_lengths, max_len):
    """
    Create padding mask for variable-length sequences.
    
    Args:
        seq_lengths: Array of actual sequence lengths (batch_size,)
        max_len: Maximum sequence length
    
    Returns:
        Padding mask of shape (batch_size, max_len)
        Values are 0.0 for valid positions and -inf for padding positions.
    """
    batch_size = len(seq_lengths)
    mask = np.zeros((batch_size, max_len))
    
    for i, length in enumerate(seq_lengths):
        if length < max_len:
            mask[i, length:] = -1e9
    
    return mask
