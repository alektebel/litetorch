"""
Complete GPT (Generative Pre-trained Transformer) implementation.

This module provides a full, educational implementation of GPT architecture
for language modeling and text generation.
"""
import numpy as np
from litetorch.nn.attention import create_causal_mask
from litetorch.nn.layers import (
    Embedding, LearnedPositionalEncoding, TransformerBlock, LayerNorm
)


class GPTModel:
    """
    Full GPT (Generative Pre-trained Transformer) implementation.
    
    GPT is an autoregressive language model that generates text one token at a time,
    conditioning on all previously generated tokens using transformer decoder blocks.
    
    Architecture:
        1. Token embeddings + Positional embeddings
        2. Stack of N transformer decoder blocks
        3. Layer normalization
        4. Language modeling head (linear layer to vocab)
    
    Key Features:
        - Causal (autoregressive) self-attention
        - Learned position embeddings
        - Layer normalization before each sub-layer (pre-norm)
        - GELU activation in feed-forward networks
    
    References:
        - Radford et al. (2018): "Improving Language Understanding by Generative Pre-Training"
        - Radford et al. (2019): "Language Models are Unsupervised Multitask Learners" (GPT-2)
    """
    
    def __init__(self, vocab_size=50257, max_seq_len=1024, num_layers=12,
                 num_heads=12, hidden_dim=768, ffn_dim=3072, dropout=0.1):
        """
        Initialize GPT model.
        
        Args:
            vocab_size: Size of vocabulary (default: GPT-2 vocabulary size)
            max_seq_len: Maximum sequence length (context window)
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads per block
            hidden_dim: Hidden dimension size (d_model)
            ffn_dim: Feed-forward network dimension (typically 4 * hidden_dim)
            dropout: Dropout probability
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        
        # Build model components
        self._build_model()
    
    def _build_model(self):
        """Build all model components."""
        # Token embeddings
        self.token_embedding = Embedding(self.vocab_size, self.hidden_dim)
        
        # Position embeddings (learned, like GPT-2)
        self.position_embedding = LearnedPositionalEncoding(
            self.max_seq_len, self.hidden_dim
        )
        
        # Transformer blocks
        self.blocks = []
        for _ in range(self.num_layers):
            block = TransformerBlock(
                self.hidden_dim,
                self.num_heads,
                self.ffn_dim,
                self.dropout
            )
            self.blocks.append(block)
        
        # Final layer normalization
        self.ln_f = LayerNorm(self.hidden_dim)
        
        # Language modeling head (projects to vocabulary)
        # In many implementations, this is tied to token embeddings
        self.lm_head = self._init_weights((self.hidden_dim, self.vocab_size))
    
    def _init_weights(self, shape):
        """Initialize weights using Xavier initialization."""
        fan_in = shape[0]
        scale = np.sqrt(1.0 / fan_in)
        return np.random.randn(*shape) * scale
    
    def forward(self, input_ids, return_hidden_states=False):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            return_hidden_states: If True, return intermediate hidden states
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden_states: (Optional) List of hidden states from each layer
        """
        batch_size, seq_len = input_ids.shape
        
        # Check sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}"
            )
        
        # Get token embeddings
        # Shape: (batch_size, seq_len, hidden_dim)
        token_embeds = self.token_embedding.forward(input_ids)
        
        # Get position embeddings
        # Shape: (seq_len, hidden_dim)
        pos_embeds = self.position_embedding.forward(seq_len)
        
        # Add token and position embeddings
        # Broadcasting: (batch_size, seq_len, hidden_dim) + (seq_len, hidden_dim)
        hidden_states = token_embeds + pos_embeds
        
        # Create causal mask for autoregressive attention
        causal_mask = create_causal_mask(seq_len)
        
        # Store hidden states if requested
        all_hidden_states = [hidden_states] if return_hidden_states else None
        
        # Pass through transformer blocks
        for block in self.blocks:
            hidden_states = block.forward(hidden_states, mask=causal_mask)
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Final layer normalization
        hidden_states = self.ln_f.forward(hidden_states)
        
        # Language modeling head
        # Shape: (batch_size, seq_len, vocab_size)
        logits = np.matmul(hidden_states, self.lm_head)
        
        if return_hidden_states:
            return logits, all_hidden_states
        return logits
    
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, 
                 top_k=None, top_p=None):
        """
        Generate text autoregressively.
        
        Args:
            prompt_ids: Initial token IDs to condition on, shape (batch_size, prompt_len)
                       or (prompt_len,) for single sequence
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
                        temperature < 1.0 makes distribution sharper (more confident)
                        temperature > 1.0 makes distribution softer (more diverse)
            top_k: If set, only sample from top k most likely tokens
            top_p: If set, sample from tokens whose cumulative probability >= top_p (nucleus sampling)
        
        Returns:
            Generated token IDs of shape (batch_size, prompt_len + max_new_tokens)
        """
        # Handle single sequence input
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids[np.newaxis, :]
        
        batch_size = prompt_ids.shape[0]
        generated = prompt_ids.copy()
        
        for _ in range(max_new_tokens):
            # Get current sequence length
            curr_len = generated.shape[1]
            
            # Truncate if needed (use last max_seq_len tokens)
            if curr_len > self.max_seq_len:
                input_ids = generated[:, -self.max_seq_len:]
            else:
                input_ids = generated
            
            # Forward pass
            logits = self.forward(input_ids)
            
            # Get logits for last position (next token prediction)
            # Shape: (batch_size, vocab_size)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Convert logits to probabilities
            probs = self._softmax(next_token_logits, axis=-1)
            
            # Sample next token
            next_token = self._sample_from_probs(probs)
            
            # Append to generated sequence
            generated = np.concatenate([generated, next_token[:, np.newaxis]], axis=1)
        
        return generated
    
    def _top_k_filtering(self, logits, top_k):
        """
        Filter logits to only keep top k tokens.
        
        Args:
            logits: Logits of shape (batch_size, vocab_size)
            top_k: Number of top tokens to keep
        
        Returns:
            Filtered logits
        """
        # Get top k values and indices
        top_k = min(top_k, logits.shape[-1])
        # Use partition to find the k-th largest value
        # partition returns array partitioned at k-th position
        kth_values = np.partition(logits, -top_k, axis=-1)[:, -top_k]
        # Expand dimensions for broadcasting
        kth_values = kth_values[:, np.newaxis]
        # Mask out values below the k-th largest
        logits = np.where(logits < kth_values, -np.inf, logits)
        return logits
    
    def _top_p_filtering(self, logits, top_p):
        """
        Filter logits using nucleus (top-p) sampling.
        
        Args:
            logits: Logits of shape (batch_size, vocab_size)
            top_p: Cumulative probability threshold
        
        Returns:
            Filtered logits
        """
        # Sort logits in descending order
        sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
        
        # Compute cumulative probabilities
        sorted_probs = self._softmax(sorted_logits, axis=-1)
        cumulative_probs = np.cumsum(sorted_probs, axis=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].copy()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original indexing
        for i in range(logits.shape[0]):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = -np.inf
        
        return logits
    
    def _softmax(self, x, axis=-1):
        """Numerically stable softmax."""
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _sample_from_probs(self, probs):
        """
        Sample token indices from probability distribution.
        
        Args:
            probs: Probabilities of shape (batch_size, vocab_size)
        
        Returns:
            Sampled token indices of shape (batch_size,)
        """
        batch_size = probs.shape[0]
        sampled = np.zeros(batch_size, dtype=np.int32)
        
        for i in range(batch_size):
            sampled[i] = np.random.choice(self.vocab_size, p=probs[i])
        
        return sampled
    
    def compute_loss(self, input_ids, target_ids):
        """
        Compute cross-entropy loss for language modeling.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            target_ids: Target token IDs of shape (batch_size, seq_len)
                       (typically input_ids shifted by 1)
        
        Returns:
            loss: Scalar cross-entropy loss
        """
        # Forward pass
        logits = self.forward(input_ids)
        
        # Compute cross-entropy loss
        loss = self._cross_entropy_loss(logits, target_ids)
        
        return loss
    
    def _cross_entropy_loss(self, logits, targets):
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Predicted logits of shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs of shape (batch_size, seq_len)
        
        Returns:
            loss: Scalar loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for easier computation
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Compute softmax probabilities
        probs = self._softmax(logits_flat, axis=-1)
        
        # Get probabilities of target tokens
        target_probs = probs[np.arange(len(targets_flat)), targets_flat]
        
        # Compute negative log likelihood
        loss = -np.mean(np.log(target_probs + 1e-10))
        
        return loss
    
    def get_num_parameters(self):
        """
        Get total number of parameters in the model.
        
        Returns:
            num_params: Total number of parameters
        """
        num_params = 0
        
        # Token embeddings
        num_params += self.token_embedding.weight.size
        
        # Position embeddings
        num_params += self.position_embedding.position_embeddings.size
        
        # Transformer blocks
        for block in self.blocks:
            # Attention (Q, K, V, O projections)
            num_params += block.attention.W_q.size
            num_params += block.attention.W_k.size
            num_params += block.attention.W_v.size
            num_params += block.attention.W_o.size
            
            # FFN (2 linear layers)
            num_params += block.ffn.W1.size + block.ffn.b1.size
            num_params += block.ffn.W2.size + block.ffn.b2.size
            
            # Layer norms (2 per block)
            num_params += block.ln1.gamma.size + block.ln1.beta.size
            num_params += block.ln2.gamma.size + block.ln2.beta.size
        
        # Final layer norm
        num_params += self.ln_f.gamma.size + self.ln_f.beta.size
        
        # LM head
        num_params += self.lm_head.size
        
        return num_params
