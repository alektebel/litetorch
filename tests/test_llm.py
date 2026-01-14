"""
Comprehensive tests for LLM implementation.

Tests attention mechanisms, transformer blocks, and complete GPT model.
"""
import unittest
import numpy as np


class TestAttentionMechanism(unittest.TestCase):
    """Test attention mechanisms."""
    
    def test_scaled_dot_product_attention_shape(self):
        """Test that scaled dot-product attention produces correct output shape."""
        from litetorch.nn.attention import ScaledDotProductAttention
        
        attention = ScaledDotProductAttention(dropout=0.0)
        
        batch_size, seq_len, d_k = 2, 10, 64
        query = np.random.randn(batch_size, seq_len, d_k)
        key = np.random.randn(batch_size, seq_len, d_k)
        value = np.random.randn(batch_size, seq_len, d_k)
        
        output, attn_weights = attention.forward(query, key, value)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, d_k))
        # Check attention weights shape
        self.assertEqual(attn_weights.shape, (batch_size, seq_len, seq_len))
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 for each query position."""
        from litetorch.nn.attention import ScaledDotProductAttention
        
        attention = ScaledDotProductAttention(dropout=0.0)
        
        batch_size, seq_len, d_k = 2, 5, 32
        query = np.random.randn(batch_size, seq_len, d_k)
        key = np.random.randn(batch_size, seq_len, d_k)
        value = np.random.randn(batch_size, seq_len, d_k)
        
        _, attn_weights = attention.forward(query, key, value)
        
        # Attention weights should sum to 1 along the key dimension
        weights_sum = np.sum(attn_weights, axis=-1)
        np.testing.assert_array_almost_equal(weights_sum, np.ones((batch_size, seq_len)))
    
    def test_causal_mask(self):
        """Test that causal mask prevents attending to future positions."""
        from litetorch.nn.attention import ScaledDotProductAttention, create_causal_mask
        
        attention = ScaledDotProductAttention(dropout=0.0)
        
        batch_size, seq_len, d_k = 1, 4, 16
        query = np.random.randn(batch_size, seq_len, d_k)
        key = np.random.randn(batch_size, seq_len, d_k)
        value = np.random.randn(batch_size, seq_len, d_k)
        
        # Create causal mask
        mask = create_causal_mask(seq_len)
        
        _, attn_weights = attention.forward(query, key, value, mask=mask)
        
        # Check that upper triangular part (future positions) has near-zero weights
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                self.assertLess(attn_weights[0, i, j], 1e-6)
    
    def test_multi_head_attention_shape(self):
        """Test multi-head attention output shape."""
        from litetorch.nn.attention import MultiHeadAttention
        
        hidden_dim, num_heads = 64, 4
        mha = MultiHeadAttention(hidden_dim, num_heads, dropout=0.0)
        
        batch_size, seq_len = 2, 8
        x = np.random.randn(batch_size, seq_len, hidden_dim)
        
        output = mha.forward(x, x, x)
        
        # Output should have same shape as input
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_dim))
    
    def test_multi_head_attention_head_dimension(self):
        """Test that hidden_dim is correctly split across heads."""
        from litetorch.nn.attention import MultiHeadAttention
        
        hidden_dim, num_heads = 96, 8
        mha = MultiHeadAttention(hidden_dim, num_heads, dropout=0.0)
        
        expected_head_dim = hidden_dim // num_heads
        self.assertEqual(mha.head_dim, expected_head_dim)
    
    def test_create_causal_mask_shape(self):
        """Test causal mask has correct shape."""
        from litetorch.nn.attention import create_causal_mask
        
        seq_len = 10
        mask = create_causal_mask(seq_len)
        
        self.assertEqual(mask.shape, (seq_len, seq_len))
    
    def test_create_causal_mask_pattern(self):
        """Test causal mask has correct pattern."""
        from litetorch.nn.attention import create_causal_mask
        
        seq_len = 5
        mask = create_causal_mask(seq_len)
        
        # Lower triangular (including diagonal) should be 0
        for i in range(seq_len):
            for j in range(i + 1):
                self.assertAlmostEqual(mask[i, j], 0.0)
        
        # Upper triangular should be -inf or very negative
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                self.assertLess(mask[i, j], -1e8)


class TestNeuralNetworkLayers(unittest.TestCase):
    """Test neural network layers."""
    
    def test_layer_norm_shape(self):
        """Test layer normalization preserves shape."""
        from litetorch.nn.layers import LayerNorm
        
        hidden_dim = 64
        ln = LayerNorm(hidden_dim)
        
        batch_size, seq_len = 2, 10
        x = np.random.randn(batch_size, seq_len, hidden_dim)
        
        output = ln.forward(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_layer_norm_mean_variance(self):
        """Test that layer norm produces zero mean and unit variance."""
        from litetorch.nn.layers import LayerNorm
        
        hidden_dim = 64
        ln = LayerNorm(hidden_dim)
        
        # Set gamma=1, beta=0 for pure normalization
        ln.gamma = np.ones(hidden_dim)
        ln.beta = np.zeros(hidden_dim)
        
        x = np.random.randn(2, 10, hidden_dim)
        output = ln.forward(x)
        
        # Check mean and variance across feature dimension
        mean = np.mean(output, axis=-1)
        var = np.var(output, axis=-1)
        
        np.testing.assert_array_almost_equal(mean, np.zeros_like(mean), decimal=5)
        np.testing.assert_array_almost_equal(var, np.ones_like(var), decimal=4)
    
    def test_feed_forward_shape(self):
        """Test feed-forward network output shape."""
        from litetorch.nn.layers import FeedForward
        
        hidden_dim, ffn_dim = 64, 256
        ffn = FeedForward(hidden_dim, ffn_dim, dropout=0.0)
        
        batch_size, seq_len = 2, 8
        x = np.random.randn(batch_size, seq_len, hidden_dim)
        
        output = ffn.forward(x)
        
        # Output should have same shape as input
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_dim))
    
    def test_embedding_shape(self):
        """Test embedding layer output shape."""
        from litetorch.nn.layers import Embedding
        
        vocab_size, embedding_dim = 1000, 128
        emb = Embedding(vocab_size, embedding_dim)
        
        batch_size, seq_len = 2, 10
        token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        
        output = emb.forward(token_ids)
        
        self.assertEqual(output.shape, (batch_size, seq_len, embedding_dim))
    
    def test_positional_encoding_shape(self):
        """Test positional encoding shape."""
        from litetorch.nn.layers import PositionalEncoding
        
        max_seq_len, embedding_dim = 512, 128
        pe = PositionalEncoding(max_seq_len, embedding_dim)
        
        seq_len = 50
        output = pe.forward(seq_len)
        
        self.assertEqual(output.shape, (seq_len, embedding_dim))
    
    def test_learned_positional_encoding_shape(self):
        """Test learned positional encoding shape."""
        from litetorch.nn.layers import LearnedPositionalEncoding
        
        max_seq_len, embedding_dim = 512, 128
        lpe = LearnedPositionalEncoding(max_seq_len, embedding_dim)
        
        seq_len = 50
        output = lpe.forward(seq_len)
        
        self.assertEqual(output.shape, (seq_len, embedding_dim))
    
    def test_transformer_block_shape(self):
        """Test transformer block output shape."""
        from litetorch.nn.layers import TransformerBlock
        
        hidden_dim, num_heads, ffn_dim = 64, 4, 256
        block = TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout=0.0)
        
        batch_size, seq_len = 2, 8
        x = np.random.randn(batch_size, seq_len, hidden_dim)
        
        output = block.forward(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_dim))


class TestGPTModel(unittest.TestCase):
    """Test complete GPT model."""
    
    def test_gpt_initialization(self):
        """Test GPT model initializes correctly."""
        from litetorch.nn.gpt import GPTModel
        
        vocab_size = 5000
        max_seq_len = 128
        num_layers = 4
        num_heads = 4
        hidden_dim = 256
        
        model = GPTModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim
        )
        
        self.assertEqual(model.vocab_size, vocab_size)
        self.assertEqual(model.max_seq_len, max_seq_len)
        self.assertEqual(model.num_layers, num_layers)
        self.assertEqual(model.num_heads, num_heads)
        self.assertEqual(model.hidden_dim, hidden_dim)
    
    def test_gpt_forward_pass_shape(self):
        """Test GPT forward pass produces correct output shape."""
        from litetorch.nn.gpt import GPTModel
        
        vocab_size = 1000
        model = GPTModel(
            vocab_size=vocab_size,
            max_seq_len=128,
            num_layers=2,
            num_heads=4,
            hidden_dim=64
        )
        
        batch_size, seq_len = 2, 10
        input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        
        logits = model.forward(input_ids)
        
        # Output should be (batch_size, seq_len, vocab_size)
        self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
    
    def test_gpt_generation(self):
        """Test GPT text generation."""
        from litetorch.nn.gpt import GPTModel
        
        vocab_size = 500
        model = GPTModel(
            vocab_size=vocab_size,
            max_seq_len=128,
            num_layers=2,
            num_heads=2,
            hidden_dim=32
        )
        
        # Single sequence prompt
        prompt = np.array([1, 2, 3, 4])
        max_new_tokens = 10
        
        generated = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)
        
        # Check output length
        expected_len = len(prompt) + max_new_tokens
        self.assertEqual(generated.shape[1], expected_len)
        
        # Check that prompt is preserved
        np.testing.assert_array_equal(generated[0, :len(prompt)], prompt)
    
    def test_gpt_generation_with_temperature(self):
        """Test GPT generation with different temperatures."""
        from litetorch.nn.gpt import GPTModel
        
        vocab_size = 500
        model = GPTModel(
            vocab_size=vocab_size,
            max_seq_len=128,
            num_layers=2,
            num_heads=2,
            hidden_dim=32
        )
        
        prompt = np.array([1, 2, 3])
        
        # Generate with low temperature (more deterministic)
        gen_low_temp = model.generate(prompt, max_new_tokens=5, temperature=0.5)
        
        # Generate with high temperature (more random)
        gen_high_temp = model.generate(prompt, max_new_tokens=5, temperature=2.0)
        
        # Both should have correct length
        self.assertEqual(gen_low_temp.shape[1], len(prompt) + 5)
        self.assertEqual(gen_high_temp.shape[1], len(prompt) + 5)
    
    def test_gpt_compute_loss(self):
        """Test GPT loss computation."""
        from litetorch.nn.gpt import GPTModel
        
        vocab_size = 500
        model = GPTModel(
            vocab_size=vocab_size,
            max_seq_len=128,
            num_layers=2,
            num_heads=2,
            hidden_dim=32
        )
        
        batch_size, seq_len = 2, 10
        input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        target_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        
        loss = model.compute_loss(input_ids, target_ids)
        
        # Loss should be a scalar
        self.assertIsInstance(loss, (float, np.floating))
        
        # Loss should be positive
        self.assertGreater(loss, 0)
    
    def test_gpt_parameter_count(self):
        """Test GPT parameter counting."""
        from litetorch.nn.gpt import GPTModel
        
        model = GPTModel(
            vocab_size=1000,
            max_seq_len=128,
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            ffn_dim=256
        )
        
        num_params = model.get_num_parameters()
        
        # Should have positive number of parameters
        self.assertGreater(num_params, 0)
        
        # Rough sanity check: should have at least vocab_size * hidden_dim params
        # (just from token embeddings)
        self.assertGreater(num_params, 1000 * 64)
    
    def test_gpt_batch_generation(self):
        """Test GPT generation with batched prompts."""
        from litetorch.nn.gpt import GPTModel
        
        vocab_size = 500
        model = GPTModel(
            vocab_size=vocab_size,
            max_seq_len=128,
            num_layers=2,
            num_heads=2,
            hidden_dim=32
        )
        
        # Batch of prompts
        batch_size = 3
        prompt_len = 5
        prompts = np.random.randint(0, vocab_size, size=(batch_size, prompt_len))
        
        generated = model.generate(prompts, max_new_tokens=8)
        
        # Check output shape
        self.assertEqual(generated.shape, (batch_size, prompt_len + 8))
    
    def test_gpt_max_sequence_length(self):
        """Test that GPT respects maximum sequence length."""
        from litetorch.nn.gpt import GPTModel
        
        max_seq_len = 64
        model = GPTModel(
            vocab_size=500,
            max_seq_len=max_seq_len,
            num_layers=2,
            num_heads=2,
            hidden_dim=32
        )
        
        # Try input longer than max_seq_len
        long_input = np.random.randint(0, 500, size=(1, max_seq_len + 10))
        
        # Should raise error
        with self.assertRaises(ValueError):
            model.forward(long_input)


class TestLLMEndToEnd(unittest.TestCase):
    """End-to-end tests for LLM training and inference."""
    
    def test_simple_training_loop(self):
        """Test a simple training step."""
        from litetorch.nn.gpt import GPTModel
        
        # Small model for testing
        model = GPTModel(
            vocab_size=100,
            max_seq_len=32,
            num_layers=2,
            num_heads=2,
            hidden_dim=32,
            ffn_dim=64
        )
        
        # Create simple training data
        batch_size, seq_len = 2, 8
        input_ids = np.random.randint(0, 100, size=(batch_size, seq_len))
        # Targets are typically input_ids shifted by 1
        target_ids = np.roll(input_ids, -1, axis=1)
        
        # Compute loss
        loss = model.compute_loss(input_ids, target_ids)
        
        # Loss should be computed successfully
        self.assertIsInstance(loss, (float, np.floating))
        self.assertGreater(loss, 0)
    
    def test_generation_consistency(self):
        """Test that generation with same seed produces consistent results."""
        from litetorch.nn.gpt import GPTModel
        
        model = GPTModel(
            vocab_size=100,
            max_seq_len=32,
            num_layers=1,
            num_heads=2,
            hidden_dim=16
        )
        
        prompt = np.array([1, 2, 3])
        
        # Generate twice with same seed
        np.random.seed(42)
        gen1 = model.generate(prompt, max_new_tokens=5, temperature=1.0)
        
        np.random.seed(42)
        gen2 = model.generate(prompt, max_new_tokens=5, temperature=1.0)
        
        # Should produce same results
        np.testing.assert_array_equal(gen1, gen2)


if __name__ == '__main__':
    unittest.main()
