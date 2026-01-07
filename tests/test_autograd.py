"""
Comprehensive test suite for automatic differentiation (autograd).

This test suite covers:
- Basic backward pass on simple operations
- Gradient computation for arithmetic operations (add, sub, mul, div)
- Gradient computation for matrix operations (matmul)
- Gradient computation for activation functions (ReLU, Sigmoid, Tanh)
- Chain rule application in computational graphs
- Gradient accumulation
- Edge cases (zero gradients, disconnected graphs, etc.)

Note: These tests define the expected API for the autograd engine.
Some tests are marked with @unittest.skip until the autograd engine is implemented.
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litetorch.tensor import Tensor


class TestAutogradBasicBackward(unittest.TestCase):
    """Test basic backward pass functionality."""
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_simple_backward_scalar(self):
        """Test backward pass on simple scalar operation."""
        # Create a tensor with requires_grad=True
        x = Tensor(shape=(1,), requires_grad=True)
        x.data = [3.0]
        
        # Simple operation: y = 2 * x
        y = x * 2.0
        
        # Backward pass
        y.backward()
        
        # Check gradient
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.data, [2.0])
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_backward_addition(self):
        """Test backward pass for addition."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        y = Tensor(shape=(2,), requires_grad=True)
        y.data = [3.0, 4.0]
        
        # z = x + y
        z = x + y
        
        # Loss: sum of all elements
        loss = z.sum()
        loss.backward()
        
        # Gradient of sum with respect to inputs is 1
        self.assertEqual(x.grad.data, [1.0, 1.0])
        self.assertEqual(y.grad.data, [1.0, 1.0])
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_backward_subtraction(self):
        """Test backward pass for subtraction."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [5.0, 6.0]
        
        y = Tensor(shape=(2,), requires_grad=True)
        y.data = [2.0, 1.0]
        
        # z = x - y
        z = x - y
        
        loss = z.sum()
        loss.backward()
        
        # d(x-y)/dx = 1, d(x-y)/dy = -1
        self.assertEqual(x.grad.data, [1.0, 1.0])
        self.assertEqual(y.grad.data, [-1.0, -1.0])
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_backward_multiplication(self):
        """Test backward pass for element-wise multiplication."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [2.0, 3.0]
        
        y = Tensor(shape=(2,), requires_grad=True)
        y.data = [4.0, 5.0]
        
        # z = x * y
        z = x * y
        
        loss = z.sum()
        loss.backward()
        
        # d(x*y)/dx = y, d(x*y)/dy = x
        self.assertEqual(x.grad.data, [4.0, 5.0])
        self.assertEqual(y.grad.data, [2.0, 3.0])
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_backward_division(self):
        """Test backward pass for division."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [8.0, 12.0]
        
        y = Tensor(shape=(2,), requires_grad=True)
        y.data = [2.0, 3.0]
        
        # z = x / y
        z = x / y
        
        loss = z.sum()
        loss.backward()
        
        # d(x/y)/dx = 1/y, d(x/y)/dy = -x/y^2
        self.assertAlmostEqual(x.grad.data[0], 0.5, places=5)
        self.assertAlmostEqual(x.grad.data[1], 1/3, places=5)
        self.assertAlmostEqual(y.grad.data[0], -2.0, places=5)
        self.assertAlmostEqual(y.grad.data[1], -4/3, places=5)
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_backward_power(self):
        """Test backward pass for power operation."""
        x = Tensor(shape=(3,), requires_grad=True)
        x.data = [2.0, 3.0, 4.0]
        
        # y = x^2
        y = x ** 2
        
        loss = y.sum()
        loss.backward()
        
        # d(x^2)/dx = 2*x
        self.assertEqual(x.grad.data, [4.0, 6.0, 8.0])


class TestAutogradChainRule(unittest.TestCase):
    """Test chain rule in computational graphs."""
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_chain_rule_simple(self):
        """Test chain rule with simple operations."""
        x = Tensor(shape=(1,), requires_grad=True)
        x.data = [2.0]
        
        # y = x * 2
        y = x * 2.0
        
        # z = y + 3
        z = y + 3.0
        
        # w = z * 4
        w = z * 4.0
        
        w.backward()
        
        # dw/dx = dw/dz * dz/dy * dy/dx = 4 * 1 * 2 = 8
        self.assertEqual(x.grad.data, [8.0])
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_chain_rule_complex(self):
        """Test chain rule with multiple operations."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        # y = x^2
        y = x ** 2
        
        # z = y * 3
        z = y * 3.0
        
        # w = z + 1
        w = z + 1.0
        
        loss = w.sum()
        loss.backward()
        
        # dL/dx = dL/dw * dw/dz * dz/dy * dy/dx
        # = 1 * 1 * 3 * 2*x = 6*x
        self.assertEqual(x.grad.data, [6.0, 12.0])
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_chain_rule_branching(self):
        """Test chain rule with branching computational graph."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [2.0, 3.0]
        
        # Create two branches
        y1 = x * 2.0
        y2 = x + 1.0
        
        # Merge branches
        z = y1 + y2
        
        loss = z.sum()
        loss.backward()
        
        # Gradient should accumulate from both branches
        # dL/dx = dy1/dx + dy2/dx = 2 + 1 = 3
        self.assertEqual(x.grad.data, [3.0, 3.0])


class TestAutogradMatrixOperations(unittest.TestCase):
    """Test gradient computation for matrix operations."""
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_matmul_gradient(self):
        """Test gradient computation for matrix multiplication."""
        # A: (2, 3), B: (3, 2)
        A = Tensor(shape=(2, 3), requires_grad=True)
        A.data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        
        B = Tensor(shape=(3, 2), requires_grad=True)
        B.data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        
        # C = A @ B, shape: (2, 2)
        C = A @ B
        
        loss = C.sum()
        loss.backward()
        
        # Gradient shapes should match input shapes
        self.assertEqual(A.grad.shape, (2, 3))
        self.assertEqual(B.grad.shape, (3, 2))
        self.assertIsNotNone(A.grad.data)
        self.assertIsNotNone(B.grad.data)
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_matmul_chain(self):
        """Test gradient through chain of matrix multiplications."""
        # A: (2, 3), B: (3, 2), C: (2, 1)
        A = Tensor(shape=(2, 3), requires_grad=True)
        A.data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        
        B = Tensor(shape=(3, 2), requires_grad=True)
        B.data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        
        C = Tensor(shape=(2, 1), requires_grad=True)
        C.data = [1.0, 2.0]
        
        # D = (A @ B) @ C
        temp = A @ B
        D = temp @ C
        
        loss = D.sum()
        loss.backward()
        
        # All gradients should be computed
        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(B.grad)
        self.assertIsNotNone(C.grad)


class TestAutogradActivationFunctions(unittest.TestCase):
    """Test gradient computation for activation functions."""
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_relu_gradient(self):
        """Test ReLU gradient computation."""
        x = Tensor(shape=(4,), requires_grad=True)
        x.data = [-2.0, -1.0, 1.0, 2.0]
        
        # y = ReLU(x) = max(0, x)
        y = x.relu()
        
        loss = y.sum()
        loss.backward()
        
        # ReLU gradient: 1 if x > 0, else 0
        self.assertEqual(x.grad.data, [0.0, 0.0, 1.0, 1.0])
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_sigmoid_gradient(self):
        """Test Sigmoid gradient computation."""
        x = Tensor(shape=(3,), requires_grad=True)
        x.data = [0.0, 1.0, -1.0]
        
        # y = sigmoid(x) = 1 / (1 + exp(-x))
        y = x.sigmoid()
        
        loss = y.sum()
        loss.backward()
        
        # Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
        self.assertIsNotNone(x.grad)
        self.assertEqual(len(x.grad.data), 3)
        
        # At x=0, sigmoid=0.5, gradient=0.25
        self.assertAlmostEqual(x.grad.data[0], 0.25, places=5)
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_tanh_gradient(self):
        """Test Tanh gradient computation."""
        x = Tensor(shape=(3,), requires_grad=True)
        x.data = [0.0, 1.0, -1.0]
        
        # y = tanh(x)
        y = x.tanh()
        
        loss = y.sum()
        loss.backward()
        
        # Tanh gradient: 1 - tanh(x)^2
        self.assertIsNotNone(x.grad)
        self.assertEqual(len(x.grad.data), 3)
        
        # At x=0, tanh=0, gradient=1
        self.assertAlmostEqual(x.grad.data[0], 1.0, places=5)
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_softmax_gradient(self):
        """Test Softmax gradient computation."""
        x = Tensor(shape=(3,), requires_grad=True)
        x.data = [1.0, 2.0, 3.0]
        
        # y = softmax(x)
        y = x.softmax(dim=0)
        
        loss = y.sum()
        loss.backward()
        
        # Softmax gradient is more complex
        self.assertIsNotNone(x.grad)
        self.assertEqual(len(x.grad.data), 3)


class TestAutogradGradientAccumulation(unittest.TestCase):
    """Test gradient accumulation."""
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_gradient_accumulation_simple(self):
        """Test that gradients accumulate correctly."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        # First forward and backward
        y1 = x * 2.0
        loss1 = y1.sum()
        loss1.backward()
        
        grad_after_first = x.grad.data.copy()
        
        # Second forward and backward (without zero_grad)
        y2 = x * 3.0
        loss2 = y2.sum()
        loss2.backward()
        
        # Gradients should accumulate
        expected = [grad_after_first[0] + 3.0, grad_after_first[1] + 3.0]
        self.assertEqual(x.grad.data, expected)
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_zero_grad(self):
        """Test that zero_grad resets gradients."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        # First backward
        y = x * 2.0
        loss = y.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        
        # Zero gradients
        x.zero_grad()
        
        # Gradients should be zero
        self.assertEqual(x.grad.data, [0.0, 0.0])


class TestAutogradEdgeCases(unittest.TestCase):
    """Test edge cases in automatic differentiation."""
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_no_grad_context(self):
        """Test computation without gradient tracking."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        # Computation without gradient tracking
        with no_grad():
            y = x * 2.0
            z = y + 3.0
        
        # Should not be able to call backward
        with self.assertRaises(RuntimeError):
            z.backward()
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_detach(self):
        """Test detaching tensor from computational graph."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        y = x * 2.0
        z = y.detach()  # Detach from graph
        w = z * 3.0
        
        loss = w.sum()
        loss.backward()
        
        # x should have no gradient since z was detached
        self.assertIsNone(x.grad)
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_backward_on_non_scalar(self):
        """Test that backward on non-scalar requires gradient argument."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        y = x * 2.0
        
        # Should raise error without gradient argument
        with self.assertRaises(RuntimeError):
            y.backward()
        
        # Should work with gradient
        grad = Tensor(shape=(2,))
        grad.data = [1.0, 1.0]
        y.backward(gradient=grad)
        
        self.assertIsNotNone(x.grad)
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_in_place_operations(self):
        """Test that in-place operations work correctly."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        y = x * 2.0
        y += 1.0  # In-place operation
        
        loss = y.sum()
        loss.backward()
        
        # Gradient should still be computed correctly
        self.assertEqual(x.grad.data, [2.0, 2.0])
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_retain_graph(self):
        """Test retaining computational graph for multiple backward passes."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        y = x * 2.0
        loss = y.sum()
        
        # First backward with retain_graph=True
        loss.backward(retain_graph=True)
        grad_first = x.grad.data.copy()
        
        # Second backward should work
        loss.backward()
        
        # Gradients should accumulate
        expected = [g * 2 for g in grad_first]
        self.assertEqual(x.grad.data, expected)


class TestAutogradComputationalGraph(unittest.TestCase):
    """Test complex computational graphs."""
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_diamond_graph(self):
        """Test computational graph with diamond structure."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        # Split into two paths
        y1 = x * 2.0
        y2 = x * 3.0
        
        # Merge paths
        z = y1 + y2
        
        loss = z.sum()
        loss.backward()
        
        # Gradient should be sum of both paths: 2 + 3 = 5
        self.assertEqual(x.grad.data, [5.0, 5.0])
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_multiple_outputs(self):
        """Test tensor used in multiple operations."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        # Use x in multiple operations
        y = x * 2.0
        z = x + 3.0
        w = x ** 2
        
        # Combine all
        loss = y.sum() + z.sum() + w.sum()
        loss.backward()
        
        # Gradient should accumulate from all paths
        # dy/dx = 2, dz/dx = 1, dw/dx = 2*x
        # Total: 2 + 1 + 2*x = 3 + 2*x
        expected = [3.0 + 2.0*1.0, 3.0 + 2.0*2.0]
        self.assertEqual(x.grad.data, expected)
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_nested_operations(self):
        """Test deeply nested operations."""
        x = Tensor(shape=(2,), requires_grad=True)
        x.data = [1.0, 2.0]
        
        # Deep nesting: ((x * 2 + 1) * 3 + 2) * 4
        y = x * 2.0
        y = y + 1.0
        y = y * 3.0
        y = y + 2.0
        y = y * 4.0
        
        loss = y.sum()
        loss.backward()
        
        # Chain rule through all operations
        # Gradient = 4 * 3 * 2 = 24
        self.assertEqual(x.grad.data, [24.0, 24.0])


class TestAutogradNeuralNetwork(unittest.TestCase):
    """Test autograd in neural network context."""
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_simple_linear_layer(self):
        """Test gradient computation through linear layer."""
        # Input: (batch_size=2, features=3)
        x = Tensor(shape=(2, 3), requires_grad=True)
        x.data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        
        # Weights: (features=3, output=2)
        W = Tensor(shape=(3, 2), requires_grad=True)
        W.data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        # Bias: (output=2)
        b = Tensor(shape=(2,), requires_grad=True)
        b.data = [0.1, 0.2]
        
        # Forward: y = x @ W + b
        y = (x @ W) + b
        
        loss = y.sum()
        loss.backward()
        
        # All parameters should have gradients
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(W.grad)
        self.assertIsNotNone(b.grad)
    
    @unittest.skip("Autograd engine not yet implemented")
    def test_two_layer_network(self):
        """Test gradient computation through two-layer network."""
        # Input
        x = Tensor(shape=(1, 3), requires_grad=True)
        x.data = [1.0, 2.0, 3.0]
        
        # Layer 1: 3 -> 4
        W1 = Tensor(shape=(3, 4), requires_grad=True)
        W1.data = [0.1] * 12
        b1 = Tensor(shape=(4,), requires_grad=True)
        b1.data = [0.1, 0.2, 0.3, 0.4]
        
        # Layer 2: 4 -> 2
        W2 = Tensor(shape=(4, 2), requires_grad=True)
        W2.data = [0.1] * 8
        b2 = Tensor(shape=(2,), requires_grad=True)
        b2.data = [0.1, 0.2]
        
        # Forward pass
        h = (x @ W1) + b1
        h = h.relu()
        y = (h @ W2) + b2
        
        loss = y.sum()
        loss.backward()
        
        # All parameters should have gradients
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(W1.grad)
        self.assertIsNotNone(b1.grad)
        self.assertIsNotNone(W2.grad)
        self.assertIsNotNone(b2.grad)


if __name__ == '__main__':
    unittest.main()
