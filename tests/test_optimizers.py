"""
Test suite for optimization algorithms.
"""
import unittest


class TestSGD(unittest.TestCase):
    """Test Stochastic Gradient Descent optimizer."""
    
    def test_sgd_step(self):
        """Test SGD parameter update."""
        # TODO: Implement after SGD is created
        pass
    
    def test_sgd_with_momentum(self):
        """Test SGD with momentum parameter update."""
        # TODO: Implement after SGD with momentum is created
        pass
    
    def test_sgd_learning_rate(self):
        """Test SGD with different learning rates."""
        # TODO: Implement after SGD is created
        pass


class TestAdam(unittest.TestCase):
    """Test Adam optimizer."""
    
    def test_adam_step(self):
        """Test Adam parameter update."""
        # TODO: Implement after Adam is created
        pass
    
    def test_adam_beta_parameters(self):
        """Test Adam with different beta parameters."""
        # TODO: Implement after Adam is created
        pass
    
    def test_adam_epsilon(self):
        """Test Adam epsilon parameter."""
        # TODO: Implement after Adam is created
        pass


class TestRMSprop(unittest.TestCase):
    """Test RMSprop optimizer."""
    
    def test_rmsprop_step(self):
        """Test RMSprop parameter update."""
        # TODO: Implement after RMSprop is created
        pass
    
    def test_rmsprop_alpha(self):
        """Test RMSprop with different alpha values."""
        # TODO: Implement after RMSprop is created
        pass


class TestAdaGrad(unittest.TestCase):
    """Test AdaGrad optimizer."""
    
    def test_adagrad_step(self):
        """Test AdaGrad parameter update."""
        # TODO: Implement after AdaGrad is created
        pass
    
    def test_adagrad_learning_rate_decay(self):
        """Test AdaGrad learning rate decay."""
        # TODO: Implement after AdaGrad is created
        pass


if __name__ == '__main__':
    unittest.main()
