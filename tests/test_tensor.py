"""
Test suite for Tensor operations and basic functionality.
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litetorch.autograd.tensor import Tensor


class TestTensorCreation(unittest.TestCase):
    """Test Tensor creation with various shapes and edge cases."""
    
    def test_tensor_creation_1d(self):
        """Test creating a 1D tensor."""
        t = Tensor(shape=(5,))
        self.assertEqual(t.shape, (5,))
        self.assertEqual(len(t.data), 5)
        self.assertEqual(t.dtype, "float32")
    
    def test_tensor_creation_2d(self):
        """Test creating a 2D tensor."""
        t = Tensor(shape=(3, 4))
        self.assertEqual(t.shape, (3, 4))
        self.assertEqual(len(t.data), 12)
    
    def test_tensor_creation_3d(self):
        """Test creating a 3D tensor."""
        t = Tensor(shape=(2, 3, 4))
        self.assertEqual(t.shape, (2, 3, 4))
        self.assertEqual(len(t.data), 24)
    
    def test_tensor_creation_single_element(self):
        """Test creating a tensor with single element."""
        t = Tensor(shape=(1,))
        self.assertEqual(t.shape, (1,))
        self.assertEqual(len(t.data), 1)
    
    def test_tensor_creation_scalar(self):
        """Test creating a scalar (0-dimensional) tensor."""
        t = Tensor(shape=())
        self.assertEqual(t.shape, ())
        self.assertEqual(len(t.data), 1)
    
    def test_tensor_creation_large(self):
        """Test creating a large tensor."""
        t = Tensor(shape=(10, 10, 10))
        self.assertEqual(t.shape, (10, 10, 10))
        self.assertEqual(len(t.data), 1000)
    
    def test_tensor_creation_invalid_shape_not_tuple(self):
        """Test that non-tuple shapes raise an exception."""
        with self.assertRaises(Exception) as context:
            Tensor(shape=[2, 3])
        self.assertIn("Shape must be a tuple", str(context.exception))
    
    def test_tensor_initial_data_is_zeros(self):
        """Test that initial tensor data is all zeros."""
        t = Tensor(shape=(2, 3))
        for val in t.data:
            self.assertEqual(val, 0)


class TestTensorAddition(unittest.TestCase):
    """Test tensor addition operations."""
    
    def test_tensor_addition_same_shape(self):
        """Test adding two tensors with the same shape."""
        t1 = Tensor(shape=(2, 3))
        t2 = Tensor(shape=(2, 3))
        
        # Set some values
        t1.data = [1, 2, 3, 4, 5, 6]
        t2.data = [10, 20, 30, 40, 50, 60]
        
        result = t1 + t2
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data, [11, 22, 33, 44, 55, 66])
    
    def test_tensor_addition_1d(self):
        """Test adding two 1D tensors."""
        t1 = Tensor(shape=(5,))
        t2 = Tensor(shape=(5,))
        
        t1.data = [1, 2, 3, 4, 5]
        t2.data = [5, 4, 3, 2, 1]
        
        result = t1 + t2
        self.assertEqual(result.shape, (5,))
        self.assertEqual(result.data, [6, 6, 6, 6, 6])
    
    def test_tensor_addition_zeros(self):
        """Test adding tensor to itself with zeros."""
        t1 = Tensor(shape=(3, 3))
        t2 = Tensor(shape=(3, 3))
        
        result = t1 + t2
        self.assertEqual(result.shape, (3, 3))
        for val in result.data:
            self.assertEqual(val, 0)
    
    def test_tensor_addition_incompatible_types(self):
        """Test that adding incompatible types raises an exception."""
        t = Tensor(shape=(2, 2))
        with self.assertRaises(Exception) as context:
            result = t + 5
        self.assertIn("Incompatible types", str(context.exception))
    
    def test_tensor_addition_incompatible_shapes(self):
        """Test that adding incompatible shapes raises an exception."""
        t1 = Tensor(shape=(2, 3))
        t2 = Tensor(shape=(3, 2))
        
        with self.assertRaises((Exception, IndexError)):
            result = t1 + t2


class TestTensorSubtraction(unittest.TestCase):
    """Test tensor subtraction operations."""
    
    def test_tensor_subtraction_not_implemented(self):
        """Test that subtraction is not yet implemented."""
        t1 = Tensor(shape=(2, 2))
        t2 = Tensor(shape=(2, 2))
        
        result = t1 - t2
        # Currently returns None (not implemented)
        self.assertIsNone(result)


class TestTensorMultiplication(unittest.TestCase):
    """Test tensor element-wise multiplication."""
    
    def test_tensor_multiplication_not_implemented(self):
        """Test that element-wise multiplication is not yet implemented."""
        t1 = Tensor(shape=(2, 2))
        t2 = Tensor(shape=(2, 2))
        
        result = t1 * t2
        # Currently returns None (not implemented)
        self.assertIsNone(result)


class TestTensorMatmul(unittest.TestCase):
    """Test tensor matrix multiplication."""
    
    def test_tensor_matmul_not_implemented(self):
        """Test that matrix multiplication is not yet implemented."""
        t1 = Tensor(shape=(2, 3))
        t2 = Tensor(shape=(3, 2))
        
        result = t1 @ t2
        # Currently returns None (not implemented)
        self.assertIsNone(result)


class TestTensorPower(unittest.TestCase):
    """Test tensor power operations."""
    
    def test_tensor_power_not_implemented(self):
        """Test that power operation is not yet implemented."""
        t = Tensor(shape=(2, 2))
        
        result = t ** 2
        # Currently returns None (not implemented)
        self.assertIsNone(result)


class TestTensorShape(unittest.TestCase):
    """Test tensor shape property."""
    
    def test_tensor_shape_1d(self):
        """Test shape of 1D tensor."""
        t = Tensor(shape=(10,))
        self.assertEqual(t.shape, (10,))
    
    def test_tensor_shape_2d(self):
        """Test shape of 2D tensor."""
        t = Tensor(shape=(5, 7))
        self.assertEqual(t.shape, (5, 7))
    
    def test_tensor_shape_3d(self):
        """Test shape of 3D tensor."""
        t = Tensor(shape=(2, 3, 4))
        self.assertEqual(t.shape, (2, 3, 4))
    
    def test_tensor_shape_scalar(self):
        """Test shape of scalar tensor."""
        t = Tensor(shape=())
        self.assertEqual(t.shape, ())


class TestTensorReshape(unittest.TestCase):
    """Test tensor reshape operation."""
    
    def test_tensor_reshape_not_implemented(self):
        """Test that reshape is not yet implemented."""
        t = Tensor(shape=(2, 3))
        
        result = t.reshape((3, 2))
        # Currently returns None (not implemented)
        self.assertIsNone(result)


class TestTensorTranspose(unittest.TestCase):
    """Test tensor transpose operation."""
    
    def test_tensor_transpose_not_implemented(self):
        """Test that transpose is not yet implemented."""
        t = Tensor(shape=(2, 3))
        
        result = t.T()
        # Currently returns None (not implemented)
        self.assertIsNone(result)


class TestTensorSum(unittest.TestCase):
    """Test tensor sum operation."""
    
    def test_tensor_sum_not_implemented(self):
        """Test that sum is not yet implemented."""
        t = Tensor(shape=(2, 3))
        
        result = t.sum()
        # Currently returns None (not implemented)
        self.assertIsNone(result)
    
    def test_tensor_sum_with_axis_not_implemented(self):
        """Test that sum with axis is not yet implemented."""
        t = Tensor(shape=(2, 3))
        
        result = t.sum(axis=0)
        # Currently returns None (not implemented)
        self.assertIsNone(result)


class TestTensorIndexing(unittest.TestCase):
    """Test tensor indexing operations."""
    
    def test_tensor_getitem_not_implemented(self):
        """Test that __getitem__ is not yet implemented."""
        t = Tensor(shape=(2, 3))
        
        result = t[0]
        # Currently returns None (not implemented)
        self.assertIsNone(result)
    
    def test_tensor_setitem_not_implemented(self):
        """Test that __setitem__ is not yet implemented."""
        t = Tensor(shape=(2, 3))
        
        # Currently does nothing (not implemented)
        t[0] = 5
        # No error should be raised, but no action taken


class TestTensorEdgeCases(unittest.TestCase):
    """Test tensor edge cases and boundary conditions."""
    
    def test_tensor_very_large_shape(self):
        """Test creating a tensor with very large dimensions."""
        t = Tensor(shape=(100, 100))
        self.assertEqual(len(t.data), 10000)
    
    def test_tensor_single_dimension_large(self):
        """Test creating a tensor with one large dimension."""
        t = Tensor(shape=(1000,))
        self.assertEqual(len(t.data), 1000)
    
    def test_tensor_many_dimensions(self):
        """Test creating a tensor with many dimensions."""
        t = Tensor(shape=(2, 2, 2, 2, 2))
        self.assertEqual(len(t.data), 32)
        self.assertEqual(t.shape, (2, 2, 2, 2, 2))
    
    def test_tensor_dimension_with_one(self):
        """Test creating a tensor with dimensions of size 1."""
        t = Tensor(shape=(1, 5, 1))
        self.assertEqual(len(t.data), 5)
        self.assertEqual(t.shape, (1, 5, 1))


if __name__ == '__main__':
    unittest.main()
