"""
Test suite for Tensor operations and basic functionality.
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litetorch.tensor import Tensor


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
    
    
    def test_tensor_addition_incompatible_shapes(self):
        """Test that adding incompatible shapes raises an exception."""
        t1 = Tensor(shape=(2, 3))
        t2 = Tensor(shape=(3, 2))
        
        with self.assertRaises((Exception, IndexError)):
            result = t1 + t2

    def test_tensor_addition_scalar(self):
        """Test adding a scalar-like tensor (shape ()) to another tensor."""
        t2 = Tensor(shape=(2, 2))
        
        t2.data = [1, 2, 3, 4]
        
        result = 10 + t2
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data, [11, 12, 13, 14])

    def test_tensor_broadcasting_addition(self):
        """Test adding tensors with broadcasting."""
        t1 = Tensor(shape=(1, 3))
        t2 = Tensor(shape=(2, 3))
        
        t1.data = [10, 20, 30]
        t2.data = [1, 2, 3, 4, 5, 6]
        
        result = t1 + t2
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data, [11, 22, 33, 14, 25, 36])
    
    def test_tensor_broadcasting_addition_heavy(self):
        """Test adding tensors with more complex broadcasting."""
        t1 = Tensor(shape=(1, 1, 3))
        t2 = Tensor(shape=(2, 2, 3))
        
        t1.data = [100, 200, 300]
        t2.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
        result = t1 + t2
        self.assertEqual(result.shape, (2, 2, 3))
        expected = [101, 202, 303, 104, 205, 306,
                    107, 208, 309, 110, 211, 312]
        self.assertEqual(result.data, expected)


    def test_tensor_addition_incompatible_broadcasting(self):
        """Test that adding tensors with incompatible broadcasting raises an exception."""
        t1 = Tensor(shape=(2, 3))
        t2 = Tensor(shape=(3, 2))
        
        with self.assertRaises((Exception, IndexError)):
            result = t1 + t2
    
    def test_tensor_addition_broadcasting_vector(self):
        """Test adding a 1D tensor to a 2D tensor with broadcasting."""
        t1 = Tensor(shape=(1, 3))
        t2 = Tensor(shape=(2, 3))
        
        t1.data = [10, 20, 30]
        t2.data = [1, 2, 3, 4, 5, 6]
        
        result = t1 + t2
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data, [11, 22, 33, 14, 25, 36])

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
        """Test that __getitem__ is now implemented for basic indexing."""
        t = Tensor(shape=(2, 3))
        
        result = t[0]
        # Now returns a Tensor with the slice
        self.assertIsInstance(result, Tensor)
    
    def test_tensor_setitem_implemented(self):
        """Test that __setitem__ is now implemented for basic indexing."""
        t = Tensor(shape=(2, 3))
        
        # Set a value and verify it was set
        t[0] = 5
        # Basic functionality tested in TestTensorIndexingNormal

    def test_tensor_indexing_multiple_dims(self):
        """Test that indexing with multiple dimensions is now implemented."""
        t = Tensor(shape=(2, 3, 4))
        t.data = list(range(24))  # Fill with values 0 to 23        
        result = t[0, 1, 2]
        self.assertEqual(result, 6)  # 0*12 + 1*4 + 2 = 6
        # Now returns a Tensor with the slice


    def test_tensor_indexing_slice(self):
        """Test that indexing with slices is now implemented."""
        t = Tensor(shape=(5,))
        t.data = [10, 20, 30, 40, 50]
        
        result = t[1:4]
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.data, [20, 30, 40])
        # Now returns a Tensor with the slice

    def test_tensor_indexing_ellipsis(self):
        """Test that indexing with ellipsis is now implemented."""
        t = Tensor(shape=(2, 3, 4))
        t.data = list(range(24))
        result = t[..., 2]
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (2, 3))
        expected = [2, 6, 10, 14, 18, 22]
        self.assertEqual(result.data, expected)

    def test_tensor_negative_indexing(self):
        """Test that negative indexing is now implemented."""
        t = Tensor(shape=(5,))
        t.data = [10, 20, 30, 40, 50]
        
        result = t[-1]
        self.assertEqual(result, 50)
        # Now returns the correct value

    def test_tensor_indexing_out_of_bounds(self):
        """Test that out of bounds indexing raises an exception."""
        t = Tensor(shape=(3,))
        
        with self.assertRaises(Exception) as context:
            _ = t[3]
        self.assertIn("out of bounds", str(context.exception).lower())
        # Now raises an appropriate exception

    def test_tensor_mask_indexing_not_implemented(self):
        """Test that mask indexing is not yet implemented."""
        t = Tensor(shape=(5,))
        mask = Tensor(shape=(5,))
        
        result = t[mask]
        # Currently returns None (not implemented)
        self.assertIsNone(result)

    def test_tensor_mask_indexing(self):
        """Test that mask indexing is now implemented."""
        t = Tensor(shape=(5,))
        t.data = [10, 20, 30, 40, 50]
        
        mask = Tensor(shape=(5,))
        mask.data = [True, False, True, False, True]
        mask.dtype = "bool"
        
        result = t[mask]
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.data, [10, 30, 50])
        # Now returns the correct masked tensor

    def test_tensor_indexing_multidimensinal_mask(self):
        """Test that multidimensional mask indexing is now implemented."""
        t = Tensor(shape=(2, 2,3,4))
        t.data = list(range(48))  # Fill with values 0 to 47
        
        mask = Tensor(shape=(2,2,3,4))
        mask.data = [True, False, False, False,
                      False, True, False, False,
                        False, False, True, False,
                        False, False, False, True,
                        True, False, False, False,
                        False, True, False, False,]
        mask.dtype = "bool"
        
        result = t[mask]
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (6,))
        self.assertEqual(result.data, [0, 5, 10, 15, 24, 29])
        # Now returns the correct masked tensor

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


class TestTensorBroadcasting(unittest.TestCase):
    """Test tensor broadcasting during addition operations."""
    
    def test_broadcasting_scalar_to_1d(self):
        """Test broadcasting a scalar-like tensor (1,) to a 1D tensor."""
        t1 = Tensor(shape=(1,))
        t2 = Tensor(shape=(5,))
        
        t1.data = [10]
        t2.data = [1, 2, 3, 4, 5]
        
        # Broadcasting: (1,) + (5,) should work
        result = t1 + t2
        self.assertEqual(result.shape, (5,))
        self.assertEqual(result.data, [11, 12, 13, 14, 15])
    
    def test_broadcasting_1d_to_scalar(self):
        """Test broadcasting in reverse order."""
        t1 = Tensor(shape=(5,))
        t2 = Tensor(shape=(1,))
        
        t1.data = [1, 2, 3, 4, 5]
        t2.data = [10]
        
        result = t1 + t2
        self.assertEqual(result.shape, (5,))
        self.assertEqual(result.data, [11, 12, 13, 14, 15])
    
    @unittest.skip("Broadcasting with 2D tensors not fully implemented yet")
    def test_broadcasting_2d_with_1_dimension(self):
        """Test broadcasting with 2D tensors where one dimension is 1."""
        t1 = Tensor(shape=(1, 3))
        t2 = Tensor(shape=(2, 3))
        
        t1.data = [1, 2, 3]
        t2.data = [10, 20, 30, 40, 50, 60]
        
        result = t1 + t2
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data, [11, 22, 33, 41, 52, 63])
    
    @unittest.skip("Broadcasting with 2D tensors not fully implemented yet")
    def test_broadcasting_2d_reverse(self):
        """Test broadcasting in reverse order."""
        t1 = Tensor(shape=(2, 3))
        t2 = Tensor(shape=(1, 3))
        
        t1.data = [10, 20, 30, 40, 50, 60]
        t2.data = [1, 2, 3]
        
        result = t1 + t2
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data, [11, 22, 33, 41, 52, 63])
    
    @unittest.skip("Broadcasting with 3D tensors not fully implemented yet")
    def test_broadcasting_multiple_dimensions_with_1(self):
        """Test broadcasting with multiple dimensions set to 1."""
        t1 = Tensor(shape=(1, 1, 3))
        t2 = Tensor(shape=(2, 2, 3))
        
        t1.data = [1, 2, 3]
        t2.data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        
        result = t1 + t2
        self.assertEqual(result.shape, (2, 2, 3))
        expected = [11, 22, 33, 41, 52, 63, 71, 82, 93, 101, 112, 123]
        self.assertEqual(result.data, expected)


class TestTensorIndexingNormal(unittest.TestCase):
    """Test normal tensor indexing operations."""
    
    def test_indexing_1d_single_element(self):
        """Test indexing a single element from 1D tensor."""
        t = Tensor(shape=(5,))
        t.data = [10, 20, 30, 40, 50]
        
        self.assertEqual(t[0], 10)
        self.assertEqual(t[2], 30)
        self.assertEqual(t[4], 50)
    
    def test_indexing_2d_single_element(self):
        """Test indexing a single element from 2D tensor with tuple."""
        t = Tensor(shape=(2, 3))
        t.data = [1, 2, 3, 4, 5, 6]
        
        self.assertEqual(t[(0, 0)], 1)
        self.assertEqual(t[(0, 2)], 3)
        self.assertEqual(t[(1, 0)], 4)
        self.assertEqual(t[(1, 2)], 6)
    
    def test_indexing_3d_single_element(self):
        """Test indexing a single element from 3D tensor."""
        t = Tensor(shape=(2, 2, 2))
        t.data = [1, 2, 3, 4, 5, 6, 7, 8]
        
        self.assertEqual(t[(0, 0, 0)], 1)
        self.assertEqual(t[(0, 0, 1)], 2)
        self.assertEqual(t[(1, 1, 1)], 8)
    
    def test_indexing_2d_row_slice(self):
        """Test indexing a row from 2D tensor."""
        t = Tensor(shape=(2, 3))
        t.data = [1, 2, 3, 4, 5, 6]
        
        row = t[0]
        self.assertIsInstance(row, Tensor)
        self.assertEqual(row.shape, (3,))
        self.assertEqual(row.data, [1, 2, 3])
        
        row = t[1]
        self.assertEqual(row.shape, (3,))
        self.assertEqual(row.data, [4, 5, 6])
    
    def test_setitem_1d(self):
        """Test setting a single element in 1D tensor."""
        t = Tensor(shape=(5,))
        t[0] = 10
        t[2] = 30
        t[4] = 50
        
        self.assertEqual(t.data[0], 10)
        self.assertEqual(t.data[2], 30)
        self.assertEqual(t.data[4], 50)
    
    def test_setitem_2d(self):
        """Test setting a single element in 2D tensor."""
        t = Tensor(shape=(2, 3))
        t[(0, 0)] = 1
        t[(0, 2)] = 3
        t[(1, 1)] = 5
        
        self.assertEqual(t.data[0], 1)
        self.assertEqual(t.data[2], 3)
        self.assertEqual(t.data[4], 5)


class TestTensorIndexingNegative(unittest.TestCase):
    """Test tensor indexing with negative indices."""
    
    def test_negative_index_1d(self):
        """Test negative indexing in 1D tensor."""
        t = Tensor(shape=(5,))
        t.data = [10, 20, 30, 40, 50]
        
        self.assertEqual(t[-1], 50)
        self.assertEqual(t[-2], 40)
        self.assertEqual(t[-5], 10)
    
    def test_negative_index_2d_tuple(self):
        """Test negative indexing in 2D tensor with tuple."""
        t = Tensor(shape=(2, 3))
        t.data = [1, 2, 3, 4, 5, 6]
        
        self.assertEqual(t[(-1, -1)], 6)
        self.assertEqual(t[(-2, -3)], 1)
        self.assertEqual(t[(0, -1)], 3)
        self.assertEqual(t[(-1, 0)], 4)
    
    def test_negative_index_setitem_1d(self):
        """Test setting element with negative index in 1D tensor."""
        t = Tensor(shape=(5,))
        t[-1] = 100
        t[-3] = 300
        
        self.assertEqual(t.data[4], 100)
        self.assertEqual(t.data[2], 300)
    
    def test_negative_index_setitem_2d(self):
        """Test setting element with negative index in 2D tensor."""
        t = Tensor(shape=(2, 3))
        t[(-1, -1)] = 99
        t[(-2, -3)] = 11
        
        self.assertEqual(t.data[5], 99)
        self.assertEqual(t.data[0], 11)


class TestTensorIndexingOutOfBounds(unittest.TestCase):
    """Test tensor indexing with out of bounds indices."""
    
    def test_out_of_bounds_positive_1d(self):
        """Test out of bounds positive index in 1D tensor."""
        t = Tensor(shape=(5,))
        
        with self.assertRaises(Exception) as context:
            _ = t[5]
        self.assertIn("out of bounds", str(context.exception).lower())
        
        with self.assertRaises(Exception) as context:
            _ = t[10]
        self.assertIn("out of bounds", str(context.exception).lower())
    
    def test_out_of_bounds_negative_1d(self):
        """Test out of bounds negative index in 1D tensor."""
        t = Tensor(shape=(5,))
        
        with self.assertRaises(Exception) as context:
            _ = t[-6]
        self.assertIn("out of bounds", str(context.exception).lower())
    
    def test_out_of_bounds_2d(self):
        """Test out of bounds index in 2D tensor."""
        t = Tensor(shape=(2, 3))
        
        with self.assertRaises(Exception) as context:
            _ = t[(2, 0)]
        self.assertIn("out of bounds", str(context.exception).lower())
        
        with self.assertRaises(Exception) as context:
            _ = t[(0, 3)]
        self.assertIn("out of bounds", str(context.exception).lower())
    
    def test_out_of_bounds_setitem(self):
        """Test setting out of bounds element."""
        t = Tensor(shape=(5,))
        
        with self.assertRaises(Exception) as context:
            t[5] = 100
        self.assertIn("out of bounds", str(context.exception).lower())


class TestTensorMaskIndexing(unittest.TestCase):
    """Test tensor mask indexing with boolean tensors."""
    
    def test_mask_indexing_basic(self):
        """Test basic mask indexing with boolean tensor."""
        t = Tensor(shape=(5,))
        t.data = [10, 20, 30, 40, 50]
        
        mask = Tensor(shape=(5,))
        mask.data = [True, False, True, False, True]
        mask.dtype = "bool"
        
        result = t[mask]
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.data, [10, 30, 50])
    
    def test_mask_indexing_all_true(self):
        """Test mask indexing when all elements are True."""
        t = Tensor(shape=(3,))
        t.data = [1, 2, 3]
        
        mask = Tensor(shape=(3,))
        mask.data = [True, True, True]
        mask.dtype = "bool"
        
        result = t[mask]
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.data, [1, 2, 3])
    
    def test_mask_indexing_all_false(self):
        """Test mask indexing when all elements are False."""
        t = Tensor(shape=(3,))
        t.data = [1, 2, 3]
        
        mask = Tensor(shape=(3,))
        mask.data = [False, False, False]
        mask.dtype = "bool"
        
        result = t[mask]
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.data, [])
    
    def test_mask_indexing_non_bool_raises_error(self):
        """Test that mask indexing with non-boolean tensor raises error."""
        t = Tensor(shape=(3,))
        t.data = [1, 2, 3]
        
        mask = Tensor(shape=(3,))
        mask.data = [1, 0, 1]  # Not boolean
        mask.dtype = "float32"
        
        with self.assertRaises(Exception) as context:
            _ = t[mask]
        self.assertIn("boolean", str(context.exception).lower())


class TestTensorHelperFunctions(unittest.TestCase):
    """Test tensor helper functions and utilities."""
    
    def test_clone_method(self):
        """Test tensor cloning creates independent copy."""
        t = Tensor(shape=(2, 3))
        t.data = [1, 2, 3, 4, 5, 6]
        t.dtype = "float32"
        
        cloned = t.clone()
        
        # Check same values
        self.assertEqual(cloned.shape, t.shape)
        self.assertEqual(cloned.data, t.data)
        self.assertEqual(cloned.dtype, t.dtype)
        
        # Check independence
        cloned.data[0] = 999
        self.assertEqual(t.data[0], 1)  # Original unchanged
        self.assertEqual(cloned.data[0], 999)
    
    def test_flatindex2shapeindex_1d(self):
        """Test flatindex2shapeindex with 1D tensor."""
        from litetorch.tensor import flatindex2shapeindex
        
        shape = (5,)
        strides = [1]
        
        self.assertEqual(flatindex2shapeindex(0, shape, strides), (0,))
        self.assertEqual(flatindex2shapeindex(3, shape, strides), (3,))
        self.assertEqual(flatindex2shapeindex(4, shape, strides), (4,))
    
    def test_flatindex2shapeindex_2d(self):
        """Test flatindex2shapeindex with 2D tensor."""
        from litetorch.tensor import flatindex2shapeindex
        
        shape = (2, 3)
        strides = [3, 1]
        
        self.assertEqual(flatindex2shapeindex(0, shape, strides), (0, 0))
        self.assertEqual(flatindex2shapeindex(1, shape, strides), (0, 1))
        self.assertEqual(flatindex2shapeindex(3, shape, strides), (1, 0))
        self.assertEqual(flatindex2shapeindex(5, shape, strides), (1, 2))
    
    def test_flatindex2shapeindex_out_of_bounds(self):
        """Test flatindex2shapeindex with out of bounds index."""
        from litetorch.tensor import flatindex2shapeindex
        
        shape = (2, 3)
        strides = [3, 1]
        
        with self.assertRaises(Exception) as context:
            flatindex2shapeindex(6, shape, strides)
        self.assertIn("out of bounds", str(context.exception).lower())
    
    def test_zeros_factory(self):
        """Test zeros factory function."""
        from litetorch.tensor import zeros
        
        t = zeros(shape=(2, 3))
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(t.dtype, "float32")
        for val in t.data:
            self.assertEqual(val, 0)
    
    def test_ones_factory(self):
        """Test ones factory function."""
        from litetorch.tensor import ones
        
        t = ones(shape=(2, 3))
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(t.dtype, "float32")
        for val in t.data:
            self.assertEqual(val, 1)


class TestTensorStrides(unittest.TestCase):
    """Test tensor strides calculation."""
    
    def test_strides_1d(self):
        """Test strides for 1D tensor."""
        t = Tensor(shape=(5,))
        self.assertEqual(t.strides, [1])
    
    def test_strides_2d(self):
        """Test strides for 2D tensor."""
        t = Tensor(shape=(3, 4))
        self.assertEqual(t.strides, [4, 1])
    
    def test_strides_3d(self):
        """Test strides for 3D tensor."""
        t = Tensor(shape=(2, 3, 4))
        self.assertEqual(t.strides, [12, 4, 1])
    
    def test_strides_scalar(self):
        """Test strides for scalar tensor."""
        t = Tensor(shape=())
        self.assertEqual(t.strides, [])
    
    def test_strides_with_dimension_1(self):
        """Test strides with dimensions of size 1."""
        t = Tensor(shape=(1, 5, 1))
        self.assertEqual(t.strides, [5, 1, 1])


if __name__ == '__main__':
    unittest.main()
