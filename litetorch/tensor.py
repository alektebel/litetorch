



import numpy as np
import math
from math import prod


def zeros(shape, dtype="float32"):
    return Tensor(shape=shape, dtype=dtype)

def ones(shape, dtype="float32"):
    t = Tensor(shape=shape, dtype=dtype)
    for i in range(len(t.data)):
        if dtype == "float32" or dtype == "int32":
            t.data[i] = 1
        if dtype == "bool":
            t.data[i] = True
    return t

def flatindex2shapeindex(index:int, shape:tuple, strides:list)->tuple:
    ''' Given a flat index, shape and strides for each dimension, return the multidimensional index'''
    if index >= prod(shape):
        raise Exception("Index out of bounds")
    result = [0 for _ in range(len(shape))]
    for i in range(len(shape)):
        result[i] = index // strides[i]
        index = index % strides[i]
    return tuple(result)


class Tensor():
    def __init__(self, shape, data=None, dtype=None):
        # shape is the list of dimensions of the tensor
        if not isinstance(shape, tuple):
            raise Exception("Shape must be a tuple");
        total_entries = prod(shape) if len(shape) > 0 else 1
        self.data = [0 for _ in range(total_entries)]
        self.shape = shape
        self.dtype = "float32"

        self.strides = []
        if len(shape) > 0:
            self.strides = [0 for _ in range(len(shape))]
            self.strides[-1] = 1
            # The last dimension has stride 1 (hyperdimensional column)
            for i in range(len(shape)-2, -1, -1):
                self.strides[i] = self.strides[i+1] * shape[i+1]
        # In order to access an element at position (i,j,k,...), we do:
        # index = i*strides[0] + j*strides[1] + k*strides[2] + ...


    def __add__(self, other):
        
        if isinstance(other, (int, float)):
            # Scalar addition
            result = Tensor(shape=self.shape)
            for i in range(len(self.data)):
                result.data[i] = self.data[i] + other
            return result

        # Here, we apply broadcasting to sum between tensors. This is, we can pad any tensor in an arbitrary number of dimensions
        # in order to be able to sum them. Shapes must be in the same order
        # this check is s merge
        stride1 = 0;
        stride2 = 0;
        if len(self.shape) != len(other.shape):
            raise Exception(f"Incompatible shapes for broadcasting {self.shape} vs {other.shape}")
        for i in range(max(len(self.shape), len(other.shape))):
            if(self.shape[i] == other.shape[i]):
                continue
            elif self.shape[i] == 1:
                stride1 += 1
            elif other.shape[i] == 1:
                stride2 += 1
            elif stride1 * stride2 != 0:
                raise Exception(f"Incompatible shapes for broadcasting {self.shape} vs {other.shape}")
            else:
                raise Exception(f"Incompatible shapes {self.shape} vs {other.shape}")
        # Now we know the shapes are compatible for broadcasting or 
        # standard multiplication
        # first assume shapes are exactly equal. Then sum is element wise
        if stride1 == 0 and stride2 == 0:
            result = Tensor(shape=self.shape)
            for i in range(len(self.data)):
                result.data[i] = self.data[i] + other.data[i]
            return result
        elif stride1 != 0 and stride2 == 0:
            # self needs broadcasting
            result = other.clone()
            broadcast_indices = []
            for i in range(len(self.shape)):
                if self.shape[i] != other.shape[i]:
                    broadcast_indices.append(i)

            for i in range(len(other.data)):
                tuple_broadcast_index = flatindex2shapeindex(i, other.shape, other.strides)
                tuple_self_index = [0 for _ in range(len(self.shape))]
                for j in range(len(self.shape)):
                    if j in broadcast_indices:
                        tuple_self_index[j] = 0
                    else:
                        tuple_self_index[j] = tuple_broadcast_index[j]

                self_index = sum([k*s for (k,s) in zip(tuple_self_index, self.strides)])
                result.data[i] = self.data[self_index] + other.data[i] 
            return result
        elif stride1 == 0 and stride2 != 0:
            # other needs broadcasting
            result = self.clone()
            broadcast_indices = []
            for i in range(len(other.shape)):
                if other.shape[i] != self.shape[i]:
                    broadcast_indices.append(i)
            for i in range(len(self.data)):
                tuple_broadcast_index = flatindex2shapeindex(i, self.shape, self.strides)
                tuple_other_index = [0 for _ in range(len(other.shape))]
                for j in range(len(other.shape)):
                    if j in broadcast_indices:
                        tuple_other_index[j] = 0
                    else:
                        tuple_other_index[j] = tuple_broadcast_index[j]

                other_index = sum([k*s for (k,s) in zip(tuple_other_index, other.strides)])
                result.data[i] = self.data[i] + other.data[other_index] 
                
            return result
        else:
            raise Exception("Broadcasting for both tensors not implemented yet")


    def __radd__(self, other):
        return self.__add__(other)


    def clone(self):
        result = Tensor(shape=self.shape)
        for i in range(len(self.data)):
            result.data[i] = self.data[i]
        result.dtype = self.dtype
        return result


    def __sub__(self, other):
        pass
    def __matmul__(self, other):
        pass
    def __pow__(self, other):
        pass
    def __mul__(self, other):
        pass


    def __eq__(self, other):
        if isinstance(other, (int, float)):
            # Scalar comparison
            result = Tensor(shape=self.shape, dtype="bool")
            for i in range(len(self.data)):
                result.data[i] = (self.data[i] == other)
            return result

    def __getitem__(self, key):
        # When key is another tensor, we would like to do mask indexing, returning a tensor with only the elements where the mask is true
        
        if isinstance(key, slice):
            # Reuse logic for tuple indexing
            return self.__getitem__((key,))

        if isinstance(key, Tensor):
            # This will be a boolean tensor
            if key.dtype != "bool":
                raise Exception("Mask indexing requires a boolean tensor")
            # Return a 1D tensor with elements where mask is True
            if key.shape != self.shape:
                raise Exception("Mask tensor must have the same shape as the tensor being indexed")

            result_data = []
            # compute the resulting shape of key

            for i in range(len(self.data)):
                if key.data[i]:

                    result_data.append(self.data[i])
            result = Tensor(shape=(len(result_data),))
            result.data = result_data
            return result

        if isinstance(key, np.ndarray):
            # Handle numpy array indexing
            if key.dtype != np.bool_:
                raise Exception("Numpy mask indexing requires a boolean array")
            # Almost exactly same case as Tensor


        # Handle : indexing
        
        # Handle tuple indexing
        if isinstance(key, tuple):
            if isinstance(key, int) and len(key) == 1:
                # When is a single integer, we assume it's indexing the first dimension

                if key[0] < 0:
                    key = (self.shape[0] + key[0],)
                result = Tensor(shape=())
                result_shape = self.shape
                result_shape[0] = 1
                result.shape = tuple(result_shape)
                result.data = [k*self.strides[0] for k in range(self.shape[0])]
                return result
            # Normalize negative indices 
            normalized_key = []
            for i, k in enumerate(key):
                if isinstance(k, int):
                    if k< 0:
                        normalized_key.append(self.shape[i] + k)
                    else:
                        normalized_key.append(k)
                elif isinstance(k, slice):
                    slices = range(*k.indices(self.shape[i]))
                    normalized_key.append(slices)
                elif k is Ellipsis:
                    raise Exception("Ellipsis indexing not implemented yet")
                    if i < len(key) - 1:
                        limit = key[i+1]
                    normalized_key.extend([range(self.shape[j]) for j in range(i, len(self.shape)-len(key)+i+1)])
                    normalized_key[i:limit] = [range(self.shape[i]) for i in self.shape[i:]]
                else:
                    normalized_key.append(k)
            # Check bounds
            for i, k in enumerate(normalized_key):
                if isinstance(k, int):
                    if k < 0 or k >= self.shape[i]:
                        raise Exception("Index out of bounds")
            if len(key) != len(self.shape):
                raise Exception("Invalid number of indices")

            result = Tensor(shape=())
            result_shape = []
            for k in normalized_key:
                if isinstance(k, range):
                    result_shape.append(len(k))
                else:
                    result_shape.append(1)
            result.shape = tuple(result_shape)
            result.strides = []
            if len(result_shape) > 0:
                result.strides = [0 for _ in range(len(result_shape))]
                result.strides[-1] = 1
                for i in range(len(result_shape)-2, -1, -1):
                    result.strides[i] = result.strides[i+1] * result_shape[i+1]
            
            result.data = [0 for _ in range(prod(result_shape))]
            # Now fill result.data

            for i in range(len(result.data)):
                shape_index = flatindex2shapeindex(i, result_shape, result.strides)
                original_index = []
                for dim in range(len(shape_index)):
                    if isinstance(normalized_key[dim], range):
                        original_index.append(list(normalized_key[dim])[shape_index[dim]])
                    else:
                        original_index.append(normalized_key[dim])
                flat_original_index = sum([k*s for (k,s) in zip(original_index, self.strides)])
                result.data[i] = self.data[flat_original_index]
            return result

        # Handle single integer index
        if isinstance(key, int):
            if len(self.shape) == 0:
                raise Exception("Cannot index scalar tensor")
            # Normalize negative index
            if key < 0:
                key = self.shape[0] + key
            if key < 0 or key >= self.shape[0]:
                raise Exception("Index out of bounds")
            # Return a tensor with the remaining dimensions
            if len(self.shape) == 1:
                return self.data[key]
            else:
                # Return a view/slice along the first dimension
                start = key * self.strides[0]
                end = start + self.strides[0]
                result_shape = self.shape[1:]
                result = Tensor(shape=result_shape)
                result.data = self.data[start:end]
                return result
        return None
    def __setitem__(self, key, value):
        # Handle tuple indexing
        if isinstance(key, tuple):
            # Normalize negative indices
            normalized_key = []
            for i, k in enumerate(key):
                if isinstance(k, int):
                    if k< 0:
                        normalized_key.append(self.shape[i] + k)
                    else:
                        normalized_key.append(k)
                elif k is Ellipsis:
                    normalized_keys[i:] = [range(self.shape[i]) for i in self.shape[i:]]
                elif isinstance(k, slice):
                    normalized_key.append(range(*k.indices(self.shape[i])))
                else:
                    normalized_key.append(k)
            # Check bounds
            for i, k in enumerate(normalized_key):
                if k < 0 or k >= self.shape[i]:
                    raise Exception("Index out of bounds")
            if isinstance(value, Tensor):
                # Check that value shape matches the slice shape
                slice_shape = []
                for k in normalized_key:
                    if isinstance(k, range):
                        slice_shape.append(len(k))
                    else:
                        slice_shape.append(1)
                if tuple(slice_shape) != value.shape:
                    raise Exception("Shape of value does not match shape of slice")
                # Now set the values
                for i in range(len(value.data)):
                    shape_index = flatindex2shapeindex(i, value.shape, value.strides)
                    original_index = []
                    for dim in range(len(shape_index)):
                        if isinstance(normalized_key[dim], range):
                            original_index.append(list(normalized_key[dim])[shape_index[dim]])
                        else:
                            original_index.append(normalized_key[dim])
                    flat_original_index = sum([k*s for (k,s) in zip(original_index, self.strides)])
                    self.data[flat_original_index] = value.data[i]
                return
            if isinstance(value, (int, float, bool)):
                # Set all elements in the slice to value
                # Generate all indices in the slice
                def generate_indices(ranges, current_index, all_indices):
                    if len(current_index) == len(ranges):
                        all_indices.append(current_index.copy())
                        return
                    for i in ranges[len(current_index)]:
                        current_index.append(i)
                        generate_indices(ranges, current_index, all_indices)
                        current_index.pop()
                all_indices = []
                generate_indices([k if isinstance(k, range) else [k] for k in normalized_key], [], all_indices)
                for index in all_indices:
                    flat_index = sum([k*s for (k,s) in zip(index, self.strides)])
                    self.data[flat_index] = value
                return
            raise Exception("Value must be a scalar or a tensor")
            return
        # Handle single integer index
        if isinstance(key, int):
            if len(self.shape) == 0:
                raise Exception("Cannot index scalar tensor")
            # Normalize negative index
            if key < 0:
                key = self.shape[0] + key
            if key < 0 or key >= self.shape[0]:
                raise Exception("Index out of bounds")
            if len(self.shape) == 1:
                self.data[key] = value
            else:
                # Set all elements in the slice
                start = key * self.strides[0]
                end = start + self.strides[0]
                if isinstance(value, Tensor):
                    self.data[start:end] = value.data
                else:
                    for i in range(start, end):
                        self.data[i] = value
            return
    def reshape(self, other):
        pass
    def T(self):
        pass
    def sum(self, axis=None):
        pass

