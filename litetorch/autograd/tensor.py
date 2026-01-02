



import numpy as np
import math
from math import prod


def zeros(shape, dtype="float32"):
    return Tensor(shape=shape, dtype=dtype)

def ones(shape, dtype="float32"):
    t = Tensor(shape=shape, dtype=dtype)
    for i in range(len(t.data)):
        t.data[i] = 1
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
    def __init__(self, data=None, shape, dtype= None):
        # shape is the list of dimensions of the tensor
        if not isinstance(shape, tuple):
            raise Exception("Shape must be a tuple");
        total_entries = prod(shape)
        print(total_entries)
        self.data = [0 for _ in range(total_entries)]
        self.shape = shape
        self.dtype = "float32"

        self.strides = [0 for _ in range(len(shape))]
        self.strides[-1] = 1
        # The last dimension has stride 1 (hiperdimensional column)
        for i in range(len(shape)-2, -1, -1):
            self.strides[i] = self.strides[i+1] * shape[i+1]
        # In order to acces an element at position (i,j,k,...), we do:
        # index = i*strides[0] + j*strides[1] + k*strides[2] + ...


    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise Exception("Incompatible types to sum together")
        # Here, we apply broadcasting to sum between tensors. This is, we can pad any tensor in an arbitrary number of dimrnsions
        # in order to be able to sum them. Shapes must be in the same order
        # this check is s merge
        stride1 = 0;
        stride2 = 0;
        for i in range(max(len(self.shape), len(other.shape))):
            if(self.shape[i+stride1] == other.other[i+stride2]):
                continue
            elif self.shape[i] == 1:
                stride1 += 1
            elif other.shape[i] == 1:
                stride2 += 1
            elif stride1 * stride2 != 0:
                raise Exception(f"Incompatible shapes for broadcasting
                {self.shape} vs {other.shape}")
            else:
                raise Exception(f"Incompatible shapes {self.shape} vs {other.shape}")
        # Now we know the shape are compatible for broadcasting or 
        # stabdard muktiplication
        # first assume shapes are exactly equal. Them sum is element wise
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
                for j in range(len(self.shape))
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
                for j in range(len(other.shape))
                    if j in broadcast_indices:
                        tuple_other_index[j] = 0
                    else:
                        tuple_other_index[j] = tuple_broadcast_index[j]

                other_index = sum([k*s for (k,s) in zip(tuple_other_index, other.strides)])
                result.data[i] = self.data[i] + other.data[other_index] 
                
            return result
        else:
            raise Exception("Broadcasting for both tensors not implemented yet")

    def clone(self):
        result = Tensor(shape=self.shape)
        for i in range(len(self.data)):
            result.data[i] = self.data[i]
        result.dtype = self.dtype
        return result


    def __sub__(self, other):
        pass
    def __matmul__(sefl, other):
        pass
    def __pow__(self, other):
        pass
    def __mul__(self, other):
        pass
    def __getitem__(self, *key):
        if key >= prod(self.shape):
            raise Exception("Index out of bounds")
        # When key is another tensor, we would like to do mask indexing, returning a tensor with only the elements where the mask is true
        if isinstance(key, Tensor):
            # This will be a boolean tensor
            if key.dtype != "bool":
                raise Exception("Mask indexing requires a boolean tensor")
        if isinstance(key, np.ndarray):
            pass
        if len(key) != len(self.shape):
            raise Exception("Invalid number of indices")
        # Key is a tuple of indices, one perdimension
        # Normal indexing
        if isinstance(key, tuple):
            return self.data[sum([k*s for (k,s) in zip(key, self.strides)])]
    def __setitem__(self, *key, value):
        if key >= prod(self.shape):
            raise Exception("Index out of bounds")
        self.data[sum([k*s for (k,s) in zip(key, self.strides)])] = value
        return
    def reshape(self, other):
        pass
    def T(self):
        pass
    def sum(self, axis=None):
        pass

