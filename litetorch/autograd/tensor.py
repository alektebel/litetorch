



import math

class Tensor():
    def __init__(self, shape, data=None, dtype=None):
        # shape is the list of dimensions of the tensor
        if not isinstance(shape, tuple):
            raise Exception("Shape must be a tuple");
        total_entries = math.prod(shape)
        self.data = [0 for _ in range(total_entries)]
        self.shape = shape
        self.dtype = "float32"
    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise Exception("Incompatible types to sum together")
        # Here, we apply broadcasting to sum between tensors. This is, we can pad any tensor in an arbitrary number of dimensions
        # in order to be able to sum them. Shapes must be in the same order.
        # This check validates shape compatibility for broadcasting.
        stride1 = 0;
        stride2 = 0;
        for i in range(max(len(self.shape), len(other.shape))):
            if(self.shape[i+stride1] == other.shape[i+stride2]):
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
        else:
            # TODO: implement broadcasting sum
            return None

    def __sub__(self, other):
        pass
    def __matmul__(self, other):
        pass
    def __pow__(self, other):
        pass
    def __mul__(self, other):
        pass
    def __getitem__(self, key):
        pass
    def __setitem__(self, key, value):
        pass
    def reshape(self, other):
        pass
    def T(self):
        pass
    def sum(self, axis=None):
        pass

