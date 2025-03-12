import numpy as np

from .ops import matmul
from .tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
    
    def __call__(self, x):
        return matmul(x, self.weight) + self.bias
    
    def load_state_dict(self, state_dict):
        self.weight.data = state_dict["weight"].copy()
        self.bias.data = state_dict["bias"].copy()
