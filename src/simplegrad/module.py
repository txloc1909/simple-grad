import numpy as np

from .ops import matmul
from .tensor import Tensor

class Module:
    def parameters(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
    

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
    
    def __call__(self, x):
        return matmul(x, self.weight.transpose()) + self.bias

    def parameters(self):
        return [self.weight, self.bias]
    
    def load_state_dict(self, state_dict):
        self.weight.data = state_dict["weight"].detach().numpy()
        self.bias.data = state_dict["bias"].detach().numpy()

    def state_dict(self):
        return dict(
            weight=self.weight.data,
            bias=self.bias.data,
        )
