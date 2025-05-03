import numpy as np


class Optimizer:
    def __init__(self):
        pass

    def step(self):
        return NotImplemented 


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__()
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.data)
