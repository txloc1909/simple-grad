import numpy as np
import torch


def relu(tensor):
    out = Tensor(np.maximum(0, tensor.data), requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            tensor.grad += (tensor.data > 0) * out.grad
    
    out._backward = _backward
    return out


def sigmoid(tensor):
    out = Tensor(1 / (1 + np.exp(-tensor.data)), requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            tensor.grad += out.data * (1 - out.data) * out.grad
    
    out._backward = _backward
    return out


def matmul(tensor1, tensor2):
    assert isinstance(tensor2, Tensor), "Operand must be a Tensor"
    out = Tensor(tensor1.data @ tensor2.data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    def _backward():
        if tensor1.requires_grad:
            tensor1.grad += out.grad @ tensor2.data.T
        if tensor2.requires_grad:
            tensor2.grad += tensor1.data.T @ out.grad
    
    out._backward = _backward
    return out


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
    
    @staticmethod
    def from_torch(tensor):
        return Tensor(tensor.detach().cpu().numpy(), requires_grad=tensor.requires_grad)
    
    def __add__(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        # TODO: implement backprop
        self.grad = np.ones_like(self.data)
        self._backward()


class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
    
    def __call__(self, x):
        return matmul(x, self.weight) + self.bias
    
    def load_state_dict(self, state_dict):
        self.weight.data = state_dict["weight"].copy()
        self.bias.data = state_dict["bias"].copy()


def main():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    layer1 = Linear(2, 3)
    layer2 = Linear(3, 1)

    out = relu(layer1(x))
    out = sigmoid(layer2(out))

    print("Forward output:", out.data)


if __name__ == "__main__":
    main()
