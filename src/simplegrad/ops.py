import numpy as np
import torch

from .tensor import Tensor


def relu(tensor):
    out = Tensor(np.maximum(0, tensor.data), requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            tensor.grad += (tensor.data > 0) * out.grad
    
    out._backward = _backward
    out._prev = {tensor, }
    return out


def sigmoid(tensor):
    out = Tensor(1 / (1 + np.exp(-tensor.data)), requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            tensor.grad += out.data * (1 - out.data) * out.grad
    
    out._backward = _backward
    out._prev = {tensor, }
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
    out._prev = {tensor1, tensor2}
    return out


