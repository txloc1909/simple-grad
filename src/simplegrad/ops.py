from itertools import product

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


def softmax(x): 
    exp_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    out = Tensor(exp_x / sum_exp, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            x.grad += out.data * (out.grad - np.sum(out.grad * out.data, axis=-1, keepdims=True))

    out._backward = _backward
    out._prev = {x, }
    return out


def conv2d(input_, weight, bias=None):
    """conv2d with stride=1, padding=0, dilation=1, layout channel-first"""
    # TODO: try implementing using im2col

    B, C_in, H_in, W_in = input_.data.shape
    C_out, _, K_H, K_W = weight.data.shape

    if bias is not None:
        assert bias.data.shape == (C_out,), f"{bias.data.shape=} mismatch"
    else: 
        bias = Tensor(np.zeros((C_out,)), requires_grad=False)

    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1

    input_buf = input_.data 
    weight_buf = weight.data
    bias_buf = bias.data
    out_buf = np.zeros((B, C_out, H_out, W_out))
    for b, c_out, h_out, w_out in product(range(B), range(C_out), range(H_out), range(W_out)):
        out_buf[b, c_out, h_out, w_out] = \
            np.sum(input_buf[b, :, h_out:h_out + K_H, w_out:w_out + K_W] * weight_buf[c_out, :, :, :]) \
            + bias_buf[c_out]

    out = Tensor(out_buf, requires_grad=input_.requires_grad or filter.requires_grad)
    
    def _backward():
        # TODO: implement
        if input_.requires_grad:
            input_.grad += np.zeros_like(input_.data)

        if weight.requires_grad:
            weight.grad += np.zeros_like(weight.data)

        if bias.requires_grad:
            bias.grad += np.sum(out.grad, axis=(0, 2, 3))

    out._backward = _backward
    out._prev = {input_, weight, bias}
    return out

