import numpy as np

from .tensor import Tensor


def cross_entropy(pred, target): 
    # NOTE: for the sake of completeness, should target has gradient?
    assert not target.requires_grad     

    assert pred.data.shape[0] == target.data.shape[0]

    bs, num_classes = pred.data.shape
    
    # softmax
    exp_x = np.exp(pred.data - np.max(pred.data, axis=-1, keepdims=True))
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    probs = exp_x / sum_exp

    target_one_hot = np.eye(num_classes)[target.data.astype(int)]

    loss = Tensor(
        np.mean(-np.sum(target_one_hot * np.log(probs + 1e-9), axis=-1))
    )

    def _backward():
        if pred.requires_grad: 
            pred.grad += (probs - target_one_hot) / bs


    loss._backward = _backward
    loss._prev = {pred, }
    return loss
