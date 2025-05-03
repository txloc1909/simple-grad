import numpy as np
import torch 
from torch.nn import functional as F

from simplegrad.tensor import Tensor
from simplegrad.loss import cross_entropy


def test_cross_entropy():
    B, C = 3, 5
    x = Tensor(np.random.randn(B, C), requires_grad=True)
    y = Tensor(np.random.randint(0, C, size=(B,)), requires_grad=False)
    loss = cross_entropy(x, y)

    pt_x = x.to_torch()
    pt_y = y.to_torch().to(torch.long)
    pt_loss = F.cross_entropy(pt_x, pt_y)

    np.testing.assert_allclose(loss.data, pt_loss.detach().numpy(),
                               atol=1e-5, rtol=1.3e-6)

    loss.backward()
    pt_loss.backward(torch.ones_like(pt_loss))
    np.testing.assert_allclose(x.grad.data, pt_x.grad.detach().numpy(),
                               atol=1e-5, rtol=1.3e-6)
