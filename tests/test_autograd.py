import numpy as np
import torch
from torch.nn import functional as F 

from simplegrad.tensor import Tensor
from simplegrad.ops import relu, sigmoid, matmul


def test_autograd():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    b = Tensor(np.random.randn(3, 3), requires_grad=True)
    c = Tensor(np.random.randn(3, 3), requires_grad=True)
    d = Tensor(np.random.randn(3, 3), requires_grad=True)

    z1 = matmul(a, b)
    z2 = z1 + c
    z3 = z2 * d
    z4 = relu(z3)
    res = sigmoid(z4)

    pt_a, pt_b, pt_c, pt_d = a.to_torch(), b.to_torch(), c.to_torch(), d.to_torch()
    pt_z1 = pt_a @ pt_b
    pt_z2 = pt_z1 + pt_c
    pt_z3 = pt_z2 * pt_d
    pt_z4 = F.relu(pt_z3)
    expected = F.sigmoid(pt_z4)

    np.testing.assert_allclose(res.data, expected.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

    pt_z1.retain_grad()
    pt_z2.retain_grad()
    pt_z3.retain_grad()
    pt_z4.retain_grad()

    res.backward()
    expected.backward(torch.ones_like(expected))

    np.testing.assert_allclose(z4.grad, pt_z4.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(z3.grad, pt_z3.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(z2.grad, pt_z2.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(z1.grad, pt_z1.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

