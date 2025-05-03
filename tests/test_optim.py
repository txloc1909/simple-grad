import numpy as np 
import torch
from torch.nn import functional as F

from simplegrad.tensor import Tensor
from simplegrad.ops import matmul, relu, sigmoid
from simplegrad.optim import SGD


def test_optimizer():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    b = Tensor(np.random.randn(3, 3), requires_grad=True)
    c = Tensor(np.random.randn(3, 3), requires_grad=True)
    d = Tensor(np.random.randn(3, 3), requires_grad=True)
    b_T = b.transpose()
    z1 = matmul(a, b_T)
    z2 = z1 + c
    z3 = z2 * d
    z4 = z3 - z1
    z5 = z4 / z1
    z6 = relu(z5)
    f = sigmoid(z6)
    f.backward()
    
    pt_a, pt_b, pt_c, pt_d = a.to_torch(), b.to_torch(), c.to_torch(), d.to_torch()
    pt_b_T = pt_b.T
    pt_z1 = pt_a @ pt_b_T
    pt_z2 = pt_z1 + pt_c
    pt_z3 = pt_z2 * pt_d
    pt_z4 = pt_z3 - pt_z1
    pt_z5 = pt_z4 / pt_z1
    pt_z6 = F.relu(pt_z5)
    pt_f = F.sigmoid(pt_z6)
    pt_f.backward(torch.ones_like(pt_f))

    np.testing.assert_allclose(a.grad.data, pt_a.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad.data, pt_b.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(c.grad.data, pt_c.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(d.grad.data, pt_d.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    
    lr = 0.01
    optimizer = SGD([a, b, c, d], lr)
    optimizer.step()
    optimizer.zero_grad()
    
    pt_optimizer = torch.optim.SGD([pt_a, pt_b, pt_c, pt_d], lr)
    pt_optimizer.step()
    pt_optimizer.zero_grad()
    
    np.testing.assert_allclose(a.data, pt_a.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.data, pt_b.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(c.data, pt_c.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(d.data, pt_d.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    
    np.testing.assert_allclose(a.grad, np.zeros_like(a.data))
    np.testing.assert_allclose(b.grad, np.zeros_like(b.data))
    np.testing.assert_allclose(c.grad, np.zeros_like(c.data))
    np.testing.assert_allclose(d.grad, np.zeros_like(d.data))
