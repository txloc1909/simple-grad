import numpy as np
import torch
from torch.nn import functional as F 

from simplegrad.tensor import Tensor
from simplegrad.ops import relu, sigmoid, matmul, softmax


def test_relu():
    pt_x = torch.rand((2, 2), requires_grad=True)
    x = Tensor.from_torch(pt_x)

    expected = F.relu(pt_x)
    result = relu(x)
    np.testing.assert_allclose(result.data, expected.detach().numpy())

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(x.grad.data, pt_x.grad.detach().numpy())


def test_sigmoid():
    pt_x = torch.rand((2, 2), requires_grad=True)
    x = Tensor.from_torch(pt_x)

    expected = F.sigmoid(pt_x)
    result = sigmoid(x)
    np.testing.assert_allclose(result.data, expected.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(x.grad.data, pt_x.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

def test_softmax():
    pt_x = torch.rand((2, 3), requires_grad=True)
    x = Tensor.from_torch(pt_x)

    expected = F.softmax(pt_x, dim=-1)
    result = softmax(x)

    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(x.grad.data, pt_x.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)


def test_matmul():
    pt_a = torch.rand((3, 4), requires_grad=True)
    pt_b = torch.rand((4, 5), requires_grad=True)
    a = Tensor.from_torch(pt_a)
    b = Tensor.from_torch(pt_b)

    expected = pt_a @ pt_b
    result = matmul(a, b)
    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(a.grad.data, pt_a.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad.data, pt_b.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)


def test_elemwise_add():
    pt_a = torch.rand((4, 4), requires_grad=True)
    pt_b = torch.rand((4, 4), requires_grad=True)
    a = Tensor.from_torch(pt_a)
    b = Tensor.from_torch(pt_b)

    expected = pt_a + pt_b
    result = a + b
    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(a.grad.data, pt_a.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad.data, pt_b.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

def test_elemwise_subtract():
    pt_a = torch.rand((4, 4), requires_grad=True)
    pt_b = torch.rand((4, 4), requires_grad=True)
    a = Tensor.from_torch(pt_a)
    b = Tensor.from_torch(pt_b)

    expected = pt_a - pt_b
    result = a - b
    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(a.grad.data, pt_a.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad.data, pt_b.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)


def test_elemwise_mul():
    pt_a = torch.rand((4, 4), requires_grad=True)
    pt_b = torch.rand((4, 4), requires_grad=True)
    a = Tensor.from_torch(pt_a)
    b = Tensor.from_torch(pt_b)

    expected = pt_a * pt_b
    result = a * b
    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(a.grad.data, pt_a.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad.data, pt_b.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)


def test_elemwise_div():
    pt_a = torch.rand((4, 4), requires_grad=True)
    pt_b = torch.rand((4, 4), requires_grad=True)
    a = Tensor.from_torch(pt_a)
    b = Tensor.from_torch(pt_b)

    expected = pt_a / pt_b
    result = a / b
    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(a.grad.data, pt_a.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad.data, pt_b.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

def test_elemwise_div_by_zero():
    pt_a = torch.rand((4, 4), requires_grad=True)
    pt_b = torch.zeros((4, 4), requires_grad=True)
    a = Tensor.from_torch(pt_a)
    b = Tensor.from_torch(pt_b)

    expected = pt_a / pt_b
    result = a / b
    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)
    assert np.all(np.isinf(result.data))

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(a.grad.data, pt_a.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(b.grad.data, pt_b.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

    zero_div_zero = b / b
    assert np.all(np.isnan(zero_div_zero.data))

def test_elemwise_negation():
    pt_x = torch.rand((2, 3), requires_grad=True)
    x = Tensor.from_torch(pt_x)

    expected = -pt_x 
    result = -x 

    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(x.grad.data, pt_x.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

def test_elemwise_pow(): 
    pt_x = torch.rand((2, 3), requires_grad=True)
    x = Tensor.from_torch(pt_x)

    expected = pt_x ** 2
    result = x ** 2

    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    np.testing.assert_allclose(x.grad.data, pt_x.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
