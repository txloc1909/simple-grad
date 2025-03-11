import numpy as np
import torch
from torch.nn import functional as F 

from main import Tensor
from main import relu, sigmoid, matmul


def test_relu():
    x = torch.rand((2, 2))
    expected = F.relu(x)
    result = relu(Tensor.from_torch(x))
    np.testing.assert_allclose(result.data, expected.numpy())


def test_sigmoid():
    x = torch.rand((2, 2))
    expected = F.sigmoid(x)
    result = sigmoid(Tensor.from_torch(x))
    np.testing.assert_allclose(result.data, expected.numpy(),
                               rtol=1.3e-6, atol=1e-5)

def test_matmul():
    a, b = torch.rand((3, 4)), torch.rand((4, 5))
    expected = a @ b
    result = matmul(Tensor.from_torch(a), Tensor.from_torch(b))
    np.testing.assert_allclose(result.data, expected.numpy(), 
                               rtol=1.3e-6, atol=1e-5)


def test_elemwise_add():
    a, b = torch.rand((4, 4)), torch.rand((4, 4))
    expected = a + b
    result = Tensor.from_torch(a) + Tensor.from_torch(b)
    np.testing.assert_allclose(result.data, expected.numpy())


def test_elemwise_mul():
    a, b = torch.rand((4, 4)), torch.rand((4, 4))
    expected = a * b
    result = Tensor.from_torch(a) * Tensor.from_torch(b)
    np.testing.assert_allclose(result.data, expected.numpy())
