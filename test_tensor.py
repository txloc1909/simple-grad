import numpy as np
import torch

from main import Tensor


def test_from_torch():
    pt_tensor = torch.rand((2, 2))
    tensor = Tensor.from_torch(pt_tensor)

    assert pt_tensor.requires_grad == tensor.requires_grad
    np.testing.assert_allclose(tensor.data, pt_tensor.numpy())


def test_to_torch():
    tensor = Tensor(np.random.rand(2, 2))
    pt_tensor = tensor.to_torch()

    assert pt_tensor.requires_grad == tensor.requires_grad
    np.testing.assert_allclose(tensor.data, pt_tensor.numpy())
