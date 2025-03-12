import numpy as np
import torch
from torch import nn

from simplegrad.tensor import Tensor
from simplegrad.module import Linear


def test_load_state_dict_from_torch():
    D_IN, D_OUT = 32, 64
    pt_module = nn.Linear(D_IN, D_OUT, bias=True)
    module = Linear(D_IN, D_OUT)
    module.load_state_dict(pt_module.state_dict())

    np.testing.assert_allclose(module.weight.data, pt_module.weight.detach().numpy())
    np.testing.assert_allclose(module.bias.data, pt_module.bias.detach().numpy())


def test_linear_module():
    B, D_IN, D_OUT = 8, 32, 64
    pt_module = nn.Linear(D_IN, D_OUT, bias=True)
    module = Linear(D_IN, D_OUT)
    module.load_state_dict(pt_module.state_dict())

    x = Tensor(np.random.rand(B, D_IN))
    pt_x = x.to_torch()

    expected = pt_module(pt_x)
    result = module(x) 

    np.testing.assert_allclose(result.data, expected.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)

