import numpy as np
import torch 
from torch.nn import functional as F

from simplegrad.tensor import Tensor
from simplegrad.ops import conv2d


def test_conv2d():
    B, C, H, W = 1, 3, 5, 5
    C_out, H_out, W_out = 2, 3, 3
    pt_input = torch.rand((B, C, H, W), requires_grad=True)
    pt_weight = torch.rand((C_out, C, H_out, W_out), requires_grad=True)
    pt_bias = torch.rand((C_out,), requires_grad=True)
    input_ = Tensor.from_torch(pt_input)
    weight = Tensor.from_torch(pt_weight)
    bias = Tensor.from_torch(pt_bias)

    expected = F.conv2d(pt_input, pt_weight, pt_bias)
    result = conv2d(input_, weight, bias)
    
    np.testing.assert_allclose(result.data, expected.detach().numpy(), 
                               rtol=1.3e-6, atol=1e-5)

    expected.backward(torch.ones_like(expected))
    result.backward()
    
    np.testing.assert_allclose(input_.grad.data, pt_input.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(weight.grad.data, pt_weight.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
    np.testing.assert_allclose(bias.grad.data, pt_bias.grad.detach().numpy(),
                               rtol=1.3e-6, atol=1e-5)
