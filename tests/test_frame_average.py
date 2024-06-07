import pytest

import torch
from torch import nn
from torch.nn import Module
from frame_averaging_pytorch import FrameAverage

@pytest.mark.parametrize('stochastic', (True, False))
@pytest.mark.parametrize('dim', (2, 3, 4))
@pytest.mark.parametrize('has_mask', (True, False))
def test_frame_average(
    stochastic: bool,
    dim: int,
    has_mask: bool
):

    net = torch.nn.Linear(dim, dim)

    net = FrameAverage(
        net,
        dim = dim,
        stochastic = stochastic
    )

    points = torch.randn(4, 1024, dim)

    mask = None
    if has_mask:
        mask = torch.ones(4, 1024).bool()

    out = net(points, frame_average_mask = mask)
    assert out.shape == points.shape

def test_frame_average_manual():

    net = torch.nn.Linear(3, 3)

    fa = FrameAverage()
    points = torch.randn(4, 1024, 3)

    framed_inputs, frame_average_fn = fa(points)

    net_out = net(framed_inputs)

    frame_averaged = frame_average_fn(net_out)

    assert frame_averaged.shape == points.shape

def test_frame_average_multiple_inputs_and_outputs():

    class Network(Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(3, 3)
            self.to_out1 = nn.Linear(3, 3)
            self.to_out2 = nn.Linear(3, 3)

        def forward(self, x, mask):
            x = x.masked_fill(~mask[..., None], 0.)
            hidden = self.net(x)
            return self.to_out1(hidden), self.to_out2(hidden)

    net = Network()
    net = FrameAverage(net)

    points = torch.randn(4, 1024, 3)
    mask = torch.ones(4, 1024).bool()

    out1, out2 = net(points, mask)

    assert out1.shape == out2.shape == points.shape
