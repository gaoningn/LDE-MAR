import torch
from torch import nn
from torch.nn import functional as F


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dimensions = [1] * (input.ndim - bias.ndim - 1)
        input = input + bias.view(1, bias.shape[0], *rest_dimensions)
    return F.leaky_relu(input, negative_slope=negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    """Leaky ReLU with a learned channel bias, implemented in pure PyTorch."""

    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
