from collections import abc

import torch
from torch.nn import functional as F


def _pair(value, name):
    if isinstance(value, abc.Iterable):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError(f"{name} must contain two integers")
        return value
    return value, value


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Pure PyTorch upsample-filter-downsample operation."""
    up_x, up_y = _pair(up, "up")
    down_x, down_y = _pair(down, "down")
    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])
    if len(pad) != 4:
        raise ValueError("pad must contain two or four integers")
    if min(up_x, up_y, down_x, down_y) <= 0:
        raise ValueError("sampling factors must be positive")
    return upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, *pad)


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    batch, channel, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    output = input.reshape(-1, in_h, 1, in_w, 1, 1)
    output = F.pad(output, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    output = output.view(-1, in_h * up_y, in_w * up_x, 1)
    output = F.pad(
        output,
        [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)],
    )
    output = output[
        :,
        max(-pad_y0, 0): output.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0): output.shape[2] - max(-pad_x1, 0),
        :,
    ]
    output = output.permute(0, 3, 1, 2)
    output = output.reshape(
        -1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1
    )
    weight = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    output = F.conv2d(output, weight)
    output = output.reshape(
        -1,
        channel,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    output = output[:, :, ::down_y, ::down_x]
    expected_shape = (
        batch,
        channel,
        (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y,
        (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x,
    )
    if output.shape != expected_shape:
        raise RuntimeError(f"Unexpected upfirdn2d shape {tuple(output.shape)}; expected {expected_shape}")
    return output
