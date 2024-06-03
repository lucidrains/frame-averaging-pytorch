import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, einsum, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# main class

class FrameAverage(Module):
    def __init__(
        self,
        net: Module,
        dim = 3,
        stochastic = False
    ):
        super().__init__()
        assert not stochastic, 'stochastic not implemented yet'

        self.net = net

        assert dim > 1

        self.dim = dim
        self.num_frames = 2 ** dim

        # frames are all permutations of the positive (+1) and negative (-1) eigenvectors for each dimension, iiuc
        # so there will be 2 ^ dim frames

        directions = torch.tensor([-1, 1])

        colon = slice(None)
        accum = []

        for ind in range(dim):
            dim_slice = [None] * dim
            dim_slice[ind] = Ellipsis

            accum.append(directions[dim_slice])

        accum = torch.broadcast_tensors(*accum)
        operations = torch.stack(accum, dim = -1)
        operations = rearrange(operations, '... d -> (...) d')

        assert operations.shape == (self.num_frames, dim)

        self.register_buffer('operations', operations)

        # whether to use stochastic frame averaging
        # proposed in https://arxiv.org/abs/2305.05577
        # one frame is selected at random

        self.stochastic = stochastic

    def forward(
        self,
        points,
        *args,
        frame_average_mask = None,
        **kwargs,
    ):
        """
        b - batch
        n - sequence
        d - dimension (input or source)
        e - dimension (target)
        f - frames
        """

        assert points.shape[-1] == self.dim, f'expected points of dimension {self.dim}, but received {t.shape[-1]}'

        # account for variable lengthed points

        if exists(frame_average_mask):
            frame_average_mask = rearrange(frame_average_mask, '... -> ... 1')
            points = points * frame_average_mask

        # shape must end with (seq, dim)

        points, batch_ps = pack_one(points, '* n d')

        # frame averaging logic

        if exists(frame_average_mask):
            num = reduce(points, 'b n d -> b 1 d', 'sum')
            den = reduce(frame_average_mask.float(), 'b n -> b 1 1', 'sum')
            centroid = num / den.clamp(min = 1)
        else:
            centroid = reduce(points, 'b n d -> b 1 d', 'mean')

        centered_points = points - centroid

        covariance = einsum(centered_points, centered_points, 'b n d, b n e -> b d e')

        _, eigenvectors = torch.linalg.eigh(covariance)

        frames = rearrange(eigenvectors, 'b d e -> b 1 d e') * rearrange(self.operations, 'f e -> f 1 e')

        inputs = einsum(frames, centered_points, 'b f d e, b n d -> b f n e')

        # merge frames into batch

        inputs = rearrange(inputs, 'b f ... -> (b f) ...')

        # main network forward

        out = self.net(inputs, *args, **kwargs)

        # split frames from batch

        out = rearrange(out, '(b f) ... -> b f ...', f = self.num_frames)

        # apply frames

        out = einsum(frames, out, 'b f d e, b f n e -> b f n d')

        # averaging across frames, thus "frame averaging"

        out = reduce(out, 'b f ... -> b ...', 'mean')

        # restore leading dimensions and return output

        out = unpack_one(out, batch_ps, '* n d')

        if exists(frame_average_mask):
            out = out * frame_average_mask

        return out
