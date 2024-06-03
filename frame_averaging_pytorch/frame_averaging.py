from random import randrange
from optree import tree_flatten, tree_unflatten

import torch
from torch.nn import Module

from einops import rearrange, repeat, reduce, einsum

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class FrameAverage(Module):
    def __init__(
        self,
        net: Module,
        dim = 3,
        stochastic = False,
        invariant_output = False
    ):
        super().__init__()
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
            dim_slice[ind] = colon

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

        self.invariant_output = invariant_output

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

        assert points.shape[-1] == self.dim, f'expected points of dimension {self.dim}, but received {points.shape[-1]}'

        # account for variable lengthed points

        if exists(frame_average_mask):
            frame_average_mask = rearrange(frame_average_mask, '... -> ... 1')
            points = points * frame_average_mask

        # shape must end with (batch, seq, dim)

        batch, seq_dim, input_dim = points.shape

        # frame averaging logic

        if exists(frame_average_mask):
            num = reduce(points, 'b n d -> b 1 d', 'sum')
            den = reduce(frame_average_mask.float(), 'b n 1 -> b 1 1', 'sum')
            centroid = num / den.clamp(min = 1)
        else:
            centroid = reduce(points, 'b n d -> b 1 d', 'mean')

        centered_points = points - centroid

        if exists(frame_average_mask):
            centered_points = centered_points * frame_average_mask

        covariance = einsum(centered_points, centered_points, 'b n d, b n e -> b d e')

        _, eigenvectors = torch.linalg.eigh(covariance)

        # if stochastic, just select one random operation

        num_frames = self.num_frames
        operations = self.operations

        if self.stochastic:
            rand_frame_index = randrange(self.num_frames)

            operations = operations[rand_frame_index:(rand_frame_index + 1)]
            num_frames = 1

        # frames

        frames = rearrange(eigenvectors, 'b d e -> b 1 d e') * rearrange(operations, 'f e -> f 1 e')

        # inverse frame op

        inputs = einsum(frames, centered_points, 'b f d e, b n d -> b f n e')

        # merge frames into batch

        inputs = rearrange(inputs, 'b f ... -> (b f) ...')

        # if batch is expanded by number of frames, any tensor being passed in for args and kwargs needed to be expanded as well
        # automatically take care of this

        if not self.stochastic:
            flattened_args_kwargs, tree_spec = tree_flatten([args, kwargs])

            mapped_args_kwargs = []

            for el in flattened_args_kwargs:
                if torch.is_tensor(el):
                    el = repeat(el, 'b ... -> (b f) ...', f = num_frames)

                mapped_args_kwargs.append(el)

            args, kwargs = tree_unflatten(tree_spec, mapped_args_kwargs)

        # main network forward

        out = self.net(inputs, *args, **kwargs)

        # handle if output is a tuple - just follow convention that first output is the one to be frame averaged
        # (todo) - handle multiple outputs that need frame averaging

        is_multiple_output = isinstance(out, tuple)

        if is_multiple_output:
            out, *rest = out

        # split frames from batch

        out = rearrange(out, '(b f) ... -> b f ...', f = num_frames)

        if not self.invariant_output:
            # apply frames

            out = einsum(frames, out, 'b f d e, b f n e -> b f n d')

        if not self.stochastic:
            # averaging across frames, thus "frame averaging"

            out = reduce(out, 'b f ... -> b ...', 'mean')
        else:
            out = rearrange(out, 'b 1 ... -> b ...')

        if not is_multiple_output:
            return out

        return (out, *rest)

