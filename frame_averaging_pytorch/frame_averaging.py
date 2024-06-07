from __future__ import annotations

from random import randrange
from optree import tree_map

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
        net: Module | None = None,
        dim = 3,
        stochastic = False,
        invariant_output = False,
        return_stochastic_as_augmented_pos = False  # will simply return points as augmented points of same shape on forward
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
        self.return_stochastic_as_augmented_pos = return_stochastic_as_augmented_pos

        # invariant output setting

        self.invariant_output = invariant_output

    def forward(
        self,
        points,
        *args,
        frame_average_mask = None,
        return_framed_inputs_and_averaging_function = False,
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

        # define the frame averaging function

        def frame_average(out):
            if not self.invariant_output:
                # apply frames

                out = einsum(frames, out, 'b f d e, b f ... e -> b f ... d')

            if not self.stochastic:
                # averaging across frames, thus "frame averaging"

                out = reduce(out, 'b f ... -> b ...', 'mean')
            else:
                out = rearrange(out, 'b 1 ... -> b ...')

            return out

        # if one wants to handle the framed inputs externally

        if return_framed_inputs_and_averaging_function or not exists(self.net):

            if self.stochastic and self.return_stochastic_as_augmented_pos:
                return rearrange(inputs, 'b 1 ... -> b ...')

            return inputs, frame_average

        # merge frames into batch

        inputs = rearrange(inputs, 'b f ... -> (b f) ...')

        # if batch is expanded by number of frames, any tensor being passed in for args and kwargs needed to be expanded as well
        # automatically take care of this

        if not self.stochastic:
            args, kwargs = tree_map(
                lambda el: (
                    repeat(el, 'b ... -> (b f) ...', f = num_frames)
                    if torch.is_tensor(el)
                    else el
                )
            , (args, kwargs))

        # main network forward

        out = self.net(inputs, *args, **kwargs)

        # use tree map to handle multiple outputs

        out = tree_map(lambda t: rearrange(t, '(b f) ... -> b f ...', f = num_frames) if torch.is_tensor(t) else t, out)
        out = tree_map(lambda t: frame_average(t) if torch.is_tensor(t) else t, out)

        return out
