"""
All transforms for the datasets.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from dect.directions import generate_multiview_directions, generate_uniform_directions

# from dect.ect import compute_ect_channels
from dect.nn import EctConfig
from torch import Tensor, nn


def compute_ect_channels(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    channels: Tensor,
    index: Tensor | None = None,
    max_channels: int | None = None,
    normalize: bool = False,
    use_diracs: bool = False,
):
    """
    Allows for channels within the point cloud to separated in different
    ECT's.

    Input is a point cloud of size (B*num_point_per_pc,num_features) with an additional feature vector with the
    channel number for each point and the output is ECT for shape [B,num_channels,num_thetas,resolution]
    """

    # Ensure that the scale is in the right device
    scale = torch.tensor([scale], device=x.device)

    # Compute maximum channels.
    if max_channels is None:
        max_channels = int(channels.max()) + 1

    if index is not None:
        batch_len = int(index.max() + 1)
    else:
        batch_len = 1
        index = torch.zeros(
            size=(len(x),),
            dtype=torch.int32,
            device=x.device,
        )

    # Fix the index to interleave with the channel info.
    index = max_channels * index + channels

    # v is of shape [ambient_dimension, num_thetas]
    num_thetas = v.shape[1]

    out_shape = (resolution, batch_len * max_channels, num_thetas)

    # Node heights have shape [num_points, num_directions]
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution, device=x.device).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    if use_diracs:
        ecc = ecc * (1 - ecc)
    output = torch.zeros(
        size=out_shape,
        device=nh.device,
    )

    output.index_add_(1, index, ecc)
    ect = output.movedim(0, 1)

    if normalize:
        ect = ect / torch.amax(ect, dim=(-2, -3))

    # Returns the ect as [batch_len, num_thetas, resolution]
    return ect.reshape(-1, max_channels, resolution, num_thetas)


@dataclass
class EctTransformConfig(EctConfig):
    resolution: int
    structured_directions: bool
    max_channels: int
    use_diracs: bool


def get_transform(config: EctTransformConfig):
    transform = EctChannelsTransform(config=config)
    return transform


class EctChannelsTransform(nn.Module):
    def __init__(self, config: EctTransformConfig):
        super().__init__()

        self.config = config

        # Instantiate structured directions.
        if config.structured_directions:
            v = generate_multiview_directions(
                config.num_thetas,
                d=config.ambient_dimension,
            )
        else:
            v = generate_uniform_directions(
                config.num_thetas,
                d=config.ambient_dimension,
                seed=config.seed,
            )
        self.v = nn.Parameter(v, requires_grad=False)
        self.ect_fn = partial(
            compute_ect_channels,
            v=self.v,
            radius=self.config.r,
            resolution=self.config.resolution,
            scale=self.config.scale,
            max_channels=self.config.max_channels,
            use_diracs=self.config.use_diracs,
            normalize=False,  # We do this in the forward.
        )

    def forward(self, x, index, channels):
        ect = self.ect_fn(x=x, index=index, channels=channels)

        amax = torch.amax(ect, dim=(-1, -2), keepdim=True)
        amax[amax == 0] = 1
        ect = ect / amax
        return ect


class EctTransform(nn.Module):
    def __init__(self, config: EctTransformConfig):
        super().__init__()

        self.config = config

        # Instantiate structured directions.
        if config.structured_directions:
            v = generate_multiview_directions(
                config.num_thetas,
                d=config.ambient_dimension,
            )
        else:
            v = generate_uniform_directions(
                config.num_thetas,
                d=config.ambient_dimension,
                seed=config.seed,
            )
        self.v = nn.Parameter(v, requires_grad=False)
        self.ect_fn = partial(
            compute_ect_point_cloud,
            v=self.v,
            radius=self.config.r,
            resolution=self.config.resolution,
            scale=self.config.scale,
        )

    def forward(self, x):
        ect = self.ect_fn(x=x)
        ect = ect / torch.amax(ect, dim=(-1, -2), keepdim=True)
        return ect


class FixedLength:
    def __init__(self, length=128):
        self.length = length

    def __call__(self, data):
        res = data.clone()
        if data.x.shape[0] < self.length:
            idx = torch.tensor(np.random.choice(len(data.x), self.length, replace=True))
        else:
            idx = torch.tensor(
                np.random.choice(len(data.x), self.length, replace=False)
            )
        res.x = data.x[idx]
        return res


class To3DNormalizedCoords:
    """Function to get the 3D coordinates from QM9."""

    def __call__(self, data):
        x = data.pos
        x -= x.mean(axis=0)
        x /= x.norm(dim=-1).max()
        x *= 0.7
        data.pos = x
        return data
