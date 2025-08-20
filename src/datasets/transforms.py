"""
All transforms for the datasets.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from dect.directions import generate_multiview_directions, generate_uniform_directions
from dect.ect import compute_ect_point_cloud
from dect.nn import EctConfig
from torch import nn


@dataclass
class EctTransformConfig(EctConfig):
    resolution: int
    structured_directions: bool


def get_transform(config: EctTransformConfig):
    transform = EctTransform(config=config)
    return transform


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
