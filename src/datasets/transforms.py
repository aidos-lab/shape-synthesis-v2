"""
All transforms for the datasets.
"""

from functools import partial

import numpy as np
import torch
from dect.directions import (
    generate_2d_directions,
    generate_multiview_directions,
    generate_uniform_directions,
)
from dect.ect import compute_ect_points
from dect.nn import EctConfig


def get_transform(compiled: bool = False, resolution: int = 128, d: int = 3):
    return EctTransform(
        config=EctConfig(
            num_thetas=resolution,
            resolution=resolution,
            r=1.1,
            scale=500,
            ect_type="points",
            ambient_dimension=d,
            normalized=True,
            seed=2013,
        ),
        structured_directions=True,
        device="cuda",
        compiled=compiled,
    )


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


class EctTransform:
    def __init__(
        self,
        config: EctConfig,
        structured_directions: bool = True,
        device: str = "cpu",
        compiled: bool = False,
    ):
        self.config = config
        if structured_directions:
            if config.ambient_dimension == 3:
                num_t = config.num_thetas // 3
                remainder = config.num_thetas % 3

                v_pre = generate_multiview_directions(num_t + 3, d=3)

                self.v = torch.hstack(
                    [
                        v_pre[0][:, :num_t],
                        v_pre[1][:, :num_t],
                        v_pre[2][:, : num_t + remainder],
                    ]
                ).to(device)
                print(self.v.shape)
                print(remainder)
                print(num_t)
            elif config.ambient_dimension == 2:
                self.v = generate_2d_directions(config.num_thetas).to(device)
        else:
            self.v = generate_uniform_directions(
                config.num_thetas,
                d=config.ambient_dimension,
                seed=config.seed,
                device=device,
            )
        self.ect_fn = partial(
            compute_ect_points,
            v=self.v,
            radius=self.config.r,
            resolution=self.config.resolution,
            scale=self.config.scale,
        )
        if compiled:
            self.ect_fn = torch.compile(self.ect_fn)

    def __call__(self, x, index):
        ect = self.ect_fn(x=x, index=index)
        ect = ect / torch.amax(ect, dim=(-1, -2), keepdim=True)
        return ect


class To3DNormalizedCoords:
    """Function to get the 3D coordinates from QM9."""

    def __call__(self, data):
        x = data.pos
        x -= x.mean(axis=0)
        x /= x.norm(dim=-1).max()
        x *= 0.7
        data.pos = x
        return data
