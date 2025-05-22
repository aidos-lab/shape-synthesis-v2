"""
All transforms for the datasets.
"""

from functools import partial

import numpy as np
import torch
import torchvision
from dect.directions import generate_uniform_directions
from dect.ect import compute_ect_point_cloud
from dect.nn import EctConfig
from torchvision.transforms import Compose


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
    def __init__(self, config: EctConfig, device="cpu"):
        self.config = config
        self.v = generate_uniform_directions(
            config.num_thetas,
            d=config.ambient_dimension,
            seed=config.seed,
            device=device,
        )
        # self.ect_fn = torch.compile(
        #     partial(
        #         compute_ect_point_cloud,
        #         v=self.v,
        #         radius=self.config.r,
        #         resolution=self.config.resolution,
        #         scale=self.config.scale,
        #     )
        # )
        self.ect_fn = partial(
            compute_ect_point_cloud,
            v=self.v,
            radius=self.config.r,
            resolution=self.config.resolution,
            scale=self.config.scale,
            normalize=True,
        )

    def __call__(self, x):
        return self.ect_fn(x)


# class MnistTransform:
#     def __init__(self):
#         xcoords = torch.linspace(-0.5, 0.5, 28)
#         ycoords = torch.linspace(-0.5, 0.5, 28)
#         self.X, self.Y = torch.meshgrid(xcoords, ycoords)
#         self.tr = torchvision.transforms.ToTensor()
#
#     def __call__(self, data: tuple) -> Data:
#         img, y = data
#         img = self.tr(img)
#         idx = torch.nonzero(img.squeeze(), as_tuple=True)
#
#         return Data(
#             x=torch.vstack([self.X[idx], self.Y[idx]]).T,
#             # face=torch.tensor(dly.cells(), dtype=torch.long).T,
#             y=torch.tensor(y, dtype=torch.long),
#         )
