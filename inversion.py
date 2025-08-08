# |%%--%%| <ws4ZVs8aUd|6bapWgakqP>

r"""°°°
Base example
°°°"""

import numba

import numpy as np
import pyvista as pv
import torch
from torch import Tensor

from dect.directions import generate_uniform_directions
from custom_ect import compute_ect


# Settings
RESOLUTION = 256
RADIUS = 1.0
SCALE = 256


def calc_idx(theta: Tensor, xg: Tensor, yg: Tensor, zg: Tensor) -> Tensor:
    R = RESOLUTION - 1
    heights = theta[0] * xg + theta[1] * yg + theta[2] * zg
    idx = ((heights + 1) * RESOLUTION / 2).long().clamp(max=R)
    return idx


def filtered_back_projection(
    v: Tensor,
    ect: Tensor,
    resolution,
    normalized: bool = True,
    threshold: float = 0.0,
) -> Tensor:
    linspace = torch.linspace(-1, 1, resolution)
    xg, yg, zg = torch.meshgrid(
        linspace,
        linspace,
        linspace,
        indexing="ij",
    )

    recon = torch.zeros(size=(RESOLUTION, RESOLUTION, RESOLUTION))

    i = 0
    for theta, slice in zip(v.T, ect.T):
        i += 1
        idx = calc_idx(theta, xg, yg, zg)
        reps = slice[idx]
        recon += reps

    if normalized:
        recon /= recon.max()
        recon[recon < threshold] = 0.0
    elif not normalized and threshold > 0.0:
        raise Warning("Setting a threshold is not used when not normalizing the density")
    return recon

v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu")
x = torch.load("x.pt")

ect = compute_ect(x, v, radius=RADIUS, resolution=RESOLUTION, scale=SCALE)

density = filtered_back_projection(
    v.numpy().astype(np.float32),
    ect.numpy().astype(np.float32),
    resolution=RESOLUTION,
)
