# |%%--%%| <ws4ZVs8aUd|6bapWgakqP>

r"""°°°
Base example
°°°"""

import numba

import numpy as np
import pyvista as pv
import torch
from dect.directions import generate_uniform_directions
from custom_ect import compute_ect


# Settings
RESOLUTION = 256
RADIUS = 1.0
SCALE = 256


def calc_idx(theta, xg, yg, zg):
    R = RESOLUTION - 1

    heights = theta[0] * xg + theta[1] * yg + theta[2] * zg
    # idx = ((heights + 1) * RESOLUTION / 2).astype(np.int32) + 1
    # idx.ravel()[idx.ravel() > R] = R
    # return idx


def filtered_back_projection(
    v,
    ect,
    resolution,
    xg,
    yg,
    zg,
    recon,
):
    for theta, slice in zip(v.T, ect.T):
        idx = calc_idx(theta, xg, yg, zg)
        # reps = slice.ravel()[idx.ravel()]
        # recon += np.zeros_like(recon)
    return recon


xg, yg, zg = np.meshgrid(
    np.linspace(-1, 1, RESOLUTION, endpoint=False),
    np.linspace(-1, 1, RESOLUTION, endpoint=False),
    np.linspace(-1, 1, RESOLUTION, endpoint=False),
    indexing="ij",
    sparse=False,
)

recon = np.zeros(shape=(RESOLUTION, RESOLUTION, RESOLUTION))

v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu")
x = torch.load("x.pt")

ect = compute_ect(x, v, radius=RADIUS, resolution=RESOLUTION, scale=SCALE)

density = filtered_back_projection(
    v.numpy().astype(np.float32),
    ect.numpy().astype(np.float32),
    resolution=RESOLUTION,
    xg=xg,
    yg=yg,
    zg=zg,
    recon=recon,
)
