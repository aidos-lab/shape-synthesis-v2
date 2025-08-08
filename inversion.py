# |%%--%%| <ws4ZVs8aUd|6bapWgakqP>

r"""°°°
Base example
°°°"""

import numpy as np
import pyvista as pv
import torch
from dect.directions import generate_uniform_directions
from torch import Tensor

from custom_ect import compute_ect

# Settings
RESOLUTION = 512
RADIUS = 1.0
SCALE = 500


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
        raise Warning(
            "Setting a threshold is not used when not normalizing the density"
        )
    return recon


v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu")
x = torch.load("x.pt")

ect = compute_ect(x, v, radius=RADIUS, resolution=RESOLUTION, scale=SCALE)

density = filtered_back_projection(
    v.cuda(),
    ect.cuda(),
    resolution=RESOLUTION,
)


torch.save(density.cpu(), "density.pt")

recon_plot = density.clone()
recon_plot /= recon_plot.max()


recon_plot[recon_plot < 0.8] = 0.0

x_plot = (x.numpy() + 1) * (RESOLUTION / 2)

# Create a PyVista grid
plotter = pv.Plotter()
plotter.add_volume(
    recon_plot.cpu().numpy(),
    cmap="viridis",
    opacity="sigmoid",
)

plotter.add_points(
    x_plot,
    render_points_as_spheres=True,
    point_size=5,
    color="red",
    show_scalar_bar=False,
)

plotter.show()
