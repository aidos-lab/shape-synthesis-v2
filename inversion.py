# |%%--%%| <ws4ZVs8aUd|6bapWgakqP>

r"""°°°
Base example
°°°"""

import numpy as np
import pyvista as pv
import torch
from dect.directions import generate_uniform_directions

from custom_ect import compute_ect

# Settings
RESOLUTION = 512
RADIUS = 1.0
SCALE = 500


def calc_idx(theta, grid):
    R = RESOLUTION - 1
    heights = torch.inner(theta.to(torch.float32), grid.movedim(0, 3).to(torch.float32))
    idx = ((heights + 1) * RESOLUTION / 2).long() + 1
    idx[idx > R] = R
    return idx


def filtered_back_projection(
    v,
    ect,
    resolution,
    grid,
    recon,
):
    for theta, slice in zip(v.T, ect.T):
        idx = calc_idx(theta, grid)
        reps = slice[idx]
        recon += reps
    return recon


xg, yg, zg = np.meshgrid(
    np.linspace(-1, 1, RESOLUTION, endpoint=False),
    np.linspace(-1, 1, RESOLUTION, endpoint=False),
    np.linspace(-1, 1, RESOLUTION, endpoint=False),
    indexing="ij",
    sparse=False,
)
grid = torch.tensor(np.stack([xg, yg, zg]))

recon = torch.tensor(
    np.zeros(shape=(RESOLUTION, RESOLUTION, RESOLUTION), dtype=np.float32)
)

v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu")
x = torch.load("x.pt")

ect = compute_ect(x, v, radius=RADIUS, resolution=RESOLUTION, scale=SCALE)

density = filtered_back_projection(
    v.cuda(),
    ect.cuda(),
    resolution=RESOLUTION,
    grid=grid.cuda(),
    recon=recon.cuda(),
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
