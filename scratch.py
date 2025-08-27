import functools

import numpy as np
import pyvista as pv
import torch
from dect.directions import generate_uniform_directions

from custom_ect import compute_ect_channels
from src.datasets.single_molecule import get_dataset
from src.inversion.fbp_improved import gather_indices, get_grid, reconstruct_point_cloud

torch.set_float32_matmul_precision("medium")

#######################################################################
np.random.seed(42)
RESOLUTION = 64  # Abbreviated to R
RADIUS = 1.0  # Abbreviated to r, fixed to 1 for now.
SCALE = 100  # Fixed hyperparameter for now. Is sets the bandwidth for the dirac approximation.
DEVICE = "cuda"  # Device to compute on.
WIDTH = 3
THRESHOLD = 0.7
#######################################################################


######################################################################
######################################################################
######################################################################

x, z, to_angstrom = get_dataset()
z[z == 1] = 0
z[z == 6] = 1
z[z == 7] = 2
z[z == 8] = 3
z[z == 9] = 4

x = x.to(DEVICE)
print(x.norm(dim=-1).max())
z = z.to(DEVICE)
max_channels = 5
v = generate_uniform_directions(RESOLUTION, d=3, seed=2013, device=DEVICE).to(DEVICE)
ect = (
    compute_ect_channels(
        x,
        v,
        radius=RADIUS,
        scale=SCALE,
        resolution=RESOLUTION,
        channels=z,
        index=torch.zeros(len(x), device=DEVICE).long(),
        max_channels=max_channels,
    ).to(DEVICE)
    # .squeeze()
)


idx = get_grid(resolution=ect.shape[-1], v=v)
recon = functools.partial(gather_indices, idx=idx)
batched_recon = torch.vmap(recon)

pts, z_recon, density = reconstruct_point_cloud(
    ect, batched_recon=batched_recon, width=WIDTH, threshold=THRESHOLD
)


# # |%%--%%| <mfb8i2jQji|DQaEKLUjnP>

print(z)
print(z_recon)
atom_number = 0
x_filtered = (x[z == atom_number] + 1) * RESOLUTION / 2
x_recon = (pts[z_recon == atom_number] + 1) * RESOLUTION / 2  # [mask[atom_number]]
print("True", x_filtered.shape)
print("RECON", x_recon.shape)

plotter = pv.Plotter()
plotter.add_points(
    x_filtered.cpu().numpy(),
    render_points_as_spheres=True,
    point_size=10,
)

plotter.add_points(
    x_recon.cpu().numpy(),
    color="red",
    render_points_as_spheres=True,
    point_size=10,
)
plotter.add_volume(
    density[atom_number].view(RESOLUTION, RESOLUTION, RESOLUTION).cpu().numpy(),
    opacity="sigmoid_5",
)
plotter.show()
#
#
#
# # print("Recon shape", coords.shape)
# # print("true shape", x.shape)
# #
# #
# # x_plot = (1 + x_filtered) / 2 * RESOLUTION
# # plotter = pv.Plotter()
# # plotter.add_points(x_plot.cpu().numpy(), render_points_as_spheres=True, point_size=10)
# #
# # plotter.add_points(coords, color="red", render_points_as_spheres=True, point_size=15)
# #
# # # plotter.add_points(x_filtered.numpy(), render_points_as_spheres=True, color="red")
# # plotter.add_volume(
# #     res[atom_type].view(RESOLUTION, RESOLUTION, RESOLUTION).cpu().numpy(),
# #     opacity="sigmoid_5",
# # )
# # plotter.show()
