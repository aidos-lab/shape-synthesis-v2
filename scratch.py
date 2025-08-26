import functools

import numpy as np
import torch
from dect.directions import generate_uniform_directions

from custom_ect import compute_ect_channels
from src.datasets.single_molecule import get_dataset
from src.inversion.fbp_improved import gather_indices, get_grid, reconstruct_point_cloud

torch.set_float32_matmul_precision("medium")

#######################################################################
np.random.seed(42)
RESOLUTION = 128  # Abbreviated to R
RADIUS = 1.0  # Abbreviated to r, fixed to 1 for now.
SCALE = 200  # Fixed hyperparameter for now. Is sets the bandwidth for the dirac approximation.
DEVICE = "cuda"  # Device to compute on.
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
z = z.to(DEVICE)
max_channels = 5
v = generate_uniform_directions(RESOLUTION, d=3, seed=2015, device=DEVICE).to(DEVICE)
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
    )
    .to(DEVICE)
    .squeeze()
)
ect = torch.vstack([ect for _ in range(4)])
print(ect.shape)


idx = get_grid(resolution=ect.shape[-1], v=v)
recon = functools.partial(gather_indices, idx=idx)
batched_recon = torch.vmap(recon)

print(torch._dynamo.list_backends())

reconstruct_point_cloud = torch.compile(
    reconstruct_point_cloud,
)

for i in range(10):
    print(i)
    pts, mask = reconstruct_point_cloud(
        ect, batched_recon=batched_recon, width=9, threshold=0.5
    )


# # |%%--%%| <mfb8i2jQji|DQaEKLUjnP>
# print(pts.shape)
#
# atom_number = 1
# x_filtered = x[z==atom_number]
# x_recon = pts[atom_number]#[mask[atom_number]]
# print("True",x_filtered.shape)
# print("RECON",pts[atom_number][mask[atom_number]].shape)
#
#
#
#
# plotter = pv.Plotter()
# plotter.add_points(x_filtered.cpu().numpy(), render_points_as_spheres=True, point_size=10)
#
# plotter.add_points(x_recon.cpu().numpy(), color="red", render_points_as_spheres=True, point_size=10)
#
# # plotter.add_points(x_filtered.numpy(), render_points_as_spheres=True, color="red")
# # plotter.add_volume(
# #     res[atom_type].view(RESOLUTION, RESOLUTION, RESOLUTION).cpu().numpy(),
# #     opacity="sigmoid_5",
# # )
# plotter.show()
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
