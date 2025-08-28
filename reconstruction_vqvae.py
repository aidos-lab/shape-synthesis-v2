import argparse
import functools
import os

import numpy as np
import pyvista as pv
import torch
import torch_geometric
from dect.directions import generate_multiview_directions, generate_uniform_directions
from torch_geometric.datasets import QM9

from src.inversion.fbp import gather_indices, get_grid, reconstruct_point_cloud

torch.set_float32_matmul_precision("medium")

#######################################################################
np.random.seed(42)
RESOLUTION = 64  # Abbreviated to R
RADIUS = 1.0  # Abbreviated to r, fixed to 1 for now.
SCALE = 200  # Fixed hyperparameter for now. Is sets the bandwidth for the dirac approximation.
DEVICE = "cuda"  # Device to compute on.
WIDTH = 3
THRESHOLD = 0.7
GLOBAL_SCALE = 6
#######################################################################

#########################################################################################################
#### Reconstruct
#########################################################################################################


v = generate_multiview_directions(RESOLUTION, d=3)
idx = get_grid(resolution=RESOLUTION, v=v)
recon = functools.partial(gather_indices, idx=idx)
batched_recon = torch.vmap(recon)


ect_recon = torch.load("results/recon.pt")
ect_gt = torch.load("results/ect.pt")

ect_gt = (ect_gt + 1) / 2
ect_recon = (ect_recon + 1) / 2

plt.imshow(ect_recon[1][2].cpu().numpy())

print(ect_recon.min(), ect_recon.max())
print(ect_gt.min(), ect_gt.max())

print((ect_recon - ect_gt).norm())


# |%%--%%| <n941Hd3alw|s3Gxt9McQl>

print(ect_recon.shape)
print(ect_gt.shape)
print(v.shape)


r_pts, r_z, density = reconstruct_point_cloud(
    ect_recon[0],
    batched_recon=batched_recon,
    width=WIDTH,
    threshold=THRESHOLD,
)

r_pts_gt, r_z_gt, density_gt = reconstruct_point_cloud(
    ect_gt[0],
    batched_recon=batched_recon,
    width=WIDTH,
    threshold=THRESHOLD,
)

print((density - density_gt).norm())

# |%%--%%| <s3Gxt9McQl|3cVAk3oTSX>

density.shape

print(r_pts_gt)
print(r_z_gt)
print(r_pts)
print(r_z)

plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_volume(density[0].squeeze().cpu().numpy(), opacity="sigmoid_8")
plotter.subplot(0, 1)
plotter.add_volume(density_gt[0].squeeze().cpu().numpy(), opacity="sigmoid_8")
plotter.link_views()
plotter.show()
