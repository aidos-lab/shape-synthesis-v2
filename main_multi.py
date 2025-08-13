import numpy as np
from dect.directions import generate_uniform_directions, generate_multiview_directions
import pyvista as pv
import torch

from custom_ect import compute_ect
from src.datasets.single_molecule import get_dataset

# from inversion import filtered_back_projection
from src.inversion.fbp import reconstruct_point_cloud
from src.metrics.molecule import compute_metrics
from src.plotting.recon import plot_reconstruction

#######################################################################
np.random.seed(42)
RESOLUTION = 300
RADIUS = 1.0
SCALE = 500
#######################################################################

#########################################################################################################
#### Reconstruct
#########################################################################################################

# v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu")
v_pre = generate_multiview_directions(RESOLUTION // 3, d=3)
v = torch.hstack([v_pre[0], v_pre[1], v_pre[2]])


x, z, to_angstrom = get_dataset()
# Compute the ECT
ect = compute_ect(x, v, radius=RADIUS, scale=SCALE, resolution=RESOLUTION)

recon_np, (recon_plot, merged_peaks) = reconstruct_point_cloud(
    ect.cuda(), v.cuda(), threshold=0.8
)


#########################################################################################################
#### Metrics
#########################################################################################################

plotter = pv.Plotter()

plotter.add_points(
    points=recon_np, render_points_as_spheres=True, point_size=8, color="red"
)
plotter.add_points(
    points=x.numpy(), render_points_as_spheres=True, point_size=8, color="blue"
)
plotter.add_points(
    points=v.T.numpy(), render_points_as_spheres=True, point_size=2, color="green"
)


plotter.show()

# (
#     mean_error,
#     max_error,
#     common,
#     diffs,
#     abs_diffs,
#     edges_recon,
#     recon_map,
#     recon_aligned,
#     orig_map,
#     Z,
# ) = compute_metrics(x, recon_np, to_angstrom, z)
#
# #########################################################################################################
# #### Plotting
# #########################################################################################################
#
#
# plot_reconstruction(
#     common,
#     diffs,
#     abs_diffs,
#     edges_recon,
#     recon_map,
#     recon_aligned,
#     orig_map,
#     Z,
#     recon_plot,
#     RESOLUTION,
#     merged_peaks,
# )
