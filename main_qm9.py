import numpy as np
import torch_geometric
from torch_geometric.datasets import TUDataset
from dect.directions import generate_uniform_directions, generate_multiview_directions

from custom_ect import compute_ect

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

v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu")

dataset = TUDataset(
    "data/",
    name="QM9",
    use_node_attr=True,
)
# |%%--%%| <FFEc22Et2k|bhLelXxfKp>

data = dataset[0].x[=1]


# |%%--%%| <bhLelXxfKp|mNazI4LQli>


#########################################################################################################
#########################################################################################################

# x, z, to_angstrom = get_dataset()
# # Compute the ECT
# ect = compute_ect(x, v, radius=RADIUS, scale=SCALE, resolution=RESOLUTION)
#
# recon_np, (recon_plot, merged_peaks) = reconstruct_point_cloud(ect.cuda(), v.cuda())
#
# #########################################################################################################
# #### Metrics
# #########################################################################################################
#
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
