import numpy as np
from dect.directions import generate_uniform_directions, generate_multiview_directions

from custom_ect import compute_ect
from src.datasets.single_molecule import get_dataset

# from inversion import filtered_back_projection
from src.inversion.fbp import reconstruct_point_cloud
from src.metrics.molecule import compute_metrics
from src.plotting.recon import plot_reconstruction

#######################################################################
np.random.seed(42)
RESOLUTION = 256  # Abbreviated to R
RADIUS = 1.0  # Abbreviated to r, fixed to 1 for now.
SCALE = 500  # Fixed hyperparameter for now. Is sets the bandwidth for the dirac approximation.
#######################################################################


#########################################################################################################
#### Reconstruct
#########################################################################################################


# v has shape [3, R]
v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu")

# [N,3] where N is the number of atoms in the molecule.
x, z, to_angstrom = get_dataset()

# Compute the ECT.
ect = compute_ect(x, v, radius=RADIUS, scale=SCALE, resolution=RESOLUTION)


# x_recon is the reconstructed point cloud.
# The additional tuple is for computing the loss.
x_recon, (recon_plot, merged_peaks) = reconstruct_point_cloud(ect.cuda(), v.cuda())


#########################################################################################################
#### Evaluation metrics
#########################################################################################################

(
    mean_error,
    max_error,
    common,
    diffs,
    abs_diffs,
    edges_recon,
    recon_map,
    recon_aligned,
    orig_map,
    Z,
) = compute_metrics(x, x_recon, to_angstrom, z)

#########################################################################################################
#### Plotting
#########################################################################################################


plot_reconstruction(
    common,
    diffs,
    abs_diffs,
    edges_recon,
    recon_map,
    recon_aligned,
    orig_map,
    Z,
    recon_plot,
    RESOLUTION,
    merged_peaks,
)
