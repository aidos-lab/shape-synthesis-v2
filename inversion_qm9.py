r"""°°°
Base example
°°°"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from torch_geometric.datasets import TUDataset
from dect.directions import generate_uniform_directions
from scipy.ndimage import maximum_filter, label
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fclusterdata
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from custom_ect import compute_ect


# Settings
np.random.seed(42)
RESOLUTION = 256
RADIUS = 1.0
scale = 500



# Needs to be changed to dirac deltas.
def compute_ect(x, v, ei=None, radius=1):
    nh = x @ v
    lin = torch.linspace(-radius, radius, RESOLUTION).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh)) * (
        1 - torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    )
    ecc = ecc.sum(axis=1)
    return ecc


# ####################################################
# ### Reconstruct 3d using filtered backprojection.
# ####################################################


def calc_idx(theta, xg, yg, zg):
    R = RESOLUTION - 1
    heights = theta[0] * xg + theta[1] * yg + theta[2] * zg
    idx = ((heights + 1) * RESOLUTION / 2).astype(np.int64) + 1
    idx[idx > R] = R
    return idx


def filtered_back_projection(
    v,
    ect,
    resolution,
):
    xg, yg, zg = np.meshgrid(
        np.linspace(-1, 1, RESOLUTION, endpoint=False),
        np.linspace(-1, 1, RESOLUTION, endpoint=False),
        np.linspace(-1, 1, RESOLUTION, endpoint=False),
        indexing="ij",
        sparse=True,
    )

    recon = np.zeros(shape=(RESOLUTION, RESOLUTION, RESOLUTION))

    i = 0
    for theta, slice in zip(v.T, ect.T):
        i += 1
        idx = calc_idx(theta, xg, yg, zg)
        reps = slice.numpy()[idx]
        recon += reps
    return recon


def find_local_maxima_3d(volume, threshold=0.0, neighborhood_size=3):
    """
    Find local maxima in a 3D volume.

    Parameters:
        density (ndarray): 3D numpy array.
        threshold (float): Minimum value to be considered a peak.
        neighborhood_size (int): Size of the neighborhood window (must be odd).

    Returns:
        ids (ndarray): Nx3 array of (z, y, x) coordinates of local maxima.
    """
    local_max = maximum_filter(volume, size=neighborhood_size, mode='reflect')
    detected_peaks = (volume == local_max)

    # Optional: apply intensity threshold
    detected_peaks &= (volume > threshold)

    # Label connected components
    labeled, _ = label(detected_peaks)

    # Extract coordinates of local maxima
    ids = np.argwhere(labeled > 0)

    return ids

def merge_close_peaks(peaks, volume=None, distance_threshold=5.0, mode='max'):
    """
    Merge peaks that are closer than a distance threshold.

    Parameters:
        peaks (ndarray): Nx3 array of peak coordinates (in voxel space).
        volume (ndarray): 3D density grid used to compare peak intensities (required for mode='max').
        distance_threshold (float): Max distance between peaks to merge them.
        mode (str): 'mean' to average peak positions, 'max' to keep highest intensity peak.

    Returns:
        merged_peaks (ndarray): Mx3 array of selected peak coordinates.
    """
    if len(peaks) == 0:
        return peaks

    if mode not in {'mean', 'max'}:
        raise ValueError("mode must be 'mean' or 'max'")

    if mode == 'max' and volume is None:
        raise ValueError("volume must be provided when mode='max'")

    # Cluster peaks
    cluster_labels = fclusterdata(peaks, t=distance_threshold, criterion='distance')
    merged_peaks = []

    for label in np.unique(cluster_labels):
        cluster = peaks[cluster_labels == label]

        if mode == 'mean':
            merged = cluster.mean(axis=0)

        elif mode == 'max':
            # Round indices to int to use in volume
            intensities = [volume[tuple(np.round(p).astype(int))] for p in cluster]
            merged = cluster[np.argmax(intensities)]

        merged_peaks.append(merged)

    return np.array(merged_peaks)




####################################################
####################################################


class To3DNormalizedCoords:
    """Function to get the 3D coordinates from QM9."""

    def __call__(self, data):
        x = data.x[:, -3:]
        x -= x.mean(axis=0)
        x /= x.norm(dim=-1).max()
        data.x = x
        return data


# v = generate_thetas()
v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu")

x = torch.tensor(
    [
        [1.38963128, 1.30881270, -1.84807340],
        [0.25134028, 1.61695797, -1.17378268],
        [2.34174165, 2.30279245, -2.05237245],
        [1.74670811, 0.30770268, -2.08813647],
        [0.05520299, 2.95681402, -0.72831891],
        [-0.81096017, 0.83736826, -0.91802960],
        [-1.10099896, 3.26937222, -0.10832467],
        [0.91391981, 3.92808926, -0.99610811],
        [-1.73558122, 2.55577479, 0.18972739],
        [-1.19481836, 4.21421525, 0.12159553],
        [2.07766676, 3.62803433, -1.63042725],
        [2.81650095, 4.38833396, -1.67628005],
        [3.23433192, 2.10381485, -2.63100063],
        [-0.57931678, -0.63039174, -0.91326348],
        [-1.77506852, -1.28327640, -0.40022312],
        [0.20195626, -0.97701420, -0.35217348],
        [-0.31077873, -0.89430482, -2.04282024],
        [-3.00526392, -1.38331587, -1.13839590],
        [-1.88488065, -1.72681611, 0.94414745],
        [-4.15739837, -2.00672026, -0.71479091],
        [-2.95735117, -1.01520162, -2.24735924],
        [-4.19469156, -2.42753431, 0.61887066],
        [-5.04818305, -2.24955496, -1.33679401],
        [-3.08464301, -2.27809040, 1.38882121],
        [-5.07725553, -2.96292210, 1.05906295],
        [-3.24228124, -2.56751763, 2.40845989],
        [-1.09224915, -1.37255578, 1.68160820],
    ]
)

scale_factor = x.norm(dim=-1).max().item()
x -= x.mean(axis=0)
x /= x.norm(dim=-1).max()
x *= 0.7

ect = compute_ect(x, v, radius=RADIUS)
density = filtered_back_projection(v.numpy(), ect, resolution=RESOLUTION)

recon_plot = torch.from_numpy(density).float().clone()
recon_plot /= recon_plot.max()
recon_plot[recon_plot < 0.8] = 0.0

peak_ids = find_local_maxima_3d(recon_plot.cpu().numpy(), threshold=0.8)
merged_peaks = merge_close_peaks(peak_ids, volume=recon_plot.cpu().numpy(), distance_threshold=5.0)

x_plot = (x.numpy() + 1) * (RESOLUTION / 2)

# Create a PyVista grid
plotter = pv.Plotter()
plotter.add_volume(
    recon_plot.cpu().numpy(),
    cmap="viridis",
    opacity="linear",
)

voxel_to_normalized = lambda p: (p / (RESOLUTION / 2)) - 1.0
recon_peaks_normalized = voxel_to_normalized(merged_peaks)

x_np = x.numpy()
recon_np = recon_peaks_normalized

# Cost matrix: distances between all original and reconstructed points
cost_matrix = cdist(x_np, recon_np)
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Matched reconstructed points
recon_matched = recon_np[col_ind]
position_errors = np.linalg.norm(x_np - recon_matched, axis=1) * scale_factor / 0.7
mean_error = position_errors.mean()
max_error = position_errors.max()

print("Mean reconstruction error (Å):", mean_error)
print("Max reconstruction error (Å):", max_error)

# plotter.add_points(
#     x_plot,
#     render_points_as_spheres=True,
#     point_size=5,
#     color="red",
#     show_scalar_bar=False,
# )

# add detected peaks
plotter.add_points(
    merged_peaks,
    render_points_as_spheres=True,
    point_size=6,
    color="red",
    show_scalar_bar=False,
)


plotter.show()
