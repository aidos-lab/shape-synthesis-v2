import numpy as np
import torch
from scipy.cluster.hierarchy import fclusterdata
from scipy.ndimage import label, maximum_filter
from torch import Tensor


def find_local_maxima_3d(volume: Tensor, threshold=0.0, neighborhood_size=3):
    """
    Find local maxima in a 3D volume.

    Parameters:
        density (Tensor): 3D torch tensor.
        threshold (float): Minimum value to be considered a peak.
        neighborhood_size (int): Size of the neighborhood window (must be odd).

    Returns:
        ids (ndarray): Nx3 array of (z, y, x) coordinates of local maxima.
    """
    local_max = maximum_filter(volume, size=neighborhood_size, mode="reflect")
    detected_peaks = volume == local_max

    # Optional: apply intensity threshold
    detected_peaks &= volume > threshold

    # Label connected components
    labeled, _ = label(detected_peaks)

    # Extract coordinates of local maxima
    ids = np.argwhere(labeled > 0)

    return ids


def merge_close_peaks(peaks, volume=None, distance_threshold=4.0, mode="max"):
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

    if mode not in {"mean", "max"}:
        raise ValueError("mode must be 'mean' or 'max'")

    if mode == "max" and volume is None:
        raise ValueError("volume must be provided when mode='max'")

    # Cluster peaks
    cluster_labels = fclusterdata(peaks, t=distance_threshold, criterion="distance")
    merged_peaks = []

    for label in np.unique(cluster_labels):
        cluster = peaks[cluster_labels == label]

        if mode == "mean":
            merged = cluster.mean(axis=0)

        elif mode == "max":
            # Round indices to int to use in volume
            intensities = [volume[tuple(np.round(p).astype(int))] for p in cluster]
            merged = cluster[np.argmax(intensities)]

        merged_peaks.append(merged)

    return np.array(merged_peaks)


def calc_idx(
    theta: Tensor,
    xg: Tensor,
    yg: Tensor,
    zg: Tensor,
    resolution: int,
) -> Tensor:
    R = resolution - 1
    heights = theta[0] * xg + theta[1] * yg + theta[2] * zg
    idx = ((heights + 1) * resolution / 2).long().clamp(max=R)
    return idx


def filtered_back_projection(
    v: Tensor,
    ect: Tensor,
    resolution: int,
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

    recon = torch.zeros(size=(resolution, resolution, resolution))

    i = 0
    for theta, slice in zip(v.T, ect.T):
        i += 1
        idx = calc_idx(theta, xg, yg, zg, resolution)
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


def reconstruct_point_cloud(ect, v):
    # Assuming square ects for now.
    resolution = ect.shape[-1]

    # get the density after backprojecting
    recon_plot = filtered_back_projection(
        v.numpy(), ect, resolution=resolution, normalized=True, threshold=0.7
    )

    peak_ids = find_local_maxima_3d(recon_plot.cpu().numpy(), threshold=0.7)
    merged_peaks = merge_close_peaks(
        peak_ids, volume=recon_plot.cpu().numpy(), distance_threshold=4.0
    )
    voxel_to_normalized = lambda p: (p / (resolution / 2)) - 1.0
    recon_peaks_normalized = voxel_to_normalized(merged_peaks)
    recon_np = recon_peaks_normalized
    return recon_np, (recon_plot, merged_peaks)
