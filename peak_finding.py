import numpy as np
import torch
from scipy.cluster.hierarchy import fclusterdata
from scipy.ndimage import maximum_filter, label
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
    local_max = maximum_filter(volume, size=neighborhood_size, mode='reflect')
    detected_peaks = (volume == local_max)

    # Optional: apply intensity threshold
    detected_peaks &= (volume > threshold)

    # Label connected components
    labeled, _ = label(detected_peaks)

    # Extract coordinates of local maxima
    ids = np.argwhere(labeled > 0)

    return ids

def merge_close_peaks(peaks, volume=None, distance_threshold=4.0, mode='max'):
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