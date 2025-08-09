import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.metrics.bond_detection import build_bonds_pairwise_torch


def _to_tuple_list(edges_t: torch.Tensor):
    return [
        (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
        for i, j in edges_t.tolist()
    ]


def compute_metrics(x, recon_np, to_angstrom, z):
    x_np = x.numpy()
    # Cost matrix: distances between all original and reconstructed points
    cost_matrix = cdist(x_np, recon_np)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Matched reconstructed points (align to original index order)
    recon_aligned = np.empty_like(recon_np)
    recon_aligned[row_ind] = recon_np[col_ind]

    # Position errors in Å
    position_errors = (
        np.linalg.norm(x_np - recon_aligned, axis=1) * to_angstrom
    )  # x_np (normalized); scale to Å
    mean_error = position_errors.mean()
    max_error = position_errors.max()

    positions_orig_A = x * to_angstrom  # x is torch (normalized); scale to Å
    positions_recon_A = (
        torch.from_numpy(recon_aligned).to(dtype=torch.float32) * to_angstrom
    )
    Z = z.to(torch.long)

    edges_orig, lengths_orig = build_bonds_pairwise_torch(positions_orig_A, Z)
    edges_recon, lengths_recon = build_bonds_pairwise_torch(positions_recon_A, Z)

    # Compare bonds present in both graphs
    orig_map = {
        e: float(l) for e, l in zip(_to_tuple_list(edges_orig), lengths_orig.tolist())
    }
    recon_map = {
        e: float(l) for e, l in zip(_to_tuple_list(edges_recon), lengths_recon.tolist())
    }

    orig_keys = set(orig_map.keys())
    recon_keys = set(recon_map.keys())

    common = sorted(orig_keys & recon_keys)
    missed = sorted(orig_keys - recon_keys)
    spurious = sorted(recon_keys - orig_keys)

    diffs = np.array([recon_map[e] - orig_map[e] for e in common], dtype=float)
    abs_diffs = np.abs(diffs)

    print("Mean reconstruction error (Å):", mean_error)
    print("Max reconstruction error (Å):", max_error)
    print(f"# bonds (orig): {len(orig_keys)}")
    print(f"# bonds (recon): {len(recon_keys)}")
    print(f"# common bonds: {len(common)}")
    print(f"# missed (orig only): {len(missed)}")
    print(f"# spurious (recon only): {len(spurious)}")

    return (
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
    )
