r"""°°°
Base example
°°°"""

import numpy as np
import pyvista as pv
import torch
from dect.directions import generate_uniform_directions
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch import Tensor

from bond_detection import build_bonds_pairwise_torch
from inversion import filtered_back_projection
from peak_finding import find_local_maxima_3d, merge_close_peaks

# Settings
np.random.seed(42)
RESOLUTION = 256
RADIUS = 1.0
scale = 500
# turn this on, if you want to find/plot bonds and evaluate reconstruction errors
evaluate_errors = False



# Needs to be changed to dirac deltas.
def compute_ect(x: Tensor, v: Tensor, ei=None, radius=1) -> Tensor:
    nh = x @ v
    lin = torch.linspace(-radius, radius, RESOLUTION).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh)) * (
        1 - torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    )
    ecc = ecc.sum(axis=1)
    return ecc

class To3DNormalizedCoords:
    """Function to get the 3D coordinates from QM9."""

    def __call__(self, x):
        data = x[:, -3:]
        data -= data.mean(axis=0)
        data /= data.norm(dim=-1).max()
        data *= 0.7
        return data


#########################################################################################################
##################### Example for a molecule with atom types ############################################
#########################################################################################################

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
z = torch.tensor([6, 6, 6, 1, 6, 8, 7, 7, 1, 1, 6, 1, 1, 6, 6, 1, 1, 6, 6, 6, 1, 6, 1, 6, 1, 1, 1])


# Normalize and get the scaling factor
to_angstrom = x.norm(dim=-1).max().item()/0.7
x = To3DNormalizedCoords()(x)

# Compute the ECT
ect = compute_ect(x, v, radius=RADIUS)

# get the density after backprojecting
recon_plot = filtered_back_projection(v.numpy(), ect, resolution=RESOLUTION, normalized=True, threshold=0.7)

peak_ids = find_local_maxima_3d(recon_plot.cpu().numpy(), threshold=0.7)
merged_peaks = merge_close_peaks(peak_ids, volume=recon_plot.cpu().numpy(), distance_threshold=4.0)


#########################################################################################################
################## Here plotting and evaluation of reconstructed positions starts #######################
#########################################################################################################

x_plot = (x.numpy() + 1) * (RESOLUTION / 2)

# Create a PyVista grid and plot the density
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

# Matched reconstructed points (align to original index order)
recon_aligned = np.empty_like(recon_np)
recon_aligned[row_ind] = recon_np[col_ind]

if evaluate_errors:
    # Position errors in Å
    position_errors = np.linalg.norm(x_np - recon_aligned, axis=1) * to_angstrom # x_np (normalized); scale to Å
    mean_error = position_errors.mean()
    max_error = position_errors.max()
    print("Mean reconstruction error (Å):", mean_error)
    print("Max reconstruction error (Å):", max_error)


    positions_orig_A = x * to_angstrom  # x is torch (normalized); scale to Å
    positions_recon_A = torch.from_numpy(recon_aligned).to(dtype=torch.float32) * to_angstrom
    Z = z.to(torch.long)

    edges_orig, lengths_orig = build_bonds_pairwise_torch(positions_orig_A, Z)
    edges_recon, lengths_recon = build_bonds_pairwise_torch(positions_recon_A, Z)

    # Compare bonds present in both graphs

    def _to_tuple_list(edges_t: torch.Tensor):
        return [
            (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            for i, j in edges_t.tolist()
        ]

    orig_map = {e: float(l) for e, l in zip(_to_tuple_list(edges_orig), lengths_orig.tolist())}
    recon_map = {e: float(l) for e, l in zip(_to_tuple_list(edges_recon), lengths_recon.tolist())}

    orig_keys = set(orig_map.keys())
    recon_keys = set(recon_map.keys())

    common = sorted(orig_keys & recon_keys)
    missed = sorted(orig_keys - recon_keys)
    spurious = sorted(recon_keys - orig_keys)

    diffs = np.array([recon_map[e] - orig_map[e] for e in common], dtype=float)
    abs_diffs = np.abs(diffs)

    print(f"# bonds (orig): {len(orig_keys)}")
    print(f"# bonds (recon): {len(recon_keys)}")
    print(f"# common bonds: {len(common)}")
    print(f"# missed (orig only): {len(missed)}")
    print(f"# spurious (recon only): {len(spurious)}")

    if len(common):
        print(
            "Bond length Δ (recon - orig) Å: mean={:.3f}, median={:.3f}, max|Δ|={:.3f}".format(
                diffs.mean(), np.median(diffs), abs_diffs.max()
            )
        )
        # Show largest 5 discrepancies
        worst_idx = np.argsort(-abs_diffs)[:5]
        print("Top-5 absolute differences (i,j,d_orig,d_recon,Δ):")
        for k in worst_idx:
            i, j = common[k]
            d_o = orig_map[(i, j)]
            d_r = recon_map[(i, j)]
            print(f"  ({i:2d},{j:2d}): {d_o:.3f} → {d_r:.3f}  Δ={d_r - d_o:.3f}")

        # ---- Visualize reconstructed bonds in the volume (lines between atoms) ----
        # Convert reconstructed aligned points (normalized) to voxel coordinates for plotting
        recon_points_vox = ((recon_aligned + 1.0) * (RESOLUTION / 2.0)).astype(np.float32)

        # Build VTK lines for PyVista: each line encoded as [2, i, j]
        if edges_recon.numel() > 0:
            e_np = edges_recon.cpu().numpy()
            lines_all = np.hstack([np.full((e_np.shape[0], 1), 2, dtype=np.int32), e_np.astype(np.int32)]).ravel()

            # Split heavy–heavy vs X–H for clearer visualization
            Z_np = Z.cpu().numpy() if hasattr(Z, 'cpu') else np.asarray(Z)
            is_hh = (Z_np[e_np[:,0]] != 1) & (Z_np[e_np[:,1]] != 1)
            is_xh = (Z_np[e_np[:,0]] == 1) ^ (Z_np[e_np[:,1]] == 1)

            # Heavy–heavy bonds (thicker, warm color)
            if np.any(is_hh):
                hh_edges = e_np[is_hh]
                hh_lines = np.hstack([np.full((hh_edges.shape[0], 1), 2, dtype=np.int32), hh_edges.astype(np.int32)]).ravel()
                hh_poly = pv.PolyData(recon_points_vox, lines=hh_lines)
                plotter.add_mesh(hh_poly, color="#ffcc66", line_width=5, opacity=1.0)

            # X–H bonds (thinner, cool color)
            if np.any(is_xh):
                xh_edges = e_np[is_xh]
                xh_lines = np.hstack([np.full((xh_edges.shape[0], 1), 2, dtype=np.int32), xh_edges.astype(np.int32)]).ravel()
                xh_poly = pv.PolyData(recon_points_vox, lines=xh_lines)
                plotter.add_mesh(xh_poly, color="#66ccff", line_width=3, opacity=1.0)

            # If there are any remaining bonds (e.g., fallback), draw them in neutral color
            rest_mask = ~(is_hh | is_xh)
            if np.any(rest_mask):
                r_edges = e_np[rest_mask]
                r_lines = np.hstack([np.full((r_edges.shape[0], 1), 2, dtype=np.int32), r_edges.astype(np.int32)]).ravel()
                r_poly = pv.PolyData(recon_points_vox, lines=r_lines)
                plotter.add_mesh(r_poly, color="white", line_width=3, opacity=1.0)

        # Optionally, label atom indices for debugging (toggle by setting to True)
        if False:
            point_labels = [str(i) for i in range(recon_points_vox.shape[0])]
            plotter.add_point_labels(recon_points_vox, point_labels, font_size=10, fill_shape=False)

# add detected peaks
plotter.add_points(
    merged_peaks,
    render_points_as_spheres=True,
    point_size=6,
    color="red",
    show_scalar_bar=False,
)

plotter.show()
