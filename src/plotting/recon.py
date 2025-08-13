import numpy as np
import pyvista as pv


def plot_reconstruction(
    common,
    diffs,
    abs_diffs,
    edges_recon,
    recon_map,
    recon_aligned,
    orig_map,
    Z,
    recon_plot,
    resolution,
    merged_peaks,
):
    # Create a PyVista grid and plot the density
    plotter = pv.Plotter()
    plotter.add_volume(
        recon_plot.cpu().numpy(),
        cmap="viridis",
        opacity="linear",
    )
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
        recon_points_vox = ((recon_aligned + 1.0) * (resolution / 2.0)).astype(
            np.float32
        )

        # Build VTK lines for PyVista: each line encoded as [2, i, j]
        if edges_recon.numel() > 0:
            e_np = edges_recon.cpu().numpy()
            lines_all = np.hstack(
                [np.full((e_np.shape[0], 1), 2, dtype=np.int32), e_np.astype(np.int32)]
            ).ravel()

            # Split heavy–heavy vs X–H for clearer visualization
            Z_np = Z.cpu().numpy() if hasattr(Z, "cpu") else np.asarray(Z)
            is_hh = (Z_np[e_np[:, 0]] != 1) & (Z_np[e_np[:, 1]] != 1)
            is_xh = (Z_np[e_np[:, 0]] == 1) ^ (Z_np[e_np[:, 1]] == 1)

            # Heavy–heavy bonds (thicker, warm color)
            if np.any(is_hh):
                hh_edges = e_np[is_hh]
                hh_lines = np.hstack(
                    [
                        np.full((hh_edges.shape[0], 1), 2, dtype=np.int32),
                        hh_edges.astype(np.int32),
                    ]
                ).ravel()
                hh_poly = pv.PolyData(recon_points_vox, lines=hh_lines)
                plotter.add_mesh(hh_poly, color="#ffcc66", line_width=5, opacity=1.0)

            # X–H bonds (thinner, cool color)
            if np.any(is_xh):
                xh_edges = e_np[is_xh]
                xh_lines = np.hstack(
                    [
                        np.full((xh_edges.shape[0], 1), 2, dtype=np.int32),
                        xh_edges.astype(np.int32),
                    ]
                ).ravel()
                xh_poly = pv.PolyData(recon_points_vox, lines=xh_lines)
                plotter.add_mesh(xh_poly, color="#66ccff", line_width=3, opacity=1.0)

            # If there are any remaining bonds (e.g., fallback), draw them in neutral color
            rest_mask = ~(is_hh | is_xh)
            if np.any(rest_mask):
                r_edges = e_np[rest_mask]
                r_lines = np.hstack(
                    [
                        np.full((r_edges.shape[0], 1), 2, dtype=np.int32),
                        r_edges.astype(np.int32),
                    ]
                ).ravel()
                r_poly = pv.PolyData(recon_points_vox, lines=r_lines)
                plotter.add_mesh(r_poly, color="white", line_width=3, opacity=1.0)

        # Optionally, label atom indices for debugging (toggle by setting to True)
        if False:
            point_labels = [str(i) for i in range(recon_points_vox.shape[0])]
            plotter.add_point_labels(
                recon_points_vox, point_labels, font_size=10, fill_shape=False
            )

    # add detected peaks
    plotter.add_points(
        merged_peaks.astype(np.float32),
        render_points_as_spheres=True,
        point_size=6,
        color="red",
        show_scalar_bar=False,
    )

    plotter.show()
