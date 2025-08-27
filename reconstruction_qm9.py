import argparse
import functools
import os

import numpy as np
import torch
import torch_geometric
from dect.directions import generate_uniform_directions
from torch_geometric.datasets import QM9

from custom_ect import compute_ect_channels
from src.inversion.fbp import gather_indices, get_grid, reconstruct_point_cloud

torch.set_float32_matmul_precision("medium")

#######################################################################
np.random.seed(42)
RESOLUTION = 128  # Abbreviated to R
RADIUS = 1.0  # Abbreviated to r, fixed to 1 for now.
SCALE = 100  # Fixed hyperparameter for now. Is sets the bandwidth for the dirac approximation.
DEVICE = "cuda"  # Device to compute on.
WIDTH = 3
THRESHOLD = 0.0
GLOBAL_SCALE = 6
#######################################################################

#########################################################################################################
#### Reconstruct
#########################################################################################################


def main(args, compute_ect_channels, reconstruct_point_cloud):
    # v has shape [3, R]
    v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu").to(DEVICE)

    idx = get_grid(resolution=RESOLUTION, v=v)
    recon = functools.partial(gather_indices, idx=idx)
    batched_recon = torch.vmap(recon)

    # Bookkeeping for the dev and prod.
    output_folder = "results/qm9_fbp_dev"
    if args.dev:
        output_folder += "_dev"

    # Ensure folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Download the full QM9 dataset.
    dataset = QM9(
        root="data",
        force_reload=False,
    )

    dl = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    # Start reconstruction.
    recon_pts = []
    recon_z = []
    orig_pts = []
    orig_z = []

    for batch_idx, batch in enumerate(dl):
        if batch_idx % 100 == 0:
            print(batch_idx)
        batch.to(DEVICE)
        z = batch.z.clone()
        z[z == 1] = 0
        z[z == 6] = 1
        z[z == 7] = 2
        z[z == 8] = 3
        z[z == 9] = 4

        m = batch.pos.mean(axis=0)
        pos = (batch.pos - m) / GLOBAL_SCALE
        ect = (
            compute_ect_channels(
                pos,
                v,
                radius=RADIUS,
                scale=SCALE,
                resolution=RESOLUTION,
                channels=z,
                index=batch.batch,
                max_channels=5,
            )
            .to(DEVICE)
            .view(-1, RESOLUTION, RESOLUTION)
        )

        r_pts, r_z, _ = reconstruct_point_cloud(
            ect,
            batched_recon=batched_recon,
            width=WIDTH,
            threshold=THRESHOLD,
        )

        recon_pts.append((r_pts * GLOBAL_SCALE + m).cpu())
        recon_z.append((r_z + 5 * len(batch) * batch_idx).cpu())

        orig_z.append((z + 5 * len(batch) * batch_idx).cpu())
        orig_pts.append(batch.pos.cpu())

        if args.dev and batch_idx == 130:
            break

    torch.save(torch.vstack(recon_pts), "results/recon_pts.pt")
    torch.save(torch.hstack(recon_z), "results/recon_z.pt")
    torch.save(torch.vstack(orig_pts), "results/orig_pts.pt")
    torch.save(torch.hstack(orig_z), "results/orig_z.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for vq vae training")
    parser.add_argument(
        "--compile", default=False, action="store_true", help="Compile all the models"
    )
    parser.add_argument(
        "--dev", default=False, action="store_true", help="Run a small subset."
    )
    args = parser.parse_args()

    if args.compile:
        compute_ect_channels = torch.compile(compute_ect_channels)
        reconstruct_point_cloud = torch.compile(
            reconstruct_point_cloud,
        )

    main(args, compute_ect_channels, reconstruct_point_cloud)
