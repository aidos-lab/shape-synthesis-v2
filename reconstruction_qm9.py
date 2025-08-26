import argparse
import functools
import os

import numpy as np
import torch
import torch_geometric
from dect.directions import generate_uniform_directions
from torch.profiler import ProfilerActivity, profile
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_batch

from custom_ect import compute_ect_channels
from src.inversion.fbp_improved import gather_indices, get_grid, reconstruct_point_cloud

torch.set_float32_matmul_precision("medium")

#######################################################################
np.random.seed(42)
RESOLUTION = 64  # Abbreviated to R
RADIUS = 1.0  # Abbreviated to r, fixed to 1 for now.
SCALE = 200  # Fixed hyperparameter for now. Is sets the bandwidth for the dirac approximation.
DEVICE = "cuda"  # Device to compute on.
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

    dl = torch_geometric.loader.DataLoader(dataset, batch_size=4)

    # Start reconstruction.
    recon_pts = []
    recon_mask = []
    orig_pts = []
    orig_mask = []

    for batch_idx, batch in enumerate(dl):
        print(batch_idx)
        batch.to(DEVICE)
        z = batch.z
        z[z == 1] = 0
        z[z == 6] = 1
        z[z == 7] = 2
        z[z == 8] = 3
        z[z == 9] = 4

        ect = (
            compute_ect_channels(
                batch.pos / 11,
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

        pts, mask = reconstruct_point_cloud(
            ect, batched_recon=batched_recon, width=3, threshold=0.5
        )
        # recon_pts.append(pts * 11)
        # recon_mask.append(mask)
        #
        # b = batch.batch * 5 + z
        # b = torch.hstack([b, torch.tensor(19).to(DEVICE)])
        # pos = torch.vstack([batch.pos, torch.tensor([11, 11, 11]).to(DEVICE)])
        #
        # values, indices = torch.sort(b)
        #
        # p, m = to_dense_batch(pos[indices], values, max_num_nodes=30)
        #
        # m = (p.norm(dim=-1) < 11) & m
        #
        # orig_mask.append(m)
        # orig_pts.append(p)
        #
        # if args.dev and batch_idx == 100:
        #     break

    # torch.save(torch.vstack(recon_pts), f"results/recon_pts.pt")
    # torch.save(torch.vstack(recon_mask), f"results/recon_mask.pt")
    # torch.save(torch.vstack(orig_mask), f"results/orig_mask.pt")
    # torch.save(torch.vstack(orig_pts), f"results/orig_pts.pt")


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
