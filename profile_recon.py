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
RESOLUTION = 128  # Abbreviated to R
RADIUS = 1.0  # Abbreviated to r, fixed to 1 for now.
SCALE = 200  # Fixed hyperparameter for now. Is sets the bandwidth for the dirac approximation.
DEVICE = "cuda"  # Device to compute on.
#######################################################################


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

    for batch_idx, batch in enumerate(dl):
        batch.to(DEVICE)
        break
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
        ect, batched_recon=batched_recon, width=9, threshold=0.5
    )

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/recon"),
        record_shapes=False,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for batch_idx, batch in enumerate(dl):
            prof.step()
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
                ect, batched_recon=batched_recon, width=9, threshold=0.5
            )

            if batch_idx == 5:
                break
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for vq vae training")
    parser.add_argument(
        "--compile", default=False, action="store_true", help="Compile all the models"
    )
    parser.add_argument(
        "--dev", default=False, action="store_true", help="Run a small subset."
    )
    args = parser.parse_args()

    main(args, compute_ect_channels, reconstruct_point_cloud)
