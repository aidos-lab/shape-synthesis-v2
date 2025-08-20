import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import torch_geometric
from dect.directions import generate_multiview_directions, generate_uniform_directions
from torch_geometric.datasets import QM9

from custom_ect import compute_ect
from src.datasets.qm9 import DataConfig, get_dataloaders
from src.datasets.transforms import get_transform

# from inversion import filtered_back_projection
from src.inversion.fbp_improved import reconstruct_point_cloud
from src.metrics.molecule import compute_metrics
from src.plotting.recon import plot_reconstruction

# from src.datasets.single_molecule import DataConfig, get_dataset
# from src.datasets.transforms import To3DNormalizedCoords


torch.set_float32_matmul_precision("medium")

#######################################################################
np.random.seed(42)
RESOLUTION = 128  # Abbreviated to R
RADIUS = 1.0  # Abbreviated to r, fixed to 1 for now.
SCALE = 100  # Fixed hyperparameter for now. Is sets the bandwidth for the dirac approximation.
DEVICE = "cuda"  # Device to compute on.
#######################################################################


#########################################################################################################
#### Reconstruct
#########################################################################################################


# v has shape [3, R]
v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu").to(DEVICE)

# # [N,3] where N is the number of atoms in the molecule.
# x, z, to_angstrom = get_dataset()

compute_ect = torch.compile(compute_ect)


# Download the full QM9 dataset.
dataset = QM9(
    root="data/qm9/dev",
    force_reload=False,
)

dl = torch_geometric.loader.DataLoader(dataset, batch_size=128)

# |%%--%%| <A90zD8uDAz|bhLelXxfKp>

correct = 0
too_few = 0
too_many = 0
for batch_idx, batch in enumerate(dl):
    batch.to(DEVICE)
    ects = []
    recon_pts = []
    recon_z = []
    orig_pts = []
    recon_batch = []
    for i in range(len(batch)):
        x = batch[i].pos
        x -= x.mean(axis=0)
        x /= x.norm(dim=-1).max()
        x *= 0.7
        orig_pts.append(x)
        for atom in [1, 6, 7, 8, 9]:
            x_filtered = x[batch[i].z == atom]
            # Compute the ECT.
            ect = compute_ect(
                x_filtered,
                v,
                radius=RADIUS,
                scale=SCALE,
                resolution=RESOLUTION,
            )

            # x_recon is the reconstructed point cloud.
            # The additional tuple is for computing the loss.
            x_recon, _ = reconstruct_point_cloud(
                ect,
                v,
                threshold=0.75,
            )

            recon_pts.append(torch.tensor(x_recon))
            recon_z.append(atom * torch.ones(len(x_recon)))
            recon_batch.append(i * torch.ones(len(x_recon)))

            # print(x.shape, x_recon.shape)
            if len(x_filtered) == len(x_recon):
                correct += 1

            if len(x_filtered) < len(x_recon):
                too_many += 1

            if len(x_filtered) > len(x_recon):
                too_few += 1

            ects.append(ect.cpu().clone())

        torch.save(torch.vstack(recon_pts), f"results/qm9_fpb/recon_pts_{batch_idx}.pt")
        torch.save(torch.hstack(recon_z), f"results/qm9_fpb/recon_z_{batch_idx}.pt")
        torch.save(
            torch.hstack(recon_batch), f"results/qm9_fpb/recon_batch_{batch_idx}.pt"
        )

        torch.save(torch.vstack(orig_pts), f"results/qm9_fpb/orig_pts_{batch_idx}.pt")
        torch.save(batch.batch, f"results/qm9_fpb/orig_batch_{batch_idx}.pt")
        torch.save(batch.z, f"results/qm9_fpb/orig_z_{batch_idx}.pt")

        torch.save(torch.vstack(ects), f"results/qm9_fpb/ects_{batch_idx}.pt")

    print(
        f"Batch: {batch_idx}, Correct: {correct},too_few: {too_few}, too_many {too_many }"
    )
