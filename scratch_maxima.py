import functools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from dect.directions import generate_uniform_directions
from torch import Tensor

from src.datasets.single_molecule import get_dataset

torch.set_float32_matmul_precision("medium")

#######################################################################
np.random.seed(42)
RESOLUTION = 64  # Abbreviated to R
RADIUS = 1.0  # Abbreviated to r, fixed to 1 for now.
SCALE = 100  # Fixed hyperparameter for now. Is sets the bandwidth for the dirac approximation.
DEVICE = "cuda"  # Device to compute on.
#######################################################################


def fetch_maxima(window_maxima):
    resolution = 128
    out = torch.ones(1, 30, 3)
    out[0, :, :] = torch.hstack([x_idx, y_idx, z_idx])
    return out


def peak_finder_3d(img, width=9):
    resolution = img.shape[-1]
    window_maxima = torch.nn.functional.max_pool3d(
        input=img,
        kernel_size=width,
        stride=1,
        padding=width // 2,
    ).squeeze()

    candidates = (img == window_maxima) & (img > 0.7 * img.max())
    idxs = candidates.nonzero().squeeze()
    print(window_maxima.shape)
    print(idxs)

    out = torch.ones(size=(len(window_maxima), 30, 3))
    return out.to(torch.float64)


def print_memory(name, x):
    mem = sys.getsizeof(x.untyped_storage()) * 0.000001
    print(f"Name: {name} Memory {mem:.2f} MB, Shape: {list(x.shape)} Dtype: {x.dtype}")


def recon(ect, idx):
    return torch.gather(ect, index=idx, dim=0).sum(axis=-1)


def compute_ect_channels(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    channels: Tensor,
    index: Tensor | None = None,
    max_channels: int | None = None,
    normalize: bool = False,
):
    """
    Allows for channels within the point cloud to separated in different
    ECT's.

    Input is a point cloud of size (B*num_point_per_pc,num_features) with an additional feature vector with the
    channel number for each point and the output is ECT for shape [B,num_channels,num_thetas,resolution]
    """

    # Ensure that the scale is in the right device
    scale = torch.tensor([scale], device=x.device)

    # Compute maximum channels.
    if max_channels is None:
        max_channels = int(channels.max()) + 1

    if index is not None:
        batch_len = int(index.max() + 1)
    else:
        batch_len = 1
        index = torch.zeros(
            size=(len(x),),
            dtype=torch.int32,
            device=x.device,
        )

    # Fix the index to interleave with the channel info.
    index = max_channels * index + channels

    # v is of shape [ambient_dimension, num_thetas]
    num_thetas = v.shape[1]

    out_shape = (resolution, batch_len * max_channels, num_thetas)

    # Node heights have shape [num_points, num_directions]
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution, device=x.device).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ecc = ecc * (1 - ecc)
    output = torch.zeros(
        size=out_shape,
        device=nh.device,
    )

    output.index_add_(1, index, ecc)
    ect = output.movedim(0, 1)

    if normalize:
        ect = ect / torch.amax(ect, dim=(-2, -3))

    # Returns the ect as [batch_len, num_thetas, resolution]
    return ect.reshape(-1, max_channels, resolution, num_thetas)


######################################################################
######################################################################
######################################################################

# x, z, to_angstrom = get_dataset()
# z[z == 1] = 0
# z[z == 6] = 1
# z[z == 7] = 2
# z[z == 8] = 3
# z[z == 9] = 4
#
# x = x.to(DEVICE)
# z = z.to(DEVICE)
# max_channels = 5
# print(x.norm(dim=-1).max())

x = (2 * torch.rand(size=(9, 3)).to(DEVICE) - 1) / 3
z = torch.zeros(size=(9,), dtype=torch.int64).to(DEVICE)
max_channels = 5


v = generate_uniform_directions(RESOLUTION, d=3, seed=2015, device=DEVICE).to(DEVICE)
ect = compute_ect_channels(
    x,
    v,
    radius=RADIUS,
    scale=SCALE,
    resolution=RESOLUTION,
    channels=z,
    index=torch.zeros(len(x), device=DEVICE).long(),
    max_channels=max_channels,
).to(DEVICE)


linspace = torch.linspace(-1, 1, RESOLUTION, device=ect.device)
xg, yg, zg = torch.meshgrid(
    linspace,
    linspace,
    linspace,
    indexing="ij",
)

grid = torch.stack([xg, yg, zg]).view(3, -1)
nh = grid.T @ v
idx = ((nh + 1) * RESOLUTION / 2).to(torch.int64).clamp(min=0, max=RESOLUTION - 1)

recon = functools.partial(recon, idx=idx)
batched_recon = torch.vmap(recon)
ect = ect.squeeze()
res = batched_recon(ect)
res = res / res.max()
res[res < 0.7] = 0


atom_type = 0
x_filtered = x[z == atom_type]

res_plot = res[atom_type]

for i in range(10):
    print(i)
    coords = peak_finder_3d(
        res.view(-1, 5, RESOLUTION, RESOLUTION, RESOLUTION), width=5
    )


# |%%--%%| <JEQRFwyxW5|ewmntleRrg>


# |%%--%%| <ewmntleRrg|noV6FwbjMy>


linspace = linspace.cpu()
rad_mask = linspace[coords.to(torch.int64).cpu()].norm(dim=-1) < 1
coords = coords[rad_mask].cpu().float().numpy()


print("Recon shape", coords.shape)
print("true shape", x.shape)


x_plot = (1 + x_filtered) / 2 * RESOLUTION
plotter = pv.Plotter()
plotter.add_points(x_plot.cpu().numpy(), render_points_as_spheres=True, point_size=10)

plotter.add_points(coords, color="red", render_points_as_spheres=True, point_size=15)

# plotter.add_points(x_filtered.numpy(), render_points_as_spheres=True, color="red")
plotter.add_volume(
    res[atom_type].view(RESOLUTION, RESOLUTION, RESOLUTION).cpu().numpy(),
    opacity="sigmoid_5",
)
plotter.show()
