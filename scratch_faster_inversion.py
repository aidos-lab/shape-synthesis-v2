import numpy as np
import torch
from dect.directions import generate_uniform_directions
from torch import Tensor

from custom_ect import compute_ect
from src.datasets.single_molecule import get_dataset

# from inversion import filtered_back_projection


#######################################################################
np.random.seed(42)
RESOLUTION = 256
RADIUS = 1.0
SCALE = 500
DEVICE = "cuda"
DTYPE = torch.float16
#######################################################################

#########################################################################################################
#### Reconstruct
#########################################################################################################


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


def calc_idx_fast(theta, grid, resolution):
    heights = torch.inner(theta.to(torch.float32), grid.movedim(0, 3).to(torch.float32))
    idx = ((heights + 1) * resolution / 2).long() + 1
    idx[idx > resolution - 1] = resolution - 1
    return idx


def get_v_slices(resolution, device):
    linspace = torch.linspace(-1, 1, resolution, device=device, dtype=DTYPE)
    xg, yg, zg = torch.meshgrid(
        linspace,
        linspace,
        linspace,
        indexing="ij",
    )
    grid = torch.stack([xg, yg, zg], dim=-1)

    heights = torch.inner(grid.to(dtype=DTYPE), v.to(dtype=DTYPE).T)
    idx = ((heights + 1) * resolution / 2).to(DTYPE).clamp(max=resolution - 1, min=0)
    return idx


@torch.compile
def filtered_back_projection(
    v_idx: Tensor,
    ect: Tensor,
):
    resolution = ect.shape[0]
    recon = torch.zeros(resolution, resolution, resolution, device=DEVICE)
    for idx_slice, ect_slice in zip(v_idx, ect.T):
        recon += ect_slice[idx_slice.long()]
    return recon


v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device=DEVICE)
x, z, to_angstrom = get_dataset()
# Compute the ECT
ect = compute_ect(x.to(DEVICE), v, radius=RADIUS, scale=SCALE, resolution=RESOLUTION)


v_idx = get_v_slices(RESOLUTION, DEVICE)
filtered_back_projection(
    v_idx.to(DEVICE),
    ect.to(DEVICE),
)


with torch.autograd.profiler.profile(use_device="cuda") as prof:
    for _ in range(100):
        filtered_back_projection(
            v_idx,
            ect,
        )

# Print the profiling results, sorted by CUDA time and limited to the top 10 rows
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
