import functools
import sys

import torch
from dect.directions import generate_uniform_directions


def peak_finder_3d(img, width=9):
    resolution = img.shape[-1]
    img = img.squeeze()

    window_maxima = torch.nn.functional.max_pool3d(
        input=img,
        kernel_size=width,
        stride=1,
        padding=width // 2,
    ).squeeze()

    candidates = (img == window_maxima) & (img > 0.7 * img.max())
    idxs = candidates.nonzero().squeeze()
    return idxs[:, 1:], idxs[:, 0]


def gather_indices(ect, idx):
    return torch.gather(ect, index=idx, dim=0).sum(axis=-1)


@functools.lru_cache(maxsize=None)
def get_grid(resolution, v):
    print("COmputeing")
    linspace = torch.linspace(-1, 1, resolution, device=v.device)
    xg, yg, zg = torch.meshgrid(
        linspace,
        linspace,
        linspace,
        indexing="ij",
    )

    grid = torch.stack([xg, yg, zg]).view(3, -1)
    nh = grid.T @ v
    idx = ((nh + 1) * resolution / 2).to(torch.int64).clamp(min=0, max=resolution - 1)
    return idx


def reconstruct_point_cloud(ect, threshold, width, batched_recon):
    resolution = ect.shape[-1]
    ect = ect.squeeze()
    res = batched_recon(ect)
    res = res / res.max()
    res[res < threshold] = 0

    res_plot = res.view(-1, 1, resolution, resolution, resolution)
    coords, batch_idx = peak_finder_3d(res_plot, width=width)
    linspace = torch.linspace(-1, 1, resolution, device=ect.device)
    pts = linspace[coords.to(torch.int64)]
    return pts, batch_idx, res_plot
