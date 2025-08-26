import functools
import sys

import torch
from dect.directions import generate_uniform_directions


def peak_finder_3d(img, width=9):
    resolution = img.shape[-1]
    window_maxima = torch.nn.functional.max_pool3d_with_indices(
        input=img,
        kernel_size=width,
        stride=1,
        padding=width // 2,
    )[1].squeeze()
    out = torch.zeros(size=(len(window_maxima), 30, 3), device=img.device)
    for i, wm in enumerate(window_maxima):
        wm = wm.ravel()
        candidates = torch.unique(wm, sorted=False, return_inverse=False)
        nice_peaks = candidates[(wm[candidates] == candidates).nonzero()]
        res = torch.hstack(
            [
                (nice_peaks // resolution) // resolution,
                (nice_peaks // resolution) % resolution,
                nice_peaks % resolution,
            ]
        )
        out[i, : len(res), :] = res

    return out


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
    coords = peak_finder_3d(res_plot, width=width)
    linspace = torch.linspace(-1, 1, resolution, device=ect.device)
    pts = linspace[coords.to(torch.int64)]
    rad_mask = pts.norm(dim=-1) < 1
    return pts, rad_mask
