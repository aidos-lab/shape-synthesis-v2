import torch
from torch import Tensor

# Settings
RESOLUTION = 256
RADIUS = 1.0
SCALE = 256


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


def filtered_back_projection(
    v: Tensor,
    ect: Tensor,
    resolution: int,
    normalized: bool = True,
    threshold: float = 0.0,
) -> Tensor:
    linspace = torch.linspace(-1, 1, resolution)
    xg, yg, zg = torch.meshgrid(
        linspace,
        linspace,
        linspace,
        indexing="ij",
    )

    recon = torch.zeros(size=(resolution, resolution, resolution))

    i = 0
    for theta, slice in zip(v.T, ect.T):
        i += 1
        idx = calc_idx(theta, xg, yg, zg, resolution)
        reps = slice[idx]
        recon += reps

    if normalized:
        recon /= recon.max()
        recon[recon < threshold] = 0.0
    elif not normalized and threshold > 0.0:
        raise Warning(
            "Setting a threshold is not used when not normalizing the density"
        )
    return recon
