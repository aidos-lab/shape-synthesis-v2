import torch
from dect.directions import generate_uniform_directions
from torch import Tensor


# Needs to be changed to dirac deltas.
def compute_ect(
    x: Tensor,
    v: Tensor,
    resolution: int,
    radius: float = 1.0,
    scale: int = 500,
) -> Tensor:
    nh = x @ v.to(x.device)
    lin = torch.linspace(-radius, radius, resolution, device=x.device).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh)) * (
        1 - torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    )
    ecc = ecc.sum(axis=1)
    return ecc
