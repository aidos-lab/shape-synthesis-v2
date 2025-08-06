import torch
from dect.directions import generate_uniform_directions


def compute_ect(x, v, radius, resolution, scale):
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh)) * (
        1 - torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    )
    ecc = ecc.sum(axis=1)
    return ecc
