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
