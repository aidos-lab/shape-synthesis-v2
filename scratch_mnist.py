import matplotlib.pyplot as plt
import torch
from dect.directions import generate_2d_directions
from skimage.morphology import skeletonize
from torch import Tensor

from src.datasets.mnist import DataConfig, create_dataset, get_dataloaders


def compute_ect_point_cloud(
    x: Tensor,
    v: Tensor,
    radius: float,
    resolution: int,
    scale: float,
    normalize: bool = False,
) -> Tensor:

    # ensure that the scale is in the right device
    scale = torch.tensor([scale], device=x.device)

    lin = torch.linspace(
        start=-radius, end=radius, steps=resolution, device=x.device
    ).view(-1, 1, 1)
    nh = (x @ v).unsqueeze(1)
    nh[nh.isnan()] = 1 * torch.inf
    nh[nh.isinf()] = 1 * torch.inf
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ect = torch.sum(ecc, dim=2)
    return ect


config = DataConfig(
    root="./data",
    raw="./data/raw",
    num_pts=256,
    module="datasets.mnist",
    batch_size=64,
    skeletonize=True,
)

create_dataset(config, dev=True)

dl, _ = get_dataloaders(config, dev=True)
for pts, img in dl:
    break


# |%%--%%| <dizt2YSnJE|aS2UcKw4xv>

idx = 12

resolution = 256
v = generate_2d_directions(resolution)
ect = compute_ect_point_cloud(pts, v, radius=1, resolution=resolution, scale=1000)

plt.imshow(ect[idx])


# |%%--%%| <aS2UcKw4xv|GFm1v0X3hP>
plt.scatter(pts[idx, :, 0], pts[idx, :, 1])

# |%%--%%| <GFm1v0X3hP|dLtVCxRUwa>

plt.imshow(img[idx])
