import matplotlib.pyplot as plt
from dect.directions import generate_2d_directions

dataset = QM9(
    root="data",
    force_reload=False,
)


config = DataConfig(
    root="./data",
    raw="./data/raw",
    batch_size=64,
    resolution=64,
)

create_dataset(config, dev=True)

dl, _ = get_dataloaders(config, dev=True)
for (ects,) in dl:
    break


# |%%--%%| <dizt2YSnJE|NFzB0xTiI1>

from torch_geometric.datasets import QM9

# Download the full QM9 dataset.
dataset = QM9(
    root="data",
    force_reload=False,
)

dataset.pos.shape

# |%%--%%| <NFzB0xTiI1|ubT5gwXQm9>

x = dataset.pos
radii = []

for data in dataset:
    radii.append(data.pos.norm(dim=-1).max())


# |%%--%%| <ubT5gwXQm9|7NAroAXTuG>

import torch

r = torch.stack(radii).numpy()

# Max radius is 11.
plt.hist(r)


# |%%--%%| <7NAroAXTuG|9kMzD6tGPx>

ects.shape

# |%%--%%| <9kMzD6tGPx|CDRTY5vPVx>

plt.imshow(ects[0, 2])

# |%%--%%| <CDRTY5vPVx|aS2UcKw4xv>

idx = 12

resolution = 256
v = generate_2d_directions(resolution)
ect = compute_ect_point_cloud(pts, v, radius=1, resolution=resolution, scale=1000)

plt.imshow(ect[idx])


# |%%--%%| <aS2UcKw4xv|GFm1v0X3hP>
plt.scatter(pts[idx, :, 0], pts[idx, :, 1])

# |%%--%%| <GFm1v0X3hP|dLtVCxRUwa>

plt.imshow(img[idx])
