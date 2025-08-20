import matplotlib.pyplot as plt
from dect.directions import generate_2d_directions

from src.datasets.mnist import DataConfig, create_dataset, get_dataloaders

config = DataConfig(
    root="./data",
    raw="./data/raw",
    num_pts=256,
    module="datasets.mnist",
    batch_size=64,
    skeletonize=True,
)

# create_dataset(config, dev=True)

dl, _ = get_dataloaders(config, dev=True)
for pts, img in dl:
    break


# |%%--%%| <dizt2YSnJE|CDRTY5vPVx>

pts.shape

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
