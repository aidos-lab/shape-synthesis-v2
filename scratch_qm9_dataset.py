%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
from dect.directions import generate_2d_directions

from src.datasets.qm9 import DataConfig, create_dataset, get_dataloaders

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


#|%%--%%| <dizt2YSnJE|9kMzD6tGPx>

ects.shape

# |%%--%%| <9kMzD6tGPx|CDRTY5vPVx>

plt.imshow(ects[0,2])

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
