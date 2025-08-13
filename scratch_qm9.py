from src.datasets.qm9 import get_dataloaders, DataConfig
from torch_geometric.datasets import QM9
import matplotlib.pyplot as plt

from src.datasets.transforms import To3DNormalizedCoords
from src.datasets.transforms import get_transform

config = DataConfig(
    root="./data",
    raw="./data/raw",
    batch_size=64,
)
train_dl, _ = get_dataloaders(config, dev=True)


tr = get_transform()

for batch in train_dl:
    imgs = tr(x=batch.pos.cuda(), index=batch.batch.cuda())
    break

plt.imshow(imgs[0].cpu().numpy())
plt.show()
