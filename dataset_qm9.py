import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import QM9

from src.datasets.qm9 import DataConfig, create_dataset, get_dataloaders

# |%%--%%| <8zGKyynUTX|s1pClpgs2y>

config = DataConfig(
    root="./data",
    raw="./data/raw",
    batch_size=64,
    resolution=64,
    use_diracs=False,
)

create_dataset(config, dev=True, force_reload=False)
dl, _ = get_dataloaders(config, dev=True)

for (ect,) in dl:
    break


# |%%--%%| <s1pClpgs2y|efee6nBAMA>

plt.imshow(ect[9].squeeze()[2].numpy())
#
# # |%%--%%| <efee6nBAMA|162BA2XY57>
#
# # Download the full QM9 dataset.
# dataset = QM9(
#     root="data",
#     force_reload=False,
# )
#
# m = []
# r = []
# for i in range(len(dataset)):
#     x = dataset[i].pos
#     x_n = x - x.mean(axis=0)
#     m.append(x - x.mean(axis=0))
#     r.append(x_n.norm(dim=-1).max())
#
# m = torch.vstack(m)
# r = torch.vstack(r)
#
#
# # |%%--%%| <162BA2XY57|QYoMCnrXGo>
#
# # r = m.norm(dim=-1)
# print(r.max())
# print(r.min())
