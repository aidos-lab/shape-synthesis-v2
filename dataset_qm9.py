import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import QM9

# Download the full QM9 dataset.
dataset = QM9(
    root="data",
    force_reload=False,
)

m = []
for i in range(len(dataset)):
    x = dataset[i].pos
    m.append(x - x.mean(axis=0))

m = torch.vstack(m)

# |%%--%%| <8zGKyynUTX|QYoMCnrXGo>

# r = m.norm(dim=-1)
# print(r.max())
