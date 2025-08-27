import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import QM9


# |%%--%%| <8zGKyynUTX|162BA2XY57>

# Download the full QM9 dataset.
dataset = QM9(
    root="data",
    force_reload=False,
)

m = []
r = []
for i in range(len(dataset)):
    x = dataset[i].pos
    x_n = x - x.mean(axis=0)
    m.append(x - x.mean(axis=0))
    r.append(x_n.norm(dim=-1).max())

m = torch.vstack(m)
r = torch.vstack(r)


# |%%--%%| <162BA2XY57|QYoMCnrXGo>

# r = m.norm(dim=-1)
print(r.max())
print(r.min())
