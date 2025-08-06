import pyvista as pv
import torch
from torch_geometric.datasets import TUDataset


class To3DNormalizedCoords:
    def __call__(self, data):
        x = data.x[:, -3:]
        x -= x.mean(axis=0)
        x /= x.norm(dim=-1).max()
        return data


dataset = TUDataset(
    root="./data",
    name="QM9",
    use_node_attr=True,
    transform=To3DNormalizedCoords(),
)
