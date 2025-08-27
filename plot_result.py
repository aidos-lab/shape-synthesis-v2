import matplotlib.pyplot as plt
import pyvista as pv
import torch
import torch_geometric
from torch_geometric.utils import to_dense_batch

from torch_geometric.datasets import QM9

# Download the full QM9 dataset.
dataset = QM9(
    root="data",
    force_reload=False,
)


recon_pts = torch.load("./results/recon_pts_final.pt")
recon_z = torch.load("./results/recon_z_final.pt")
orig_pts = torch.load("./results/orig_pts_final.pt")
orig_z = torch.load("./results/orig_z_final.pt")

print(orig_z)
print(orig_pts)
print(recon_z)
print(recon_pts)


orig_z, idxs_orig = torch.sort(orig_z)
batched_orig_pts, _ = to_dense_batch(
    orig_pts[idxs_orig], orig_z, max_num_nodes=30, fill_value=11
)
recon_z, idxs_recon = torch.sort(recon_z)
batched_recon_pts, _ = to_dense_batch(
    recon_pts[idxs_recon], recon_z, max_num_nodes=30, fill_value=11
)
print(batched_recon_pts.shape)

# |%%--%%| <KuCf0w9t4Q|LAVOKcKo6t>

# density = torch.load("results/density.pt")
# density.shape
# plotter = pv.Plotter()
# plotter.add_volume(density[0].cpu().squeeze().numpy(), opacity="sigmoid_6")
# plotter.show()


# # |%%--%%| <LAVOKcKo6t|H0RB36qHOI>


print(batched_recon_pts.shape)
batch_idx = 8
plotter = pv.Plotter(shape=(1, 3))


plotter.subplot(0, 0)
for atom, color in zip(
    [0, 1, 2, 3, 4], ["red", "green", "blue", "lightblue", "yellow"]
):
    pts = batched_orig_pts[5 * batch_idx + atom].cpu()
    radius = pts.norm(dim=-1)
    pts = pts[radius < 11]
    print(pts)
    if len(pts) > 0:
        plotter.add_points(
            pts.numpy(),
            render_points_as_spheres=True,
            point_size=10,
            color=color,
            show_scalar_bar=False,
        )

plotter.subplot(0, 1)
for atom, color in zip(
    [0, 1, 2, 3, 4], ["red", "green", "blue", "lightblue", "yellow"]
):
    pts = batched_recon_pts[5 * batch_idx + atom].cpu()
    radius = pts.norm(dim=-1)
    pts = pts[radius < 11]
    print(pts)
    if len(pts) > 0:
        plotter.add_points(
            pts.numpy(),
            render_points_as_spheres=True,
            point_size=10,
            color=color,
            show_scalar_bar=False,
        )
plotter.subplot(0, 2)
for atom, color in zip(
    [1, 6, 7, 8, 9],
    ["red", "green", "blue", "lightblue", "yellow"],
):
    pts = dataset[batch_idx].pos.numpy()
    z = dataset[batch_idx].z
    print(z)
    if len(pts[z == atom]) > 0:
        plotter.add_points(
            pts[z == atom],
            render_points_as_spheres=True,
            point_size=10,
            color=color,
            show_scalar_bar=False,
        )

plotter.link_views()
plotter.show()
