import matplotlib.pyplot as plt
import pyvista as pv
import torch
import torch_geometric
from torch_geometric.utils import to_dense_batch

from torch_geometric.datasets import QM9


# |%%--%%| <LAVOKcKo6t|ol5RGVrqeR>

# Download the full QM9 dataset.
dataset = QM9(
    root="data",
    force_reload=False,
)


recon_pts = torch.load("./results/recon_pts.pt")
recon_z = torch.load("./results/recon_z.pt")
orig_pts = torch.load("./results/orig_pts.pt")
orig_z = torch.load("./results/orig_z.pt")


orig_z, idxs_orig = torch.sort(orig_z)
batched_orig_pts, _ = to_dense_batch(
    orig_pts[idxs_orig], orig_z, max_num_nodes=30, fill_value=11
)
recon_z, idxs_recon = torch.sort(recon_z)
batched_recon_pts, _ = to_dense_batch(
    recon_pts[idxs_recon], recon_z, max_num_nodes=30, fill_value=11
)

pts_res = []
z_res = []
for batch_idx in range(batched_recon_pts.shape[0] // 5):
    pts_tmp = []
    z_tmp = []
    for atom, color in zip(
        [0, 1, 2, 3, 4], ["red", "green", "blue", "lightblue", "yellow"]
    ):
        pts = batched_recon_pts[5 * batch_idx + atom].cpu()
        radius = pts.norm(dim=-1)
        pts = pts[radius < 11]
        z_tmp.append(atom * torch.ones(len(pts)))
        pts_tmp.append(pts)
    pts_res.append(torch.vstack(pts_tmp))
    z_res.append(torch.hstack(z_tmp))

torch.save(pts_res, "pts_list.pt")
torch.save(z_res, "z_list.pt")

pts_res = []
z_res = []
for batch_idx in range(batched_orig_pts.shape[0] // 5):
    pts_tmp = []
    z_tmp = []
    for atom, color in zip(
        [0, 1, 2, 3, 4], ["red", "green", "blue", "lightblue", "yellow"]
    ):
        pts = batched_orig_pts[5 * batch_idx + atom].cpu()
        radius = pts.norm(dim=-1)
        pts = pts[radius < 11]
        z_tmp.append(atom * torch.ones(len(pts)))
        pts_tmp.append(pts)
    pts_res.append(torch.vstack(pts_tmp))
    z_res.append(torch.hstack(z_tmp))

torch.save(pts_res, "orig_pts_list.pt")
torch.save(z_res, "orig_z_list.pt")

# |%%--%%| <ol5RGVrqeR|lHXH78APyQ>

# for i in range(100):
#     print(pts_res[i].shape[0])


# |%%--%%| <lHXH78APyQ|pUzzxloZfW>

density = torch.load("results/density.pt")

plotter = pv.Plotter()


plotter.add_volume(density[0].cpu().squeeze().numpy(), opacity="sigmoid_6")
plotter.show()


# # |%%--%%| <pUzzxloZfW|H0RB36qHOI>

batch_idx = 969
plotter = pv.Plotter(shape=(1, 3))
density = torch.load("results/density.pt")
plotter.subplot(0, 0)
plotter.add_volume(density.squeeze()[0].cpu().numpy())

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


# plotter.subplot(0, 1)
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
# plotter.subplot(0, 2)
for atom, color in zip(
    [1, 6, 7, 8, 9],
    ["red", "green", "blue", "lightblue", "yellow"],
):
    pts = dataset[batch_idx].pos.numpy()
    z = dataset[batch_idx].z
    print(pts)
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
