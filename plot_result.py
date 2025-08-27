import matplotlib.pyplot as plt
import pyvista as pv
import torch
import torch_geometric
from torch_geometric.utils import to_dense_batch

recon_pts = torch.load(f"./results/recon_pts.pt")
recon_z = torch.load(f"./results/recon_z.pt")
orig_pts = torch.load(f"./results/orig_pts.pt")
orig_z = torch.load(f"./results/orig_z.pt")

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

density = torch.load("results/density.pt")
density.shape
plotter = pv.Plotter()
plotter.add_volume(density[0].cpu().squeeze().numpy(), opacity="sigmoid_6")
plotter.show()


# # |%%--%%| <LAVOKcKo6t|H0RB36qHOI>

print(batched_recon_pts.shape)
batch_idx = 1
plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
for atom, color in zip(
    [0, 1, 2, 3, 4], ["red", "green", "blue", "lightblue", "yellow"]
):
    pts = batched_orig_pts[batch_idx + atom].cpu()
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
    pts = batched_recon_pts[batch_idx + atom].cpu()
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
plotter.link_views()
plotter.show()
