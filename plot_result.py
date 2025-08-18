import matplotlib.pyplot as plt
import pyvista as pv
import torch

batch_idx = 1


recon_batch = torch.load(f"./results/qm9_fpb/recon_batch_{batch_idx}.pt")
recon_pts = torch.load(f"./results/qm9_fpb/recon_pts_{batch_idx}.pt")
recon_z = torch.load(f"./results/qm9_fpb/recon_z_{batch_idx}.pt")


orig_batch = torch.load(f"./results/qm9_fpb/orig_batch_{batch_idx}.pt")
orig_pts = torch.load(f"./results/qm9_fpb/orig_pts_{batch_idx}.pt")
orig_z = torch.load(f"./results/qm9_fpb/orig_z_{batch_idx}.pt")


# |%%--%%| <ys2uBHun3I|MQf6q7QzfY>


print(orig_batch)
print(recon_batch)


# |%%--%%| <MQf6q7QzfY|vSS1bnu78K>

pc_idx = 121

x_recon = recon_pts[recon_batch == pc_idx]
z_recon = recon_z[recon_batch == pc_idx]
x_orig = orig_pts[orig_batch == pc_idx]
z_orig = orig_z[orig_batch == pc_idx]


plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
for atom, color in zip(
    [1, 6, 7, 8, 9], ["red", "green", "blue", "lightblue", "yellow"]
):
    pts = x_recon[z_recon == atom].cpu().numpy()
    if len(pts) > 0:
        plotter.add_points(
            pts,
            render_points_as_spheres=True,
            point_size=10,
            color=color,
            show_scalar_bar=False,
        )


plotter.subplot(0, 1)
for atom, color in zip(
    [1, 6, 7, 8, 9], ["red", "green", "blue", "lightblue", "yellow"]
):
    pts = x_orig[z_orig == atom].cpu().numpy()
    if len(pts) > 0:
        plotter.add_points(
            pts,
            render_points_as_spheres=True,
            point_size=10,
            color=color,
            show_scalar_bar=False,
        )

    # plotter.add_points(
    #     x_orig.cpu().numpy(),
    #     render_points_as_spheres=True,
    #     point_size=5,
    #     color="blue",
    #     show_scalar_bar=False,
    # )

plotter.show()

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
#
#
# for ax, ref_ect, recon_ect in zip(axes.T, ref, recon):
#     ax[0].imshow(recon_ect.cpu().squeeze().numpy())
#     ax[0].axis("off")
#     ax[1].imshow(ref_ect.cpu().squeeze().numpy())
#     ax[1].axis("off")
#
# plt.tight_layout()
# plt.show()
