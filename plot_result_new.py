import matplotlib.pyplot as plt
import pyvista as pv
import torch

pts = torch.load("results/recon_pts.pt")
mask = torch.load("results/recon_mask.pt")

pts_orig = torch.load("results/orig_pts.pt")
mask_orig = torch.load("results/orig_mask.pt")

idx = 35

pts_recon = pts[5 * idx : 5 * idx + 5]
mask_recon = mask[5 * idx : 5 * idx + 5]
pts_orig = pts_orig[5 * idx : 5 * idx + 5]
mask_orig = mask_orig[5 * idx : 5 * idx + 5]


plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
for i, color in enumerate(["red", "green", "blue", "lightblue", "cyan"]):
    pts_npy = pts_recon[i][mask_recon[i]].numpy()
    if len(pts_npy) > 0:
        plotter.add_points(
            points=pts_npy,
            render_points_as_spheres=True,
            point_size=10,
            show_scalar_bar=False,
            color=color,
        )
plotter.subplot(0, 1)
for i, color in enumerate(["red", "green", "blue", "lightblue", "cyan"]):
    pts_npy = pts_orig[i][mask_orig[i]].cpu().numpy()
    if len(pts_npy) > 0:
        plotter.add_points(
            points=pts_npy,
            render_points_as_spheres=True,
            point_size=10,
            show_scalar_bar=False,
            color=color,
        )
plotter.link_views()
plotter.show()
# |%%--%%| <H0RB36qHOI|tegxLvC8B6>
i = 0
print(pts_orig[i][mask_orig[i]].cpu().numpy())
print(pts_recon[i][mask_recon[i]].numpy())

# # |%%--%%| <tegxLvC8B6|MQf6q7QzfY>
#
#
# print(orig_batch.shape)
# print(orig_z.shape)
# print(orig_pts.shape)
#
# print(recon_batch.shape)
# print(recon_means.shape)
# print(recon_radii.shape)
#
#
# # |%%--%%| <MQf6q7QzfY|vSS1bnu78K>
#
# pc_idx = 100
#
# x_recon = recon_pts[recon_batch == pc_idx]
# z_recon = recon_z[recon_batch == pc_idx]
# x_orig = orig_pts[orig_batch == pc_idx]
# z_orig = orig_z[orig_batch == pc_idx]
#
#
# plotter = pv.Plotter(shape=(1, 2))
#
# plotter.subplot(0, 0)
# for atom, color in zip(
#     [1, 6, 7, 8, 9], ["red", "green", "blue", "lightblue", "yellow"]
# ):
#     pts = x_recon[z_recon == atom].cpu().numpy()
#     if len(pts) > 0:
#         plotter.add_points(
#             pts,
#             render_points_as_spheres=True,
#             point_size=10,
#             color=color,
#             show_scalar_bar=False,
#         )
#
#
# plotter.subplot(0, 1)
# for atom, color in zip(
#     [1, 6, 7, 8, 9], ["red", "green", "blue", "lightblue", "yellow"]
# ):
#     pts = x_orig[z_orig == atom].cpu().numpy()
#     if len(pts) > 0:
#         plotter.add_points(
#             pts,
#             render_points_as_spheres=True,
#             point_size=10,
#             color=color,
#             show_scalar_bar=False,
#         )
#
#     # plotter.add_points(
#     #     x_orig.cpu().numpy(),
#     #     render_points_as_spheres=True,
#     #     point_size=5,
#     #     color="blue",
#     #     show_scalar_bar=False,
#     # )
# plotter.link_views()
# plotter.show()
#
# # fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
# #
# #
# # for ax, ref_ect, recon_ect in zip(axes.T, ref, recon):
# #     ax[0].imshow(recon_ect.cpu().squeeze().numpy())
# #     ax[0].axis("off")
# #     ax[1].imshow(ref_ect.cpu().squeeze().numpy())
# #     ax[1].axis("off")
# #
# # plt.tight_layout()
# # plt.show()
