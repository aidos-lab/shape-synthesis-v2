import numpy as np
import torch
from dect.directions import generate_uniform_directions
from torch import Tensor

# Settings
RESOLUTION = 128
RADIUS = 1.0
SCALE = 256


def compute_ect(x, v, radius, resolution, scale):
    nh = x @ v
    lin = torch.linspace(-radius, radius, resolution).view(-1, 1, 1)
    # ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh)) * (
    #     1 - torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    # )
    # ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    ecc = torch.heaviside(scale * torch.sub(lin, nh), torch.tensor(0.0))
    ecc = ecc.sum(axis=1)
    return ecc


def calc_idx(theta, grid, resolution):
    R = RESOLUTION - 1
    heights = torch.inner(theta.to(torch.float32), grid.movedim(0, 3).to(torch.float32))
    idx = ((heights + 1) * RESOLUTION / 2).long() + 1
    idx[idx > R] = R
    return idx


def filtered_back_projection(
    v: Tensor,
    ect: Tensor,
    resolution: int,
    normalized: bool = True,
    threshold: float = 0.0,
) -> Tensor:
    linspace = torch.linspace(-1, 1, resolution)
    xg, yg, zg = torch.meshgrid(
        linspace,
        linspace,
        linspace,
        indexing="ij",
    )
    grid = torch.tensor(np.stack([xg, yg, zg]))
    recon = torch.zeros(size=(resolution, resolution, resolution))

    i = 0
    for theta, slice in zip(v.T, ect.T):
        i += 1
        idx = calc_idx(theta, grid, resolution)
        reps = slice[idx]
        recon += reps

    if normalized:
        recon /= recon.max()
        recon[recon < threshold] = 0.0
    elif not normalized and threshold > 0.0:
        raise Warning(
            "Setting a threshold is not used when not normalizing the density"
        )
    return recon


#
# recon = torch.tensor(
#     np.zeros(shape=(RESOLUTION, RESOLUTION, RESOLUTION), dtype=np.float32)
# )

# v = generate_uniform_directions(RESOLUTION, d=3, seed=2025, device="cpu")

# ect = compute_ect(x, v, radius=RADIUS, resolution=RESOLUTION, scale=SCALE)
#
# import matplotlib.pyplot as plt
#
# plt.imshow(ect.numpy())
#
# # |%%--%%| <6bapWgakqP|jVdKYsCSz3>
#
# plt.hist(ect.ravel().numpy(), bins=200)
#
#
# # |%%--%%| <jVdKYsCSz3|iEsZ22ZKTZ>
#
#
# # density = filtered_back_projection(
# #     v.cuda(),
# #     ect.cuda(),
# #     resolution=RESOLUTION,
# #     grid=grid.cuda(),
# #     recon=recon.cuda(),
# # )
# #
# #
# # torch.save(density.cpu(), "density.pt")
# #
# # recon_plot = density.clone()
# # recon_plot /= recon_plot.max()
# #
# #
# # recon_plot[recon_plot < 0.8] = 0.0
# #
# # x_plot = (x.numpy() + 1) * (RESOLUTION / 2)
# #
# # # Create a PyVista grid
# # plotter = pv.Plotter()
# # plotter.add_volume(
# #     recon_plot.cpu().numpy(),
# #     cmap="viridis",
# #     opacity="sigmoid",
# # )
# #
# # plotter.add_points(
# #     x_plot,
# #     render_points_as_spheres=True,
# #     point_size=5,
# #     color="red",
# #     show_scalar_bar=False,
# # )
# #
# # plotter.show()
