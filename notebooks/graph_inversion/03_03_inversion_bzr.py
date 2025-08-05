r"""°°°
# Graph reconstruction in 3d

In 3d there is no implemented filtered Back projection, hence we accomodate it
by structuring the ECT to make the FBP easier.

Some notes:

Most likely it is better to, instead of computing the ECT, compute the impulses,
e.g. map the heights along a direction to an

°°°"""

# |%%--%%| <7QC7ccvxzj|ws4ZVs8aUd>


import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from torch_geometric.datasets import TUDataset

np.random.seed(42)

NUM_STEPS = 512


def generate_thetas():
    v = []
    for theta in torch.linspace(0, torch.pi, 8):
        for phi in torch.linspace(0, torch.pi, 64):
            v.append(
                torch.tensor(
                    [
                        torch.sin(phi) * torch.cos(theta),
                        torch.sin(phi) * torch.sin(theta),
                        torch.cos(phi),
                    ]
                )
            )
    return torch.vstack(v).T


v = generate_thetas()

# |%%--%%| <ws4ZVs8aUd|GNFK8Jmwqe>

scale = 1000


# Needs to be changed to dirac deltas.
def compute_ect(x, v, ei=None, radius=1):
    nh = x @ v
    lin = torch.linspace(-radius, radius, NUM_STEPS).view(-1, 1, 1)
    ecc = torch.nn.functional.sigmoid(scale * torch.sub(lin, nh)) * (
        1 - torch.nn.functional.sigmoid(scale * torch.sub(lin, nh))
    )
    ecc = ecc.sum(axis=1)
    if ei is not None:
        eh = nh[ei].mean(axis=0)
        eccedge = torch.nn.functional.sigmoid(scale * torch.sub(lin, eh)) * (
            1 - torch.nn.functional.sigmoid(scale * torch.sub(lin, eh))
        )
        eccedge = eccedge.sum(axis=1)
        ecc -= eccedge
    return ecc


# |%%--%%| <GNFK8Jmwqe|hXqHlmje0j>


dataset = TUDataset(root="./data", name="BZR", use_node_attr=True)

data = dataset[1]
# x = torch.hstack([data.x[:, :3], torch.zeros(len(data.x), 1)])
x = data.x[:, :3]
x -= x.mean(axis=0)
x /= x.norm(dim=-1).max()

x *= 0.7

ei = data.edge_index
print(ei.shape)
print(x.shape)


# |%%--%%| <hXqHlmje0j|GoejiMqV4O>

# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# # The graph to visualize
# # G = nx.cycle_graph(20)
# # pos = nx.spring_layout(G, dim=3, seed=779)
#
# # 3d spring layout
# # Extract node and edge positions from the layout
# node_xyz = data.x.numpy()
# pts = data.x.numpy()
#
# edge_xyz = np.array([(pts[u], pts[v]) for u, v in data.edge_index.T])
#
#
# # node_xyz = pts
# # edge_xyz = np.array(ei_recon).T
#
# # Create the 3D figure
# fig = plt.figure()
# ax = fig.add_subplot(121, projection="3d")
#
# # Plot the nodes - alpha is scaled by "depth" automatically
# ax.scatter(*node_xyz.T, s=100, ec="w")
#
# # Plot the edges
# for vizedge in edge_xyz:
#     ax.plot(*vizedge.T, color="tab:gray")
#
#
# def _format_axes(ax):
#     """Visualization options for the 3D axes."""
#     # Turn gridlines off
#     ax.grid(False)
#     # Suppress tick labels
#     for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
#         dim.set_ticks([])
#     # Set axes labels
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#
#
# _format_axes(ax)
# fig.tight_layout()

# |%%--%%| <GoejiMqV4O|dkK3uOn7LI>


# ect = compute_ect(data.x, v, ei=data.edge_index)
ect = compute_ect(x, v, radius=1)

ect.shape
plt.imshow(ect)

# |%%--%%| <dkK3uOn7LI|xJPxvQLINr>

# ####################################################
# ### Reconstruct 3d using filtered backprojection.
# ####################################################

xg, yg, zg = np.meshgrid(
    np.linspace(-1, 1, NUM_STEPS, endpoint=False),
    np.linspace(-1, 1, NUM_STEPS, endpoint=False),
    np.linspace(-1, 1, NUM_STEPS, endpoint=False),
    indexing="ij",
    sparse=True,
)

recon = torch.zeros(NUM_STEPS, NUM_STEPS, NUM_STEPS)


def calc_idx(theta, xg, yg, zg):
    heights = theta[0] * xg + theta[1] * yg + theta[2] * zg
    idx = ((heights + 1) * NUM_STEPS / 2).long() + 1
    idx[idx > NUM_STEPS - 1] = NUM_STEPS - 1
    return idx


i = 0
for theta, slice in zip(v.T, ect.T):
    i += 1
    idx = calc_idx(theta, xg, yg, zg)
    reps = slice[idx]
    recon += reps


plt.imshow(recon[:, :, int(NUM_STEPS / 2)])

print(recon.shape)

# |%%--%%| <xJPxvQLINr|d1HKzKerNr>


# |%%--%%| <d1HKzKerNr|2vaJ8BMbLs>

recon_plot = recon.clone()
recon_plot /= recon_plot.max()

recon_plot[recon_plot < 0.75] = 0.0

# Create a PyVista grid
plotter = pv.Plotter()
plotter.add_volume(recon_plot.cpu().numpy(), cmap="viridis", opacity="sigmoid")
plotter.add_points(points=(x.numpy() + 1) * (NUM_STEPS / 2))
plotter.show()

# |%%--%%| <2vaJ8BMbLs|VjPxVbsmbF>

from scipy.ndimage import maximum_filter, minimum_filter

recon_np = recon.numpy()
res = maximum_filter(recon_np, footprint=np.ones((11, 11, 11)))
mask = recon_np == res
# plt.imshow(mask[:,:,100])

# |%%--%%| <VjPxVbsmbF|YAmSaqWlwQ>

idxx, idxy, idxz = np.nonzero(mask)
vals = recon[idxx, idxy, idxz]

vals /= vals.max()

idx = np.where(vals > 0.4)

idx_x = idxx[idx]
idx_y = idxy[idx]
idx_z = idxz[idx]

lin = np.linspace(-1, 1, NUM_STEPS, endpoint=False)


pts = torch.tensor(np.vstack([lin[idx_x], lin[idx_y], lin[idx_z]]).T)

print(pts.shape)
# print(data.x)


# sorted(vals,reverse=True)[:30]

# |%%--%%| <YAmSaqWlwQ|QKJfXRsfM4>

plt.scatter(data.x[:, 0], data.x[:, 2])
plt.scatter(pts[:, 0], pts[:, 2])


# |%%--%%| <QKJfXRsfM4|aRpxrMnJBL>
r"""°°°
# Reconstruct Edges
°°°"""
# |%%--%%| <aRpxrMnJBL|iBUF8TJn4u>

res = minimum_filter(recon_np, footprint=np.ones((11, 11, 11)))
mask = recon_np == res
plt.imshow(mask[:, :, 128])

# |%%--%%| <iBUF8TJn4u|jbWubb8bIf>

idxx, idxy, idxz = np.nonzero(mask)
vals = recon[idxx, idxy, idxz]
vals /= vals.min()
idx = np.where(vals > 0.8)

idx_x = idxx[idx]
idx_y = idxy[idx]
idx_z = idxz[idx]

lin = np.linspace(-1, 1, NUM_STEPS, endpoint=False)


edge_pts = torch.tensor(np.vstack([lin[idx_x], lin[idx_y], lin[idx_z]]).T)

print(vals.sort()[0])


# |%%--%%| <jbWubb8bIf|Oen2zoZpy2>

ei_true = []
for ei_idx in ei.T:
    # print(ei_idx)
    ei_true.append((x[ei_idx[0]] + x[ei_idx[1]]) / 2)

ei_true = torch.vstack(ei_true)
print(ei_true)
print(edge_pts)

# |%%--%%| <Oen2zoZpy2|YnajWBqfaM>

# plt.scatter(pts[:,0],pts[:,1])
plt.scatter(edge_pts[:, 0], edge_pts[:, 1])
plt.scatter(ei_true[:, 0], ei_true[:, 1])

# |%%--%%| <YnajWBqfaM|jtwE4DSBLv>

print(ei.shape)
print(edge_pts.shape)

# |%%--%%| <jtwE4DSBLv|qAAsU1TwOZ>

adj = np.zeros((len(pts), len(pts)))
ei_recon = []
for i in range(len(pts)):
    for j in range(len(pts)):
        pt_i = pts[i].reshape(1, 3)
        pt_j = pts[j].reshape(1, 3)
        pt = (pt_i + pt_j) / 2

        for epts in edge_pts:
            if torch.norm(epts - pt) < 0.005:
                ei_recon.append([i, j])

        # pt.repeat(edge_pts.shape[0],axis=0)-edge_pts
        # print(np.linalg.norm(pt.repeat(edge_pts.shape[0],dim=0)-edge_pts,axis=1).min())
        # if np.linalg.norm(pt.repeat(edge_pts.shape[0],axis=0)-edge_pts,axis=1).min() < .1:
        # print("true")
print(torch.tensor(ei_recon).shape)
print(torch.tensor(ei_recon))

# |%%--%%| <qAAsU1TwOZ|1ghGXulIEI>

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# The graph to visualize
# G = nx.cycle_graph(20)
# pos = nx.spring_layout(G, dim=3, seed=779)

# 3d spring layout
# Extract node and edge positions from the layout
node_xyz = pts.numpy()

edge_xyz = np.array([(pts[u].numpy(), pts[v].numpy()) for u, v in ei_recon])


# node_xyz = pts
# edge_xyz = np.array(ei_recon).T

# Create the 3D figure
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")

# Plot the nodes - alpha is scaled by "depth" automatically
ax.scatter(*node_xyz.T, s=100, ec="w")

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")


def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


_format_axes(ax)
fig.tight_layout()

########################################################################
###
########################################################################

ax = fig.add_subplot(122, projection="3d")

node_xyz = data.x.numpy()

edge_xyz = np.array([(node_xyz[u], node_xyz[v]) for u, v in data.edge_index.T])

# Plot the nodes - alpha is scaled by "depth" automatically
ax.scatter(*node_xyz.T, s=100, ec="w")

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")


def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


_format_axes(ax)
fig.tight_layout()


plt.show()


# |%%--%%| <1ghGXulIEI|dUlCAOPsk5>
