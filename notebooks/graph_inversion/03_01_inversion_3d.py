r"""°°°
# Graph reconstruction in 3d

In 3d there is no implemented filtered Back projection, hence we accomodate it
by structuring the ECT to make the FBP easier.


°°°"""

# |%%--%%| <Snf6GbPzpk|YzTK1wytsZ>

import itertools
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as f

np.random.seed(42)

NUM_STEPS = 256
NUM_PTS = 10


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

# |%%--%%| <YzTK1wytsZ|5B7ui2I9hc>

scale = 450


def compute_ect(x, v, ei=None):
    nh = x @ v
    lin = torch.linspace(-1, 1, NUM_STEPS).view(-1, 1, 1)
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


x = torch.tensor(
    [
        [0, 0, 0.0],
        [0, -0.7, 0.0],
        [0.5, 0.5, 0.0],
    ]
)


# x = torch.vstack(
#     [
#        .6 * torch.sin(
#             torch.tensor(
#                 np.linspace(0, torch.pi * 2, NUM_PTS, endpoint=False),
#                 dtype=torch.float32,
#             )
#         ),
#         .3 * torch.cos(
#             torch.tensor(
#                 np.linspace(0, torch.pi * 2, NUM_PTS, endpoint=False),
#                 dtype=torch.float32,
#             )
#         ),
#         torch.zeros_like(torch.linspace(0, torch.pi * 2, NUM_PTS))
#     ]
#     ).T

x = torch.tensor(np.random.uniform(-0.7, 0.7, size=(NUM_PTS, 3)), dtype=torch.float)
from torch_geometric.utils import erdos_renyi_graph

ei = erdos_renyi_graph(NUM_PTS, 0.1)
# ei = torch.vstack([torch.zeros(NUM_PTS-1,dtype=torch.long),torch.arange(1,NUM_PTS,dtype=torch.long)])
# ei = torch.tensor([[0, 1, 2],
#                    [1, 2, 0]])

# |%%--%%| <5B7ui2I9hc|9eyLCW9TEJ>

ei

# |%%--%%| <9eyLCW9TEJ|oK20vzn4Rl>

ect = compute_ect(x, v, ei=ei)

ect.shape
plt.imshow(ect)

# |%%--%%| <oK20vzn4Rl|aSODlXrrWn>

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
    # if i==3:
    #     recon += reps
    #     break


plt.imshow(recon[:, :, int(NUM_STEPS / 2)])

# |%%--%%| <aSODlXrrWn|z5Djt9irNg>

from scipy.ndimage import gaussian_filter

recon_f = gaussian_filter(recon, sigma=2)

plt.imshow(recon_f[:, :, 128])

# |%%--%%| <z5Djt9irNg|3OjThdHNdw>

from scipy.ndimage import maximum_filter, minimum_filter

recon_np = recon.numpy()
res = maximum_filter(recon_np, footprint=np.ones((11, 11, 11)))
mask = recon_np == res
plt.imshow(mask[:, :, 100])

# |%%--%%| <3OjThdHNdw|9bhuA7UvPV>

idxx, idxy, idxz = np.nonzero(mask)
vals = recon[idxx, idxy, idxz]

vals /= vals.max()

idx = np.where(vals > 0.5)

idx_x = idxx[idx]
idx_y = idxy[idx]
idx_z = idxz[idx]

lin = np.linspace(-1, 1, NUM_STEPS, endpoint=False)


pts = torch.tensor(np.vstack([lin[idx_x], lin[idx_y], lin[idx_z]]).T)

print(pts)
print(x)


sorted(vals)

# |%%--%%| <9bhuA7UvPV|9xi7Uslor5>

plt.scatter(pts[:, 0], pts[:, 2])
plt.scatter(x[:, 0], x[:, 2])


# |%%--%%| <9xi7Uslor5|OgI1DYxMFG>
r"""°°°
# Reconstruct Edges
°°°"""
# |%%--%%| <OgI1DYxMFG|vXU9UDnMRm>

res = minimum_filter(recon_np, footprint=np.ones((11, 11, 11)))
mask = recon_np == res

# |%%--%%| <vXU9UDnMRm|EBf5VInMAu>

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

# |%%--%%| <EBf5VInMAu|l80LUXDsuY>

ei_true = []
for ei_idx in ei.T:
    # print(ei_idx)
    ei_true.append((x[ei_idx[0]] + x[ei_idx[1]]) / 2)

ei_true = torch.vstack(ei_true)
print(ei_true)
print(edge_pts)

# |%%--%%| <l80LUXDsuY|qUsbch8nM9>

# plt.scatter(pts[:,0],pts[:,1])
plt.scatter(edge_pts[:, 0], edge_pts[:, 1])
plt.scatter(ei_true[:, 0], ei_true[:, 1])

# |%%--%%| <qUsbch8nM9|oRyXbwEPmn>

print(ei.shape)
print(edge_pts.shape)

# |%%--%%| <oRyXbwEPmn|ahRrG4nIUt>

adj = np.zeros((len(pts), len(pts)))
ei_recon = []
for i in range(len(pts)):
    for j in range(len(pts)):
        pt_i = pts[i].reshape(1, 3)
        pt_j = pts[j].reshape(1, 3)
        pt = (pt_i + pt_j) / 2

        for epts in edge_pts:
            if torch.norm(epts - pt) < 0.2:
                ei_recon.append([i, j])

        # pt.repeat(edge_pts.shape[0],axis=0)-edge_pts
        # print(np.linalg.norm(pt.repeat(edge_pts.shape[0],dim=0)-edge_pts,axis=1).min())
        # if np.linalg.norm(pt.repeat(edge_pts.shape[0],axis=0)-edge_pts,axis=1).min() < .1:
        # print("true")
torch.tensor(ei_recon)

# |%%--%%| <ahRrG4nIUt|tJW2cY2jwZ>

pt = np.array([[1, 1, 0]])
pt2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]])


# |%%--%%| <tJW2cY2jwZ|aD37tpNqzY>

lst = [[0, 1], [1, 0], [3, 1]]
lst1 = list(set([tuple(sorted(l)) for l in lst]))
lst1
torch.tensor(lst1)

# |%%--%%| <aD37tpNqzY|Ym6uHQghFB>
