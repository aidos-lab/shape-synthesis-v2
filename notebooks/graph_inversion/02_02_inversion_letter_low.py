import itertools
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torchvision.transforms.functional as f
from torch_geometric.datasets import TUDataset

np.random.seed(42)

NUM_STEPS = 512
NUM_PTS = 15

scale = 200


def compute_ect(x, v, ei=None, radius=1.1):
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


# |%%--%%| <Up37Ib12cp|Tuht7rCsAo>

dataset = TUDataset(root="./data", name="Letter-low", use_node_attr=True)

data = dataset[4]
x = data.x
ei = data.edge_index
g = torch_geometric.utils.to_networkx(data, to_undirected=True)
nx.draw(g, pos=x.numpy())
plt.show()
print(ei)
print(x)


# |%%--%%| <Tuht7rCsAo|op1gzHi9q9>

v = torch.vstack(
    [
        torch.sin(torch.linspace(0, 2 * torch.pi, NUM_STEPS)),
        torch.cos(torch.linspace(0, 2 * torch.pi, NUM_STEPS)),
    ]
)

ect = compute_ect(x, v, ei=ei, radius=5)

plt.imshow(ect)

# |%%--%%| <op1gzHi9q9|fT063nmbwL>

from skimage.transform import iradon

sinogram = ect.numpy()
theta = torch.linspace(0, 360, NUM_STEPS).numpy()

reconstruction_fbp = iradon(sinogram, theta=theta, filter_name="ramp") * 100

plt.imshow(reconstruction_fbp * 100)
plt.show()

# |%%--%%| <fT063nmbwL|cLoPzRvFNQ>

# Local peak detection

import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data, img_as_float
from skimage.feature import peak_local_max

im = img_as_float(reconstruction_fbp)

# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(im, size=5, mode="constant")

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(im, min_distance=20, threshold_rel=0.1)

# display results
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(im, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("Original")

ax[1].imshow(image_max, cmap=plt.cm.gray)
ax[1].axis("off")
ax[1].set_title("Maximum filter")

ax[2].imshow(im, cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(coordinates[:, 1], coordinates[:, 0], "r.")
ax[2].axis("off")
ax[2].set_title("Peak local max")

fig.tight_layout()

plt.show()


# |%%--%%| <cLoPzRvFNQ|zdkjH6MxsJ>

# Peak intensities
ect[coordinates[:, 0], coordinates[:, 1]]


# x = torch.tensor([
#                     [.6,0],
#                     [0.0,.5],
#                     [-.4,.4],
#                     [0,0]])


# Vertex coordinates
lin = torch.linspace(-1.3, 1.3, NUM_STEPS).view(-1, 1, 1)
x_hat = torch.vstack(
    [lin[512 - coordinates[:, 0]].squeeze(), lin[coordinates[:, 1]].squeeze()]
).T
x_hat.shape
print(x_hat)

# |%%--%%| <zdkjH6MxsJ|K0g6OgZZet>

import itertools

# Reconstruct adjacency matrix

recon_ect = []

for x_hat_i in x_hat:
    recon_ect.append(compute_ect(x_hat_i, v))


adj = torch.zeros((len(x_hat), len(x_hat)))

for i in range(len(x_hat)):
    for j in range(len(x_hat)):
        ect = compute_ect((x_hat[i] + x_hat[j]) / 2, v)
        rec = torch.tensor(
            iradon(ect.numpy(), theta=theta, filter_name="ramp"), dtype=torch.float
        )

        adj[i, j] = -1 * (rec * reconstruction_fbp).sum()

adj[adj < 1] = 0
adj[adj > 1] = 1
print(adj)


# |%%--%%| <K0g6OgZZet|W1Yf2lp4ro>

# # Create object from it
from torch_geometric.data import Data

data_recon = Data(x=x_hat, edge_index=torch.nonzero(adj).T)
g = torch_geometric.utils.to_networkx(data_recon, to_undirected=True)
nx.draw(g, pos=data_recon.x.numpy())

# |%%--%%| <W1Yf2lp4ro|AaSZ5VMgaM>

x

# |%%--%%| <AaSZ5VMgaM|mJdgsNu6CP>

ei

# |%%--%%| <mJdgsNu6CP|cMqOTPiN0p>


# |%%--%%| <cMqOTPiN0p|dfxUOAIjxD>


# |%%--%%| <dfxUOAIjxD|LV7jabei1a>


# import numpy as np
# import matplotlib.pyplot as plt

# from skimage import measure

# r = reconstruction_fbp * 1000

# # # Construct some test data
# # x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
# # r = np.sin(np.exp(np.sin(x)**3 + np.cos(y)**2))


# # Find contours at a constant value of 0.8
# # contours = measure.find_contours(r)

# # # Comparison between image_max and im to find the coordinates of local maxima
# # coordinates = peak_local_max(r, min_distance=50)


# # Display the image and plot all contours found
# fig, ax = plt.subplots()
# ax.imshow(r, cmap=plt.cm.gray)
# # ax.scatter(coordinates[:,0],coordinates[:,1])

# import cv2
# sobelxy = cv2.Sobel(src=r, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# plt.imshow(sobelxy)
# plt.show()

# for contour in contours:
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()


# # # error = reconstruction_fbp - image
# # # print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')


# # # # ####################################################
# # # # ### Reconstruct using filtered backprojection.
# # # # ####################################################

# # # img = torch.zeros(NUM_STEPS,NUM_STEPS)
# # # for slice, theta in zip(ect.T,torch.linspace(0,360,NUM_STEPS)):
# # #     reps = slice.unsqueeze(1).repeat(1, NUM_STEPS).unsqueeze(0)
# # #     img +=f.rotate(reps, theta.item()).squeeze()

# # # plt.imshow(img.numpy())
# # # plt.show()


# # # # m = nn.MaxPool1d(kernel_size=3,stride=1,ceil_mode=True,padding=1)
# # # # output = m(ect.T).T
# # # # out2 = (torch.abs(output) > 1)
# # # # out = torch.abs(output - ect) == 0.0

# # # # out_full = torch.logical_and(out,out2)
# # # # plt.imshow(out_full)
# # # # plt.show()
# # # # thetas_idx, height_idx  = torch.nonzero(out_full.squeeze(),as_tuple=True)
