import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda:0"
ECT_PLOT_CONFIG = {"cmap": "bone", "vmin": -0.5, "vmax": 1.5}
PC_PLOT_CONFIG = {"s": 5, "c": ".5"}
LIGHTRED = [255, 100, 100]

def plot_point_clouds_grid_2d(pcs):

    grid_size = 4
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(grid_size * 2, grid_size *  2))

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i*grid_size + j 
            ax = axes[i,j]
            ax.scatter(pcs[idx][:, 0], pcs[idx][:, 1], **PC_PLOT_CONFIG)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_aspect(1)
            ax.axis("off")
    return fig

