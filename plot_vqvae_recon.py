import matplotlib.pyplot as plt
import torch

ect_gt = torch.load("results/ect.pt").cpu().squeeze()
ect_recon = torch.load("results/recon.pt").cpu().squeeze()
pc_gt = torch.load("results/pc.pt").cpu()

# |%%--%%| <n9JXwsk4pi|4kmXxsvRul>


fig, axes = plt.subplots(2, 5, figsize=(5, 2))

for ax, rec, gt in zip(axes.T, ect_recon, ect_gt):
    ax[0].imshow(rec)
    ax[0].axis("off")
    ax[1].imshow(gt)
    ax[1].axis("off")


fig.tight_layout()


# |%%--%%| <4kmXxsvRul|FdwZLETIeu>

ect_gt_diffs = torch.diff(ect_gt, dim=1)
ect_recon_diffs = torch.diff(ect_recon, dim=1)

fig, axes = plt.subplots(2, 5, figsize=(5, 2))

for ax, rec, gt in zip(axes.T, ect_recon_diffs, ect_gt_diffs):
    ax[0].imshow(rec)
    ax[0].axis("off")
    ax[1].imshow(gt)
    ax[1].axis("off")


fig.tight_layout()


# |%%--%%| <FdwZLETIeu|aU68wOViBB>

from skimage.transform import iradon

fpb_gt = ect_recon_diffs[9].numpy()
theta = torch.linspace(0, 360, 64).numpy()

fig, axes = plt.subplots(3, 5, figsize=(5, 3))

for ax, rec, gt, pc in zip(axes.T, ect_recon_diffs, ect_gt_diffs, pc_gt):
    recon_fbp = iradon(rec.numpy(), theta=theta, filter_name=None)
    gt_fbp = iradon(gt.numpy(), theta=theta, filter_name=None)

    ax[0].imshow(recon_fbp)
    ax[0].axis("off")
    ax[1].imshow(gt_fbp)
    ax[1].axis("off")
    ax[2].scatter(pc[:, 0], pc[:, 1])
    ax[2].axis("off")

fig.tight_layout()
