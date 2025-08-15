import matplotlib.pyplot as plt
import torch

recon = torch.load("./results/recon.pt")
ref = torch.load("./results/ref.pt")

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))


for ax, ref_ect, recon_ect in zip(axes.T, ref, recon):
    ax[0].imshow(recon_ect.cpu().squeeze().numpy())
    ax[0].axis("off")
    ax[1].imshow(ref_ect.cpu().squeeze().numpy())
    ax[1].axis("off")

plt.tight_layout()
plt.show()
