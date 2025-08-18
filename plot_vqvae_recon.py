import matplotlib.pyplot as plt
import torch

ect_gt = (
    torch.load("mnist/vqvae_autoencoder_samples/ground_truth_0002.pt").cpu().squeeze()
)
ect_recon = (
    torch.load("mnist/vqvae_autoencoder_samples/reconstruction_0002.pt").cpu().squeeze()
)
# |%%--%%| <n9JXwsk4pi|4kmXxsvRul>

plt.imshow(ect_recon[0])
