import argparse

import matplotlib.pyplot as plt
import torch
from dect.nn import EctConfig
from torch import nn
from torch.optim import Adam
from torchvision.transforms import Resize
from torchvision.utils import save_image

from shape_synthesis.datasets.mnist import DataConfig, get_dataloaders
from shape_synthesis.datasets.transforms import EctTransform
from shape_synthesis.models.vae import VAE

IMG_SIZE = 28
LATENT_DIM = 200
HIDDEN_DIM = 400


def loss_function(x, x_hat, mean, log_var):
    # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def show_image(x, idx):
    x = x.view(-1, IMG_SIZE, IMG_SIZE)
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(3 * 5, 6))
    for ax, gt in zip(axes.T, x):
        ax.imshow(gt.cpu().squeeze().numpy())
        ax.axis("off")
    plt.show()


def main():
    epochs = 30
    device = "cuda"
    config = DataConfig(
        root="./data",
        raw="./data/raw",
        num_pts=256,
        module="datasets.mnist",
        batch_size=100,
    )
    train_dataloader, test_dataloader = get_dataloaders(config, dev=False)
    model = VAE(in_dim=IMG_SIZE, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(
        device
    )
    optimizer = Adam(model.parameters(), lr=1e-3)

    ect_config = EctConfig(
        num_thetas=28,
        resolution=28,
        r=3,
        scale=14,
        ect_type="points",
        ambient_dimension=2,
        normalized=True,
        seed=2011,
    )

    ect_transform = EctTransform(config=ect_config, device=device)
    resize_transform = Resize(size=IMG_SIZE)

    print("Start training VAE...")
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, imgs) in enumerate(train_dataloader):

            x = imgs.to(device)
            x = resize_transform(x).flatten(start_dim=1)

            # x = ect_transform(x.to(device)).flatten(start_dim=1)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            "\tEpoch",
            epoch + 1,
            "complete!",
            "\tAverage Loss: ",
            overall_loss / (batch_idx * config.batch_size),
        )

    print("Finish!!")

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, imgs) in enumerate(test_dataloader):
            # x = ect_transform(x.to(device)).flatten(start_dim=1)
            x = imgs.to(device)
            x = resize_transform(x).flatten(start_dim=1)
            x_hat, _, _ = model(x)
            show_image(x, idx=0)
            show_image(x_hat, idx=0)
            break

    model.eval()
    with torch.no_grad():
        noise = torch.randn(config.batch_size, LATENT_DIM)
        generated_images = model.forward_decoder(noise.to(device))
        show_image(generated_images, idx=0)
        save_image(
            generated_images.view(config.batch_size, 1, IMG_SIZE, IMG_SIZE),
            "generated_sample.png",
        )
