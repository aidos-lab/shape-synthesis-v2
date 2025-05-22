import argparse

import matplotlib.pyplot as plt
import torch
from dect.nn import EctConfig
from torch import nn
from torch.optim import Adam

from shape_synthesis.datasets.mnist import DataConfig, get_dataloaders
from shape_synthesis.datasets.transforms import EctTransform
from shape_synthesis.models.vae import VAE


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def show_image(x, idx):
    x = x.view(-1, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())
    plt.show()


def main():
    epochs = 30
    config = DataConfig(
        root="./data",
        raw="./data/raw",
        num_pts=256,
        module="datasets.mnist",
        batch_size=32,
    )
    train_dataloader, test_dataloader = get_dataloaders(config, dev=True)
    model = VAE()
    BCE_loss = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

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

    ect_transform = EctTransform(config=ect_config)

    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (pc,) in enumerate(train_dataloader):

            x = ect_transform(pc).flatten(start_dim=1)

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
            overall_loss / (batch_idx * 32),
        )

    print("Finish!!")

    model.eval()
    with torch.no_grad():
        for batch_idx, (pc,) in enumerate(test_dataloader):
            x = ect_transform(pc).flatten(start_dim=1)
            x_hat, _, _ = model(x)
            show_image(x, idx=0)
            show_image(x_hat, idx=0)
            break

    with torch.no_grad():
        noise = torch.randn(32, 200)
        generated_images = model.forward_decoder(noise)
        show_image(generated_images, idx=0)
