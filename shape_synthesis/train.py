from dataclasses import dataclass

import torch
from dect.nn import EctConfig
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from torch import optim
from torchvision import datasets, transforms
from torchvision.transforms import Resize

from shape_synthesis.datasets.mnist import DataConfig, get_dataloaders
from shape_synthesis.metrics.losses import vanilla_loss_function as loss_function
from shape_synthesis.models.sigma_vae import ConvVAE
from shape_synthesis.models.vae import VAE

""" This script is an example of Sigma VAE training in PyTorch. The code was adapted from:
https://github.com/pytorch/examples/blob/master/vae/main.py """

DEVICE = "cuda"


@dataclass
class TrainConfig:
    epochs = 50


@dataclass
class LogConfig:
    log_dir: str = "base_vae"
    model_str: str = "base_vae"
    log_interval: int = 10


def main():
    train_config = TrainConfig()
    log_config = LogConfig()

    data_config = DataConfig(
        root="./data",
        raw="./data/raw",
        num_pts=256,
        module="datasets.mnist",
        batch_size=128,
    )
    train_loader, test_loader = get_dataloaders(config=data_config)

    # Logger
    # logger = CustomLogger(log_dir=log_dir, log_interval=log_interval)
    logger = TensorBoardLogger(
        f"./vae_logs/{log_config.model_str}", name=log_config.model_str
    )

    ## Build Model
    model = ConvVAE(1, log_config.model_str, data_config.batch_size, "cuda").to(DEVICE)
    # model = VAE(in_dim=28, hidden_dim=400, latent_dim=200).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, train_config.epochs + 1):
        train(
            epoch,
            model,
            train_loader,
            optimizer,
            logger=logger,
        )
        test(epoch, model, test_loader, logger, data_config.batch_size)
    torch.save(
        model.state_dict(),
        "vae_logs/{}/checkpoint_{}.pt".format(log_config.log_dir, str(epoch)),
    )


def train(epoch, model, train_loader, optimizer, logger):
    model.train()
    train_loss = 0
    for batch_idx, (_, data) in enumerate(train_loader):
        data = data.unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()

        # Run VAE
        recon_batch, mu, logvar = model(data)
        # Compute loss
        rec, kl = loss_function(recon_batch, data, mu, logvar)

        total_loss = rec + kl
        total_loss.backward()
        train_loss += total_loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    logger.log_metrics(
        {
            "train/elbo": train_loss,
            "train/rec": rec.item() / len(data),
            "train/kld": kl.item() / len(data),
            # "train/log_sigma": model.log_sigma,
        },
        epoch,
    )


def test(epoch, model, test_loader, logger, batch_size):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (_, data) in enumerate(test_loader):
            data = data.unsqueeze(1).to(DEVICE)
            recon_batch, mu, logvar = model(data)
            # Pass the second value from posthoc VAE
            rec, kl = loss_function(recon_batch, data, mu, logvar)
            test_loss += rec + kl
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(batch_size, 1, 28, 28)[:n]]
                )
                logger.experiment.add_images(
                    "Reconstruction", comparison, global_step=epoch, dataformats="NCHW"
                )
    test_loss /= len(test_loader.dataset)
    with torch.no_grad():
        sample = model.sample(64).cpu()
        logger.experiment.add_images(
            "Sample", sample, global_step=epoch, dataformats="NCHW"
        )
    logger.log_metrics({"test/elbo": test_loss}, epoch)


# def main():
#     epochs = 30
#     device = "cuda"
#     config = DataConfig(
#         root="./data",
#         raw="./data/raw",
#         num_pts=256,
#         module="datasets.mnist",
#         batch_size=100,
#     )
#     train_dataloader, test_dataloader = get_dataloaders(config, dev=False)
#     model = VAE(in_dim=IMG_SIZE, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(
#         device
#     )
#     optimizer = Adam(model.parameters(), lr=1e-3)
#
#     ect_config = EctConfig(
#         num_thetas=28,
#         resolution=28,
#         r=3,
#         scale=14,
#         ect_type="points",
#         ambient_dimension=2,
#         normalized=True,
#         seed=2011,
#     )
#
#     ect_transform = EctTransform(config=ect_config, device=device)
#     resize_transform = Resize(size=IMG_SIZE)
#
#     print("Start training VAE...")
#     model.train()
#     for epoch in range(epochs):
#         overall_loss = 0
#         for batch_idx, (x, imgs) in enumerate(train_dataloader):
#
#             x = imgs.to(device)
#             x = resize_transform(x).flatten(start_dim=1)
#
#             # x = ect_transform(x.to(device)).flatten(start_dim=1)
#
#             optimizer.zero_grad()
#
#             x_hat, mean, log_var = model(x)
#             loss = loss_function(x, x_hat, mean, log_var)
#
#             overall_loss += loss.item()
#
#             loss.backward()
#             optimizer.step()
#
#         print(
#             "\tEpoch",
#             epoch + 1,
#             "complete!",
#             "\tAverage Loss: ",
#             overall_loss / (batch_idx * config.batch_size),
#         )
#
#     print("Finish!!")
#
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (x, imgs) in enumerate(test_dataloader):
#             # x = ect_transform(x.to(device)).flatten(start_dim=1)
#             x = imgs.to(device)
#             x = resize_transform(x).flatten(start_dim=1)
#             x_hat, _, _ = model(x)
#             show_image(x, idx=0)
#             show_image(x_hat, idx=0)
#             break
#
#     model.eval()
#     with torch.no_grad():
#         noise = torch.randn(config.batch_size, LATENT_DIM)
#         generated_images = model.forward_decoder(noise.to(device))
#         show_image(generated_images, idx=0)
#         save_image(
#             generated_images.view(config.batch_size, 1, IMG_SIZE, IMG_SIZE),
#             "generated_sample.png",
#         )
