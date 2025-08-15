from dataclasses import dataclass

import torch
import tqdm
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from torch import optim

from datasets.qm9 import DataConfig, get_dataloaders
from metrics.loss import compute_mse_kld_loss_fn

# from shape_synthesis.models.sigma_vae import ConvVAE
from models.vae_base import VAE

# from torchmetrics.image.fid import FrechetInceptionDistance


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
        batch_size=64,
    )

    train_loader, test_loader = get_dataloaders(config=data_config, dev=False)

    # Logger
    logger = TensorBoardLogger(
        f"./vae_logs/{log_config.model_str}", name=log_config.model_str
    )

    ## Build Model
    model = torch.compile(VAE(in_dim=120, hidden_dim=600, latent_dim=128)).to(DEVICE)
    # model = VAE(in_dim=120, hidden_dim=600, latent_dim=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, train_config.epochs + 1):
        print("Epoch ", epoch)
        train(
            epoch,
            model,
            train_loader,
            optimizer,
            logger=logger,
        )
        # if (epoch % 10 == 0):
    test(-1, model, test_loader, logger, data_config.batch_size)
    # torc.save(
    #     model.state_dict(),
    #     "vae_logs/{}/checkpoint_{}.pt".format(log_config.log_dir, str(epoch)),
    # )


def train(epoch, model, train_loader, optimizer, logger):
    model.train()
    train_loss = 0
    for batch_idx, (imgs,) in tqdm.tqdm(enumerate(train_loader)):
        imgs = imgs.to(DEVICE)
        # Transform the point cloud to an ECT.

        optimizer.zero_grad()

        # Run VAE
        recon_batch, mu, logvar = model(imgs.squeeze())

        # Compute loss
        loss, _, _ = compute_mse_kld_loss_fn(
            recon_batch.squeeze(),
            mu,
            logvar,
            imgs.squeeze(),
            beta=0.00,
        )

        # total_loss = rec + kl
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(epoch, train_loss)


def test(epoch, model, test_loader, logger, batch_size):
    # fid_rec = FrechetInceptionDistance(
    #     feature=64, input_img_size=(256, 256), normalize=True
    # ).cuda()
    # fid_sample = FrechetInceptionDistance(
    #     feature=64, input_img_size=(256, 256), normalize=True
    # ).cuda()
    model.eval()
    test_loss = 0
    kl_loss = 0
    rec_loss = 0
    with torch.no_grad():
        for i, (imgs,) in enumerate(test_loader):
            imgs = imgs.to(DEVICE)

            recon_batch, mu, logvar = model(imgs.squeeze())
            # Pass the second value from posthoc VAE
            # rec, kl = model.loss_function(recon_batch, imgs, mu, logvar)
            _, rec, kl = compute_mse_kld_loss_fn(
                recon_batch.squeeze(),
                mu,
                logvar,
                imgs.squeeze(),
                beta=0.0001,
            )

            sample = model.sample(len(imgs))

            rec_loss += rec
            kl_loss += kl
            test_loss += rec + kl

    torch.save(sample, "./results/sample.pt")
    torch.save(recon_batch, "./results/recon.pt")
    torch.save(imgs, "./results/ref.pt")
    test_loss /= len(test_loader.dataset)
    logger.log_metrics(
        {
            "test/elbo": test_loss,
            "test/rec": rec_loss,
            "test/kld": kl_loss,
            # "test/fid_rec": fid_rec.compute(),
            # "test/fid_sample": fid_sample.compute(),
        },
        epoch,
    )


if __name__ == "__main__":
    main()
