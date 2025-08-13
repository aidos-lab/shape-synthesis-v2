from dataclasses import dataclass

import torch
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from torch import optim
from torchmetrics.image.fid import FrechetInceptionDistance

from datasets.qm9 import DataConfig, get_dataloaders
from datasets.transforms import get_transform

# from shape_synthesis.models.sigma_vae import ConvVAE
from models.vae_base import VAE

from metrics.loss import compute_mse_kld_loss_fn

""" This script is an example of Sigma VAE training in PyTorch. The code was adapted from:
https://github.com/pytorch/examples/blob/master/vae/main.py """

DEVICE = "cuda"


@dataclass
class TrainConfig:
    epochs = 100


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

    train_loader, test_loader = get_dataloaders(config=data_config, dev=True)

    transform = get_transform(compiled=False)

    # Logger
    logger = TensorBoardLogger(
        f"./vae_logs/{log_config.model_str}", name=log_config.model_str
    )

    ## Build Model
    model = VAE(in_dim=120, hidden_dim=600, latent_dim=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, train_config.epochs + 1):
        print("Epoch ", epoch)
        train(
            epoch,
            model,
            train_loader,
            optimizer,
            transform,
            logger=logger,
        )
        # if (epoch % 10 == 0):
        test(epoch, model, test_loader, transform, logger, data_config.batch_size)
    torch.save(
        model.state_dict(),
        "vae_logs/{}/checkpoint_{}.pt".format(log_config.log_dir, str(epoch)),
    )


def train(epoch, model, train_loader, optimizer, transform, logger):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        batch.to(DEVICE)
        # Transform the point cloud to an ECT.
        imgs = transform(x=batch.pos.cuda(), index=batch.batch.cuda())

        optimizer.zero_grad()

        # Run VAE
        recon_batch, mu, logvar = model(imgs)

        # Compute loss
        loss, rec, kl = compute_mse_kld_loss_fn(
            recon_batch,
            mu,
            logvar,
            imgs.unsqueeze(1),
            beta=0.00,
        )
        # rec, kl = model.loss_function(recon_batch, imgs, mu, logvar)

        # total_loss = rec + kl
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(train_loss)
    # train_loss /= len(train_loader.dataset)
    # logger.log_metrics(
    #     {
    #         "train/elbo": train_loss,
    #         "train/rec": rec.item() / len(imgs),
    #         "train/kld": kl.item() / len(imgs),
    #         # "train/log_sigma": model.log_sigma,
    #     },
    #     epoch,
    # )


def test(epoch, model, test_loader, transform, logger, batch_size):
    fid_rec = FrechetInceptionDistance(
        feature=64, input_img_size=(256, 256), normalize=True
    ).cuda()
    fid_sample = FrechetInceptionDistance(
        feature=64, input_img_size=(256, 256), normalize=True
    ).cuda()
    model.eval()
    test_loss = 0
    kl_loss = 0
    rec_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch.to(DEVICE)

            imgs = transform(x=batch.pos.cuda(), index=batch.batch.cuda())
            # Transform the point cloud to an ECT.

            recon_batch, mu, logvar = model(imgs)
            # Pass the second value from posthoc VAE
            # rec, kl = model.loss_function(recon_batch, imgs, mu, logvar)
            _, rec, kl = compute_mse_kld_loss_fn(
                recon_batch,
                mu,
                logvar,
                imgs,
                beta=0.0001,
            )

            sample = model.sample(len(imgs))

            fid_rec.update(imgs.unsqueeze(1).repeat(1, 3, 1, 1), real=True)
            fid_rec.update(recon_batch.repeat(1, 3, 1, 1), real=False)

            fid_sample.update(imgs.unsqueeze(1).repeat(1, 3, 1, 1), real=True)
            fid_sample.update(sample.repeat(1, 3, 1, 1), real=False)

            rec_loss += rec
            kl_loss += kl
            test_loss += rec + kl

            if i == 0:
                n = min(imgs.size(0), 8)
                comparison = torch.cat(
                    [
                        imgs.unsqueeze(1)[:n],
                        recon_batch.view(batch_size, 1, 256, 256)[:n],
                    ]
                )
                logger.experiment.add_images(
                    "Reconstruction", comparison, global_step=epoch, dataformats="NCHW"
                )
                logger.experiment.add_images(
                    "Sample", sample, global_step=epoch, dataformats="NCHW"
                )
    test_loss /= len(test_loader.dataset)
    logger.log_metrics(
        {
            "test/elbo": test_loss,
            "test/rec": rec_loss,
            "test/kld": kl_loss,
            "test/fid_rec": fid_rec.compute(),
            "test/fid_sample": fid_sample.compute(),
        },
        epoch,
    )


if __name__ == "__main__":
    main()
