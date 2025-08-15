from dataclasses import dataclass
import matplotlib.pyplot as plt

from metrics.loss import chamfer2DECT
import torch
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from torch import optim

from datasets.mnist import DataConfig, get_dataloaders
from datasets.transforms import get_transform
from models.encoder import Model, ModelConfig


""" This script is an example of Sigma VAE training in PyTorch. The code was adapted from:
https://github.com/pytorch/examples/blob/master/vae/main.py """

DEVICE = "cuda"


@dataclass
class TrainConfig:
    epochs = 10


@dataclass
class LogConfig:
    log_dir: str = "base_encoder"
    model_str: str = "encoder"
    log_interval: int = 10


def train(epoch, model, train_loader, optimizer, transform, logger):
    model.train()
    for pcs, _ in train_loader:
        optimizer.zero_grad()
        ect_gt = transform(pcs.to(DEVICE))
        pcs_pred = model(ect_gt).reshape(-1, 256, 2)
        ect_pred = transform(pcs_pred)

        loss, ect_loss, cd_loss = chamfer2DECT(
            pcs_pred, pcs.to(DEVICE), ect_pred, ect_gt
        )

        loss.backward()

        optimizer.step()
        logger.log_metrics(
            {
                "train/loss": loss,
                "train/ect": ect_loss,
                "train/cd": cd_loss,
            },
            epoch,
        )


@torch.no_grad()
def test(epoch, model, test_loader, transform, logger):
    model.eval()
    for idx, (pcs, _) in enumerate(test_loader):
        pcs = pcs.to(DEVICE)
        ect_gt = transform(pcs)
        pcs_pred = model(ect_gt).reshape(-1, 256, 2)
        ect_pred = transform(pcs_pred)

        loss, ect_loss, cd_loss = chamfer2DECT(pcs_pred, pcs, ect_pred, ect_gt)

        logger.log_metrics(
            {
                "test/loss": loss,
                "test/ect": ect_loss,
                "test/cd": cd_loss,
            },
            epoch,
        )
        if idx == 0:
            # Scatter plot of a point cloud figure to tensorboard

            plt.scatter(
                pcs_pred[1][:, 0].cpu().detach().numpy(),
                pcs_pred[1][:, 1].cpu().detach().numpy(),
            )
            plt.title("example title")
            logger.experiment.add_figure("my_figure_batch", plt.gcf(), epoch)

            print("adding")


def main():
    train_config = TrainConfig()
    log_config = LogConfig()

    data_config = DataConfig(
        root="./data",
        raw="./data/raw",
        num_pts=256,
        module="datasets.mnist",
        batch_size=64,
    )

    train_loader, test_loader = get_dataloaders(config=data_config, dev=False)

    transform = get_transform(compiled=False)

    # Logger
    logger = TensorBoardLogger(
        f"./encoder_logs/{log_config.model_str}", name=log_config.model_str
    )

    model_config = ModelConfig(
        module="",
        num_pts=256,
        num_thetas=32,
        resolution=32,
        ambient_dimension=2,
    )

    ## Build Model
    model = Model(model_config).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        test(epoch, model, test_loader, transform, logger)
    # torch.save(
    #     model.state_dict(),
    #     "encoder_logs/{}/checkpoint_{}.pt".format(log_config.log_dir, str(epoch)),
    # )


if __name__ == "__main__":
    main()
