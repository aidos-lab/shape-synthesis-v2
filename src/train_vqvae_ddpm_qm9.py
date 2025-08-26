"""
NOTE: Under construction. Works, but needs to be spliced into the current structure.
Source:
"""

import argparse
import os

import numpy as np
import torch
from lightning.fabric import Fabric
from torch.optim import Adam
from tqdm import tqdm

from configs import load_config
from datasets.qm9 import get_dataloaders
from src.models import vqvae
from src.models.unet import Unet
from src.models.vqvae import VQVAE
from src.schedulers.linear_scheduler import LinearNoiseScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("medium")


def train(args):

    # Set up Fabric
    fabric = Fabric(accelerator="cuda")  # , precision="16-mixed")

    # Parse the args
    config_path = args.config_path
    dev: bool = args.dev
    compile: bool = args.compile

    config, _ = load_config(config_path)

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(config=config.noise_scheduler)
    if compile:
        scheduler = torch.compile(scheduler)
    scheduler = fabric.setup(scheduler)

    dataloader, _ = get_dataloaders(config.data, dev=dev)

    dataloader = fabric.setup_dataloaders(dataloader)

    # Instantiate the model
    model = Unet(
        im_channels=config.vae.z_channels,
        model_config=config.ldm_params,
    )

    if compile:
        model = torch.compile(model)
    optimizer = Adam(model.parameters(), lr=config.train.ldm_lr)

    model, optimizer = fabric.setup(model, optimizer)

    criterion = torch.nn.MSELoss()

    vae = VQVAE(config.vae)
    # if compile:
    #     vae = torch.compile(vae)
    state = {"model": vae}
    fabric.load("trained_models/vqvae.ckpt", state)
    vae.eval()
    vae.to(fabric.device)

    # Specify training parameters
    num_epochs = config.train.ldm_epochs

    # Run training
    for param in vae.parameters():
        param.requires_grad = False

    model.train()
    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(dataloader):
            im = im[0][:, :3, :, :]
            optimizer.zero_grad()
            # im = im.float().to(device)
            with torch.no_grad():
                im, _ = vae.encode(im)

            # Sample random noise
            noise = torch.randn_like(im)

            # Sample timestep
            t = torch.randint(
                0,
                config.noise_scheduler.num_timesteps,
                (im.shape[0],),
                device=fabric.device,
            )

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            fabric.backward(loss)
            optimizer.step()
        print(
            "Finished epoch:{} | Loss : {:.4f}".format(epoch_idx + 1, np.mean(losses))
        )

        torch.save(model.state_dict(), "trained_models/latent_ddpm.ckpt")

    print("Done Training ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    parser.add_argument(
        "--config", dest="config_path", default="config/qm9.yaml", type=str
    )
    parser.add_argument(
        "--dev", default=False, action="store_true", help="Run a small subset."
    )
    parser.add_argument(
        "--compile", default=False, action="store_true", help="Compile modules"
    )
    args = parser.parse_args()

    train(args)
