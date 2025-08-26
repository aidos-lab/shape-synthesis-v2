"""
NOTE: Under construction. Works, but needs to be spliced into the current structure.
Source:
"""

import argparse
import os

import numpy as np
import torch
from configs import load_config
import yaml
from datasets.qm9 import get_dataloaders
from src.schedulers.linear_scheduler import LinearNoiseScheduler
from torch.optim import Adam
from tqdm import tqdm

from src.models.unet import Unet
from src.models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    # Parse the args
    config_path = args.config_path
    dev = args.dev

    config, _ = load_config(config_path)

    print("im here")

    # dataset_config = config["dataset_params"]
    # diffusion_model_config = config["ldm_params"]
    # autoencoder_model_config = config["autoencoder_params"]
    # config.train = config["train_params"]

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(config=config.noise_scheduler)

    dataloader, _ = get_dataloaders(config.data, dev=dev)

    # Instantiate the model
    model = Unet(
        im_channels=config.vae.z_channels,
        model_config=config.ldm_params,
    ).to(device)
    model.train()

    vae = VQVAE(config.vae).to(device)
    vae.eval()

    # Load vae if found
    if os.path.exists(
        os.path.join(config.train.task_name, config.train.vqvae_autoencoder_ckpt_name)
    ):
        print("Loaded vae checkpoint")
        print(
            os.path.join(
                config.train.task_name,
                config.train.vqvae_autoencoder_ckpt_name,
            )
        )
        vae.load_state_dict(
            torch.load(
                os.path.join(
                    config.train.task_name,
                    config.train.vqvae_autoencoder_ckpt_name,
                ),
                map_location=device,
                weights_only=True,
            ),
            strict=True,
        )

    # Specify training parameters
    num_epochs = config.train.ldm_epochs
    optimizer = Adam(model.parameters(), lr=config.train.ldm_lr)
    criterion = torch.nn.MSELoss()

    # Run training
    for param in vae.parameters():
        param.requires_grad = False

    for epoch_idx in range(num_epochs):
        print(epoch_idx)
        losses = []
        for im in tqdm(dataloader):
            im = im[0][:, :3, :, :].to(device)
            optimizer.zero_grad()
            # im = im.float().to(device)
            with torch.no_grad():
                im, _ = vae.encode(im)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(
                0,
                config.noise_scheduler.num_timesteps,
                (im.shape[0],),
            ).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(
            "Finished epoch:{} | Loss : {:.4f}".format(epoch_idx + 1, np.mean(losses))
        )

        torch.save(model.state_dict(), "results/latent_ddpm.ckpt")

    print("Done Training ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    parser.add_argument(
        "--config", dest="config_path", default="config/qm9.yaml", type=str
    )
    parser.add_argument(
        "--dev", default=False, action="store_true", help="Run a small subset."
    )
    args = parser.parse_args()

    train(args)
