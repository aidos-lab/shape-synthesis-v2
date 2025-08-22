"""
NOTE: Under construction. Works, but needs to be spliced into the current structure.
Source:
"""

import argparse
import os

import numpy as np
import torch
import yaml
from datasets.qm9 import get_dataloaders
from src.schedulers.linear_scheduler import LinearNoiseScheduler
from torch.optim import Adam
from tqdm import tqdm

from src.models.unet_base import Unet
from src.models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    ########################

    dataset_config = config["dataset_params"]
    diffusion_model_config = config["ldm_params"]
    autoencoder_model_config = config["autoencoder_params"]
    train_config = config["train_params"]

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(config=config.diffusion)

    # data_loader = DataLoader(
    #     im_dataset, batch_size=train_config["ldm_batch_size"], shuffle=True
    # )

    config = DataConfig(
        root="./data/ectmnist",
        raw="./data/raw",
        num_pts=256,
        module="datasets.mnist",
        batch_size=32,
        resolution=64,
    )
    data_loader, _ = get_dataloaders(config, dev=False)
    # Instantiate the model
    model = Unet(
        im_channels=autoencoder_model_config["z_channels"],
        model_config=diffusion_model_config,
    ).to(device)
    model.train()

    # Load VAE ONLY if latents are not to be used or are missing
    if not im_dataset.use_latents:
        print("Loading vqvae model as latents not present")
        vae = VQVAE(
            im_channels=dataset_config["im_channels"],
            model_config=autoencoder_model_config,
        ).to(device)
        vae.eval()

        # Load vae if found
        if os.path.exists(
            os.path.join(
                train_config["task_name"], train_config["vqvae_autoencoder_ckpt_name"]
            )
        ):
            print("Loaded vae checkpoint")
            print(
                os.path.join(
                    train_config["task_name"],
                    train_config["vqvae_autoencoder_ckpt_name"],
                )
            )
            vae.load_state_dict(
                torch.load(
                    os.path.join(
                        train_config["task_name"],
                        train_config["vqvae_autoencoder_ckpt_name"],
                    ),
                    map_location=device,
                    weights_only=True,
                ),
                strict=True,
            )

    # Specify training parameters
    num_epochs = train_config["ldm_epochs"]
    optimizer = Adam(model.parameters(), lr=train_config["ldm_lr"])
    criterion = torch.nn.MSELoss()

    # Run training
    if not im_dataset.use_latents:
        for param in vae.parameters():
            param.requires_grad = False

    for epoch_idx in range(num_epochs):
        losses = []
        for im, _ in tqdm(data_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            if not im_dataset.use_latents:
                with torch.no_grad():
                    im, _ = vae.encode(im)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config["num_timesteps"], (im.shape[0],)).to(
                device
            )

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

        torch.save(
            model.state_dict(),
            os.path.join(train_config["task_name"], train_config["ldm_ckpt_name"]),
        )

    print("Done Training ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    parser.add_argument(
        "--config", dest="config_path", default="config/mnist.yaml", type=str
    )
    args = parser.parse_args()
    train(args)
