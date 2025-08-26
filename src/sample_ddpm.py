import argparse
import os

import torch
import torchvision
import yaml
from lightning.fabric import Fabric
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm

from configs import load_config
from src.models.unet import Unet
from src.models.vqvae import VQVAE
from src.schedulers.linear_scheduler import LinearNoiseScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main(args):

    # Parse the args
    config_path = args.config_path

    # Set up Fabric
    fabric = Fabric()
    config, _ = load_config(config_path)

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(config=config.noise_scheduler)

    # Instantiate the model
    model = Unet(
        im_channels=config.vae.z_channels,
        model_config=config.ldm_params,
    ).to(device)

    fabric.load("trained_models/latent_ddpm.ckpt")

    # Load the model
    vae = VQVAE(config.vae).to("cuda")

    state = {"model": vae}
    fabric.load("trained_models/vqvae.ckpt", state)
    vae.eval()

    im_size = config.transform.resolution // 2 ** sum(config.vae.down_sample)
    xt = torch.randn(
        (config.train.num_samples, config.vae.z_channels, im_size, im_size)
    ).to(device)

    save_count = 0
    for i in tqdm(reversed(range(config.noise_scheduler.num_timesteps))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(
            xt, noise_pred, torch.as_tensor(i).to(device)
        )

        # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode ONLY the final iamge to save time
            ims = vae.decode(xt)
        else:
            ims = xt

        ims = torch.clamp(ims, -1.0, 1.0).detach().cpu()
        grid = make_grid(ims, nrow=config.train.num_grid_rows)
        img = torchvision.transforms.ToPILImage()(grid[:3, :, :])
        img.save(f"results/generated_ects_{i}.png")
        img.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm image generation")
    parser.add_argument(
        "--config", dest="config_path", default="config/qm9.yaml", type=str
    )
    args = parser.parse_args()
    main(args)
