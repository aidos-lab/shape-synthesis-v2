import argparse
import os
import random

import numpy as np
import torch
import torchvision
import yaml
from dect.directions import generate_2d_directions
from dect.ect import compute_ect_point_cloud
from lightning.fabric import Fabric
from torch.optim import Adam
from torchvision.utils import make_grid
from tqdm import tqdm

from datasets.mnist import DataConfig, get_dataloaders
from datasets.transforms import get_transform
from models.discriminator import Discriminator
from models.lpips import LPIPS
from models.vqvae import VQVAE

torch.set_float32_matmul_precision("medium")

fabric = Fabric(accelerator="cuda", precision="16-mixed")


# @torch.compile
def train(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dataset_config = config["dataset_params"]
    autoencoder_config = config["autoencoder_params"]
    train_config = config["train_params"]

    # Set the desired seed value #
    seed = train_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if device == "cuda":
    #     torch.cuda.manual_seed_all(seed)
    #############################

    # Create the model and dataset #
    model = torch.compile(
        VQVAE(
            im_channels=dataset_config["im_channels"], model_config=autoencoder_config
        )
    )

    config = DataConfig(
        root="./data",
        raw="./data/raw",
        num_pts=150,
        module="datasets.mnist",
        batch_size=32,
        skeletonize=True,
    )

    data_loader_pre, _ = get_dataloaders(config, dev=True)

    transform = get_transform(compiled=False, resolution=64, d=2)
    v = generate_2d_directions(64).to("cuda")

    data_loader = fabric.setup_dataloaders(data_loader_pre)

    # Create output directories
    if not os.path.exists(train_config["task_name"]):
        os.mkdir(train_config["task_name"])

    num_epochs = train_config["autoencoder_epochs"]

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()

    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = torch.compile(LPIPS()).eval()
    discriminator = torch.compile(
        Discriminator(im_channels=dataset_config["im_channels"])
    )

    optimizer_d = Adam(
        discriminator.parameters(),
        lr=train_config["autoencoder_lr"],
        betas=(0.5, 0.999),
    )
    optimizer_g = Adam(
        model.parameters(), lr=train_config["autoencoder_lr"], betas=(0.5, 0.999)
    )

    model, optimizer_g = fabric.setup(model, optimizer_g)
    discriminator, optimizer_d = fabric.setup(discriminator, optimizer_d)
    lpips_model = fabric.setup(lpips_model)

    disc_step_start = train_config["disc_start"]
    step_count = 0

    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config["autoencoder_acc_steps"]
    image_save_steps = train_config["autoencoder_img_save_steps"]
    img_save_count = 0

    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
        # commitment_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []

        optimizer_g.zero_grad(set_to_none=False)
        optimizer_d.zero_grad(set_to_none=True)

        for pc, _ in tqdm(data_loader):

            if step_count > disc_step_start:
                disc_scale_loss = 1
            else:
                disc_scale_loss = 1

            pc = pc.cuda()

            im = compute_ect_point_cloud(
                x=pc, v=v, radius=1, scale=500, resolution=64, normalize=True
            ).unsqueeze(1)

            step_count += 1

            # Fetch autoencoders output(reconstructions)
            output, _, quantize_losses = model(im)

            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im) / acc_steps
            g_loss = (
                recon_loss
                + (
                    train_config["codebook_weight"]
                    * quantize_losses["codebook_loss"]
                    / acc_steps
                )
                + (
                    train_config["commitment_beta"]
                    * quantize_losses["commitment_loss"]
                    / acc_steps
                )
            )
            # Adversarial loss only if disc_step_start steps passed
            # if step_count > disc_step_start:
            disc_fake_pred = discriminator(output)
            disc_fake_loss = disc_criterion(
                disc_fake_pred,
                torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device),
            )
            g_loss += (
                disc_scale_loss
                * train_config["disc_weight"]
                * disc_fake_loss
                / acc_steps
            )

            lpips_loss = torch.mean(lpips_model(output, im))
            g_loss += train_config["perceptual_weight"] * lpips_loss / acc_steps
            fabric.backward(g_loss)
            #####################################

            ######### Optimize Discriminator #######
            # if step_count > disc_step_start:
            fake = output
            disc_fake_pred = discriminator(fake.detach())
            disc_real_pred = discriminator(im)
            disc_fake_loss = disc_criterion(
                disc_fake_pred,
                torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device),
            )
            disc_real_loss = disc_criterion(
                disc_real_pred,
                torch.ones(disc_real_pred.shape, device=disc_real_pred.device),
            )
            disc_loss = (
                train_config["disc_weight"] * (disc_fake_loss + disc_real_loss) / 2
            )
            # disc_losses.append(disc_loss.item())
            disc_loss = disc_scale_loss * disc_loss / acc_steps

            fabric.backward(disc_loss)
            if step_count % acc_steps == 0:
                optimizer_d.step()
                optimizer_d.zero_grad(set_to_none=True)
            #####################################

            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

        optimizer_d.step()
        optimizer_d.zero_grad(set_to_none=True)
        optimizer_g.step()
        optimizer_g.zero_grad()

        # # Save stuff.
        # sample_size = min(8, im.shape[0])
        # save_output = torch.clamp(output[:sample_size], -1.0, 1.0).detach().cpu()
        # save_output = (save_output + 1) / 2
        # save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
        #
        # torch.save(
        #     save_input,
        #     os.path.join(
        #         train_config["task_name"],
        #         "vqvae_autoencoder_samples",
        #         "ground_truth_{:04}.pt".format(img_save_count),
        #     ),
        # )
        # torch.save(
        #     save_output,
        #     os.path.join(
        #         train_config["task_name"],
        #         "vqvae_autoencoder_samples",
        #         "reconstruction_{:04}.pt".format(img_save_count),
        #     ),
        # )
        #
        # grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
        # img = torchvision.transforms.ToPILImage()(grid)
        # if not os.path.exists(
        #     os.path.join(train_config["task_name"], "vqvae_autoencoder_samples")
        # ):
        #     os.mkdir(
        #         os.path.join(train_config["task_name"], "vqvae_autoencoder_samples")
        #     )
        # img.save(
        #     os.path.join(
        #         train_config["task_name"],
        #         "vqvae_autoencoder_samples",
        #         "current_autoencoder_sample_{:04}.png".format(img_save_count),
        #     )
        # )
        # img_save_count += 1
        # img.close()
        #
        # # if len(disc_losses) > 0:
        # #     print(
        # #         "Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | "
        # #         "Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}".format(
        # #             epoch_idx + 1,
        # #             np.mean(recon_losses),
        # #             np.mean(perceptual_losses),
        # #             np.mean(codebook_losses),
        # #             np.mean(gen_losses),
        # #             np.mean(disc_losses),
        # #         )
        # #     )
        # # else:
        # #     print(
        # #         "Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}".format(
        # #             epoch_idx + 1,
        # #             np.mean(recon_losses),
        # #             np.mean(perceptual_losses),
        # #             np.mean(codebook_losses),
        # #         )
        # #     )
        #
        # # torch.save(
        # #     model.state_dict(),
        # #     os.path.join(
        # #         train_config["task_name"], train_config["vqvae_autoencoder_ckpt_name"]
        # #     ),
        # # )
        # # torch.save(
        # #     discriminator.state_dict(),
        # #     os.path.join(
        # #         train_config["task_name"], train_config["vqvae_discriminator_ckpt_name"]
        # #     ),
        # # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for vq vae training")
    parser.add_argument(
        "--config", dest="config_path", default="config/mnist.yaml", type=str
    )
    args = parser.parse_args()
    train(args)
