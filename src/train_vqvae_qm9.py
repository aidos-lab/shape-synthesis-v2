import argparse

import torch
from lightning import seed_everything
from lightning.fabric import Fabric
from torch.optim import Adam
from tqdm import tqdm

from configs import load_config
from datasets.qm9 import get_dataloaders
from datasets.transforms import get_transform
from models.discriminator import Discriminator
from models.lpips import LPIPS
from models.vqvae import VQVAE

# Global settings.
torch.set_float32_matmul_precision("medium")
fabric = Fabric(accelerator="cuda", precision="bf16-mixed")


def train(
    config,
    dataloader,
    transform,
    model,
    lpips_model,
    discriminator,
    optimizer_d,
    optimizer_g,
    recon_criterion,
    disc_criterion,
):
    step_count = 0
    for epoch_idx in range(config.train.autoencoder_epochs):
        optimizer_g.zero_grad(set_to_none=False)
        optimizer_d.zero_grad(set_to_none=True)
        for pc, _ in tqdm(dataloader):
            step_count += 1
            # Start adding the discrimminator after 1k steps.
            disc_scale_loss = 0
            if step_count > 1000:
                disc_scale_loss = 1

            ect = transform(pc).unsqueeze(1)

            # Fetch autoencoders output(reconstructions)
            output, _, quantize_losses = model(ect)

            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, ect)
            g_loss = (
                recon_loss
                + (config.train.codebook_weight * quantize_losses["codebook_loss"])
                + (config.train.commitment_beta * quantize_losses["commitment_loss"])
            )

            # Adversarial loss only if disc_step_start steps passed
            # if step_count > disc_step_start:
            disc_fake_pred = discriminator(output)
            disc_fake_loss = disc_criterion(
                disc_fake_pred,
                torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device),
            )
            g_loss += disc_scale_loss * config.train.disc_weight * disc_fake_loss

            lpips_loss = torch.mean(lpips_model(output, ect))
            g_loss += config.train.perceptual_weight * lpips_loss
            fabric.backward(g_loss)
            #####################################

            ######### Optimize Discriminator #######
            fake = output
            disc_fake_pred = discriminator(fake.detach())
            disc_real_pred = discriminator(ect)
            disc_fake_loss = disc_criterion(
                disc_fake_pred,
                torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device),
            )
            disc_real_loss = disc_criterion(
                disc_real_pred,
                torch.ones(disc_real_pred.shape, device=disc_real_pred.device),
            )
            disc_loss = config.train.disc_weight * (disc_fake_loss + disc_real_loss) / 2
            fabric.backward(disc_loss)

            optimizer_d.step()
            optimizer_d.zero_grad(set_to_none=True)
            optimizer_g.step()
            optimizer_g.zero_grad()

        state = {"model": model}
        fabric.save("trained_models/vqvae.ckpt", state)


def main(args):
    # Parse the args
    config_path = args.config_path
    compile: bool = args.compile
    dev = args.dev

    config, _ = load_config(config_path)
    seed_everything(config.train.seed)

    ###################################################################
    ### Setup models
    ###################################################################

    dataloader, _ = get_dataloaders(config.data, dev=dev)
    dataloader = fabric.setup_dataloaders(dataloader)

    # Transforms an ect to an image at runtime.
    transform = get_transform(config.transform)
    if compile:
        transform = torch.compile(transform)
    transform = fabric.setup_module(transform)

    # Create the model and dataset.
    model = VQVAE(config.vae)
    if compile:
        model = torch.compile(model)
    optimizer_g = Adam(
        model.parameters(),
        lr=config.train.autoencoder_lr,
        betas=(0.5, 0.999),
    )
    model, optimizer_g = fabric.setup(model, optimizer_g)

    ############################################################
    ### Discrimminators
    ############################################################

    discriminator = Discriminator(config.discriminator)
    if compile:
        discriminator = torch.compile(discriminator)

    optimizer_d = Adam(
        discriminator.parameters(),
        lr=config.train.autoencoder_lr,
        betas=(0.5, 0.999),
    )
    discriminator, optimizer_d = fabric.setup(discriminator, optimizer_d)

    ############################################################
    ### LPIPS
    ############################################################

    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval()
    lpips_model = fabric.setup(lpips_model)

    ############################################################
    ### Loss functions
    ############################################################

    # L1/L2 loss for Reconstruction
    loss_fn_recon = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    loss_fn_disc = torch.nn.MSELoss()

    ##########################################################
    ### Start the training.
    ##########################################################

    train(
        config,
        dataloader,
        transform,
        model,
        lpips_model,
        discriminator,
        optimizer_d,
        optimizer_g,
        loss_fn_recon,
        loss_fn_disc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for vq vae training")
    parser.add_argument(
        "--config", dest="config_path", default="config/qm9.yaml", type=str
    )
    parser.add_argument(
        "--compile", default=False, action="store_true", help="Compile all the models"
    )
    parser.add_argument(
        "--dev", default=False, action="store_true", help="Run a small subset."
    )
    args = parser.parse_args()
    main(args)
