import argparse

import torch
from lightning import seed_everything
from lightning.fabric import Fabric
from torch.optim import Adam
from tqdm import tqdm

from configs import load_config
from datasets.mnist import get_dataloaders
from datasets.transforms import get_transform
from models.discriminator import Discriminator
from models.lpips import LPIPS
from models.vqvae import VQVAE

# Global settings.
torch.set_float32_matmul_precision("medium")

# |%%--%%| <NPqJC3R2WA|HPAqxzb4rh>


def evaluate(
    config,
    dataloader,
    transform,
    model,
):
    model.eval()

    for pc, _ in tqdm(dataloader):
        ect = transform(pc).unsqueeze(1)

        # Fetch autoencoders output(reconstructions)
        output, _, quantize_losses = model(ect)

        torch.save(output.detach().cpu(), "results/recon.pt")
        torch.save(ect.cpu(), "results/ect.pt")
        torch.save(pc.cpu(), "results/pc.pt")

        break


def main(args):

    fabric = Fabric()
    # Parse the args
    config_path = args.config_path
    compile: bool = args.compile
    dev = args.dev

    config, _ = load_config(config_path)
    seed_everything(config.train.seed)

    ###################################################################
    ### Setup models
    ###################################################################

    _, dataloader = get_dataloaders(config.data, dev=dev)
    dataloader = fabric.setup_dataloaders(dataloader)

    # Transforms an ect to an image at runtime.
    transform = get_transform(config.transform)
    if compile:
        transform = torch.compile(transform)
    transform = fabric.setup_module(transform)

    # Load the model
    model = VQVAE(config.vae).to("cuda")
    state = {"model": model}
    fabric.load("trained_models/vqvae.ckpt", state)

    ##########################################################
    ### Start the evaluation.
    ##########################################################

    evaluate(
        config,
        dataloader,
        transform,
        model,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for vq vae training")
    parser.add_argument(
        "--config", dest="config_path", default="config/mnist.yaml", type=str
    )
    parser.add_argument(
        "--compile", default=False, action="store_true", help="Compile all the models"
    )
    parser.add_argument(
        "--dev", default=False, action="store_true", help="Run a small subset."
    )
    args = parser.parse_args()
    main(args)
