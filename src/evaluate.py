import argparse

import torch
from lightning import seed_everything
from lightning.fabric import Fabric
from tqdm import tqdm
import torchvision

from configs import load_config
from datasets.qm9 import get_dataloaders
from datasets.transforms import get_transform
from models.vqvae import VQVAE

from torchvision.utils import make_grid

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

    for ect in tqdm(dataloader):
        ect = ect[0][:, :3, :, :]
        # Fetch autoencoders output(reconstructions)
        recon, _, quantize_losses = model(ect)

        torch.save(recon.detach().cpu(), "results/recon.pt")
        torch.save(ect.cpu(), "results/ect.pt")
        # torch.save(pc.cpu(), "results/pc.pt")

        sample_size = min(8, recon.shape[0])
        save_output = torch.clamp(recon[:sample_size], -1.0, 1.0).detach().cpu()
        save_output = (save_output + 1) / 2
        save_output = save_output[:, :3, :, :]
        save_input = ((ect[:sample_size][:, :3, :, :] + 1) / 2).detach().cpu()

        grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
        img = torchvision.transforms.ToPILImage()(grid)
        img.save("results/ect_recon.png")
        img.close()
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
