from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn.functional as F
from kaolin.metrics.pointcloud import chamfer_distance
from torchmetrics.regression import KLDivergence

Tensor: TypeAlias = torch.Tensor


def compute_mse_loss_fn(ect_hat, ect):
    pixelwise = F.mse_loss(ect_hat, ect)
    return pixelwise


def compute_mse_kld_loss_beta_annealing_fn(
    decoded_ect: Tensor,
    z_mean: Tensor,
    z_log_var: Tensor,
    ect: Tensor,
    current_epoch: int,
    period: int,
    beta_min: float,
    beta_max: float,
    prefix: str,
) -> dict[str, Tensor]:
    """
    Computes an annealed schedule for the KL Loss and the MSE Loss.
    It starts with a pure reconstruction loss and cycles every max-epochs.
    """

    beta = (beta_max - beta_min) * (
        1 - torch.cos(2 * torch.pi * torch.tensor(current_epoch - 100) / period)
    ) / 2 + beta_min

    # First focus on recon loss.
    if current_epoch < 100:
        beta = beta_min

    # # beta = 0.0005
    beta = 0.01

    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + z_log_var - z_mean**2 - z_log_var.exp(), dim=1), dim=0
    )

    mse_loss = F.mse_loss(decoded_ect, ect, reduction="mean")

    return {
        f"{prefix}loss": mse_loss + beta * kld_loss,
        f"{prefix}kld_loss": kld_loss,
        f"{prefix}ect_loss": mse_loss,
        "beta": beta,
    }


# I am not sure if I used mean or sum here.
def compute_mse_kld_loss_fn(decoded, mu, log_var, ect, beta=0.0):
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )

    mse_loss = F.mse_loss(decoded, ect, reduction="mean")

    return mse_loss + beta * kld_loss, kld_loss, mse_loss


def chamfer3D(pred_pc, ref_pc):
    return chamfer_distance(pred_pc, ref_pc).mean()


def chamfer2D(pred_pc, ref_pc):
    pred_pc = F.pad(input=pred_pc, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0)
    ref_pc = F.pad(input=ref_pc, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0)
    return chamfer_distance(pred_pc, ref_pc).mean()


def chamfer3DECT(pred_pc, ref_pc, ect_pred, ect):
    ch_loss = chamfer3D(pred_pc, ref_pc)
    mse_loss = F.mse_loss(ect_pred, ect)
    return 100 * (ch_loss + 10 * mse_loss), mse_loss, ch_loss


def chamfer2DECT(pred_pc, ref_pc, ect_pred, ect):
    ch_loss = chamfer2D(pred_pc, ref_pc)
    mse_loss = F.mse_loss(ect_pred, ect)
    return ch_loss + 10 * mse_loss, mse_loss, ch_loss


def dect_loss_fn(ect_pred, ect):
    mse_loss = F.mse_loss(ect_pred, ect)
    return mse_loss
