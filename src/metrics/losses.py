import torch
import torch.nn.functional as F


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)
    return result_tensor


def gaussian_nll(mu, log_sigma, x):
    return (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * torch.log(2 * torch.tensor(torch.pi))
    )


def loss_function(x_hat, x, mu, logvar):
    log_sigma = ((x - x_hat) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()
    log_sigma = softclip(log_sigma, -6)
    rec = gaussian_nll(x_hat, log_sigma, x).sum()
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rec, kl


def vanilla_loss_function(x, x_hat, mean, log_var):
    rec = F.mse_loss(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return rec, kl
