import torch


def compute_mse_accuracies(metrics, y_hat, y, name):
    return [{"name": name, "value": metrics(y_hat, y)}]
