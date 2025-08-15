from torchmetrics.regression import MeanSquaredError


def get_mse_metrics():
    return (
        MeanSquaredError(),
        MeanSquaredError(),
        MeanSquaredError(),
    )
