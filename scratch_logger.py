import torch
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger

# logger = CSVLogger("./logs/scratch", name="my_model")
logger = TensorBoardLogger("./logs/scratch", name="my_model")

for i in range(10):
    logger.log_metrics({"epoch": i, "loss": 0.235, "acc": 0.75})
    logger.experiment.add_images(
        "imgage", torch.rand(10, 1, 28, 28), dataformats="NCHW"
    )
logger.finalize("success")
