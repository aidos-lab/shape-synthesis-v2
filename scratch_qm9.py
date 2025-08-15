import matplotlib.pyplot as plt

from src.datasets.qm9 import DataConfig, get_dataloaders

config = DataConfig(
    root="./data",
    raw="./data/raw",
    batch_size=64,
)
train_dl, _ = get_dataloaders(config, dev=False)


for (batch,) in train_dl:
    print(batch.shape)
#
# # |%%--%%| <jBjrkN8WBb|O8MjtA7Kp7>
