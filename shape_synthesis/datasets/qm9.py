import os
from dataclasses import dataclass

import torch
from torch_geometric.datasets import QM9


from shape_synthesis.configs import print_config, save_config


@dataclass
class DataConfig:
    root: str
    raw: str
    batch_size: int


def create_dataset(config: DataConfig, dev: bool = False):
    """
    Create the datasets for processing. Creates either
    the dev dataset or the full dataset.
    """

    dataset_type = "dev" if dev else "prod"

    path = f"{config.root}/qm9/{dataset_type}"
    raw_path = f"{config.raw}/qm9"
    print(dataset_type)
    print("Creating:", os.path.dirname(path))
    print("Creating:", os.path.dirname(raw_path))

    os.makedirs(path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)

    torch.manual_seed("1337")

    # Download the full QM9 dataset.
    dataset = QM9("data/QM9").shuffle()

    qm9_train = dataset[:110_000]
    qm9_val = dataset[110_000:120_000]
    qm9_test = dataset[120_000:]

    if dev:
        qm9_train = torch.utils.data.Subset(qm9_train, torch.arange(0, 64))
        qm9_val = torch.utils.data.Subset(qm9_val, torch.arange(0, 64))
        qm9_test = torch.utils.data.Subset(qm9_test, torch.arange(0, 64))

    torch.save(qm9_train, f"{path}/train.pt")
    torch.save(qm9_val, f"{path}/val.pt")
    torch.save(qm9_test, f"{path}/test.pt")

    save_config(config=config, path=f"{path}/config.yaml")


if __name__ == "__main__":
    config = DataConfig(
        root="./data",
        raw="./data/raw",
        batch_size=64,
    )
    create_dataset(config, dev=True)
    create_dataset(config, dev=False)

    # print(72 * "=")
    # print("Data Configuration")
    # print_config(config)
    # print(72 * "=")

    # get_dataloaders(config, dev=False)
    # get_dataloaders(config, dev=True)
