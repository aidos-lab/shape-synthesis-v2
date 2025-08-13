import os
from torch_geometric.loader import DataLoader
from dataclasses import dataclass

import torch
from torch_geometric.datasets import QM9

from src.datasets.transforms import To3DNormalizedCoords


from src.configs import print_config, save_config


@dataclass
class DataConfig:
    root: str
    raw: str
    batch_size: int


def create_dataset(config: DataConfig, dev: bool = False, force_reload=False):
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
    dataset = QM9(
        root=path,
        pre_transform=To3DNormalizedCoords(),
        force_reload=force_reload,
    ).shuffle()

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


def get_dataloaders(config: DataConfig, dev: bool = False):
    """Returns two dataloaders, train and test."""

    dataset_type = "dev" if dev else "prod"
    path = f"{config.root}/qm9/{dataset_type}"

    train_ds = torch.load(f"{path}/train.pt")
    val_ds = torch.load(f"{path}/val.pt")
    test_ds = torch.load(f"{path}/test.pt")

    # train_ds = torch.utils.data.TensorDataset(train_ds)
    # test_ds = torch.utils.data.TensorDataset(test_ds)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True if not dev else False,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_dl, test_dl


if __name__ == "__main__":
    config = DataConfig(
        root="./data",
        raw="./data/raw",
        batch_size=64,
    )
    create_dataset(config, dev=True, force_reload=True)
    create_dataset(config, dev=False, force_reload=True)
