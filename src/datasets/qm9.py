import os
from dataclasses import dataclass

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from src.datasets.transforms import To3DNormalizedCoords, get_transform


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
        root=raw_path,
        force_reload=force_reload,
    ).shuffle()

    tr = To3DNormalizedCoords()
    ect_tr = get_transform()
    res = []
    for idx, data in enumerate(dataset):
        data_new = tr(data)
        ects = ect_tr(data_new.pos.cuda(), index=None).cpu()
        res.append(ects)
        if dev and idx == 256:
            break

    transformed = torch.stack(res)

    total_el = len(transformed)
    num_train = int(0.7 * total_el)
    num_test = int(0.2 * total_el)
    num_val = total_el - num_train - num_test

    qm9_train = transformed[:num_train]
    qm9_test = transformed[num_train : num_train + num_test]
    qm9_val = transformed[-num_val:]

    torch.save(qm9_train, f"{path}/train.pt")
    torch.save(qm9_val, f"{path}/val.pt")
    torch.save(qm9_test, f"{path}/test.pt")


def get_dataloaders(config: DataConfig, dev: bool = False):
    """Returns two dataloaders, train and test."""

    dataset_type = "dev" if dev else "prod"
    path = f"{config.root}/qm9/{dataset_type}"

    train_ds = torch.load(f"{path}/train.pt", weights_only=True).cpu()
    # val_ds = torch.load(f"{path}/val.pt", weights_only=True)
    test_ds = torch.load(f"{path}/test.pt", weights_only=True).cpu()

    train_ds = torch.utils.data.TensorDataset(train_ds)
    test_ds = torch.utils.data.TensorDataset(test_ds)

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
    create_dataset(config, dev=True, force_reload=False)
    create_dataset(config, dev=False, force_reload=False)
