import os
from dataclasses import dataclass

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from shape_synthesis.configs import print_config, save_config


@dataclass
class DataConfig:
    module: str
    root: str
    raw: str
    num_pts: int
    batch_size: int


def sample_point_cloud_from_image(dataset, config: DataConfig, dev: bool = False):
    num_tries = 10
    # Transform the train set.
    full_point_cloud = torch.empty(size=(len(dataset), config.num_pts, 2))
    imgs = torch.empty(size=(len(dataset), 28, 28))
    for idx, (img, _) in enumerate(dataset):
        if dev and idx > 64:
            break

        img = img.squeeze()
        imgs[idx] = img
        # Transform so the point cloud is with the "right side up".
        img = torch.rot90(img.flipud(), 3)

        point_cloud = torch.empty(size=(config.num_pts, 2))
        for i in range(num_tries):
            # Create large vector of samples, this may fail.
            XY = torch.rand(size=(40 * config.num_pts, 2))
            Z = torch.rand(size=(40 * config.num_pts,))

            XY_idx = torch.floor(28 * XY).to(torch.int)
            mask = Z < img[XY_idx[:, 0], XY_idx[:, 1]]
            point_cloud = XY[mask]
            # Transform so the point cloud is with the "right side up".
            if len(point_cloud) >= config.num_pts:
                full_point_cloud[idx, :, :] = point_cloud[: config.num_pts, :]
                break
            else:
                print(idx, "Num tries left", num_tries - i)
                if i == num_tries - 1:
                    raise ValueError()
    return full_point_cloud, imgs


def create_dataset(config: DataConfig, dev: bool = False):
    """
    Create the datasets for processing. Creates either
    the dev dataset or the full dataset.
    """

    dataset_type = "dev" if dev else "prod"

    path = f"{config.root}/mnist/{dataset_type}"
    raw_path = f"{config.raw}/mnist"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    # Download the full MNIST dataset.
    mnist_train = MNIST(root=raw_path, transform=ToTensor(), train=True, download=True)
    mnist_test = MNIST(root=raw_path, transform=ToTensor(), train=False, download=True)

    if dev:
        mnist_train = torch.utils.data.Subset(mnist_train, torch.arange(0, 64))
        mnist_test = torch.utils.data.Subset(mnist_test, torch.arange(0, 64))

    train_point_cloud, train_imgs = sample_point_cloud_from_image(
        mnist_train, config=config
    )
    test_point_cloud, test_imgs = sample_point_cloud_from_image(
        mnist_test, config=config
    )

    torch.save(train_point_cloud, f"{path}/train.pt")
    torch.save(test_point_cloud, f"{path}/test.pt")
    torch.save(train_imgs, f"{path}/train_imgs.pt")
    torch.save(test_imgs, f"{path}/test_imgs.pt")

    save_config(config=config, path=f"{path}/config.yaml")


def get_dataloaders(config: DataConfig, dev: bool = False):
    """Returns two dataloaders, train and test."""

    dataset_type = "dev" if dev else "prod"
    path = f"{config.root}/mnist/{dataset_type}"

    train_point_cloud = torch.load(f"{path}/train.pt")
    train_imgs = torch.load(f"{path}/train_imgs.pt")
    test_point_cloud = torch.load(f"{path}/test.pt")
    test_imgs = torch.load(f"{path}/test_imgs.pt")

    train_ds = torch.utils.data.TensorDataset(train_point_cloud, train_imgs)
    test_ds = torch.utils.data.TensorDataset(test_point_cloud, test_imgs)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    test_dl = torch.utils.data.DataLoader(
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
        num_pts=256,
        module="datasets.mnist",
        batch_size=32,
    )
    # create_dataset(config, dev=True)
    # create_dataset(config, dev=False)

    # print(72 * "=")
    # print("Data Configuration")
    # print_config(config)
    # print(72 * "=")

    # get_dataloaders(config, dev=False)
    # get_dataloaders(config, dev=True)
