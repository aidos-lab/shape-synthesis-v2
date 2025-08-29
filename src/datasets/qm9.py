import os
from dataclasses import dataclass

import torch
from torch_geometric.datasets import QM9
from torch.utils.data import Dataset, DataLoader, TensorDataset
import glob

import numpy as np
from PIL import Image
import json

from src.datasets.transforms import (
    EctChannelsTransform,
    EctTransformConfig,
    To3DNormalizedCoords,
)


@dataclass
class DataConfig:
    root: str
    raw: str
    batch_size: int
    resolution: int


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
    print("[qm9] PNG export uses dataset-wide per-channel μ/σ (computed on train)")

    os.makedirs(path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)

    img_root = os.path.join(path, "images")
    init_root = os.path.join(path, "init_images")
    os.makedirs(os.path.join(img_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(img_root, "test"), exist_ok=True)
    os.makedirs(os.path.join(img_root, "val"), exist_ok=True)
    os.makedirs(os.path.join(init_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(init_root, "test"), exist_ok=True)
    os.makedirs(os.path.join(init_root, "val"), exist_ok=True)

    torch.manual_seed(1337)

    # Download the full QM9 dataset.
    dataset = QM9(
        root=raw_path,
        force_reload=force_reload,
    ).shuffle()

    tr = To3DNormalizedCoords()
    ect_config = EctTransformConfig(
        num_thetas=config.resolution,
        resolution=config.resolution,
        r=1,
        scale=500,
        ambient_dimension=3,
        ect_type="points",
        seed=129,
        normalized=True,
        structured_directions=True,
        max_channels=5,
    )
    ect_tr = EctChannelsTransform(ect_config).to(device="cuda")
    res = []
    pts = []
    batch = []

    for idx, data in enumerate(dataset):
        data_new = tr(data)
        z = data.z
        z[z == 1] = 0
        z[z == 6] = 1
        z[z == 7] = 2
        z[z == 8] = 3
        z[z == 9] = 4

        ects_full = ect_tr(
            data_new.pos.cuda(),
            index=torch.zeros(len(data_new.pos), dtype=torch.int64).cuda(),
            channels=z.cuda(),
        )
        # Accept [B,C,H,W] or [C,H,W]
        if ects_full.dim() == 4:
            ects_full = ects_full[0]
        # Extract C,N,O channels (indices 1..3), keep as float32 CPU
        ects_cno = ects_full[1:4].to(torch.float32).cpu()  # [3,H,W]
        # Optional scale to [-1,1] not needed for PNG; tensors saved later remain in this normalized-by-amax space from the transform.

        # Collect for later export and stats
        res.append(ects_cno)
        pts.append(data_new.pos)
        batch.append(data.batch)
        if dev and idx == 256:
            break

    transformed = torch.stack(res, dim=0)  # [N, 3, H, W] now
    N, C, H, W = transformed.shape
    # Split indices first so we compute stats on train only
    total_el = N
    num_train = int(0.7 * total_el)
    num_test = int(0.2 * total_el)
    num_val = total_el - num_train - num_test

    qm9_train = transformed[:num_train]
    qm9_test = transformed[num_train : num_train + num_test]
    qm9_val = transformed[-num_val:]

    # Compute per-channel mean/std on TRAIN set only
    flat = qm9_train.view(num_train, C, -1)
    mu = flat.mean(dim=(0, 2))              # [3]
    std = flat.std(dim=(0, 2), unbiased=False)  # [3]
    std[std < 1e-8] = 1.0
    mu_np = mu.tolist(); std_np = std.tolist()

    # Sanity check shapes
    assert qm9_train.dim() == 4 and qm9_train.size(1) == 3, f"Unexpected train shape: {qm9_train.shape}"
    assert qm9_test.dim() == 4 and qm9_test.size(1) == 3, f"Unexpected test shape: {qm9_test.shape}"
    assert qm9_val.dim() == 4 and qm9_val.size(1) == 3, f"Unexpected val shape: {qm9_val.shape}"
    assert len(mu_np) == 3 and len(std_np) == 3, "mu/std must be 3-dim"

    # Save paired PNG images to disk for img2img fine-tuning with dataset-wide norm
    def save_split(tensor_split, indices, split_name):
        # tensor_split is the FULL transformed tensor; we index by global indices
        for i in indices:
            x = transformed[i]  # [3,H,W]
            # Standardize by train μ/σ, clamp to [-3,3], map to [0,1]
            x_std = (x - mu.view(3,1,1)) / std.view(3,1,1)
            x_std = x_std.clamp(-3.0, 3.0)
            x01 = (x_std + 3.0) / 6.0
            img_np = (x01.clamp(0,1).permute(1,2,0).mul(255).byte().numpy())
            fname = f"{i:06d}.png"
            Image.fromarray(img_np).save(os.path.join(img_root, split_name, fname))
            Image.fromarray(img_np).save(os.path.join(init_root, split_name, fname))

    train_indices = list(range(0, num_train))
    test_indices = list(range(num_train, num_train + num_test))
    val_indices = list(range(total_el - num_val, total_el))

    save_split(transformed, train_indices, "train")
    save_split(transformed, test_indices, "test")
    save_split(transformed, val_indices, "val")

    torch.save(qm9_train, f"{path}/train.pt")
    torch.save(qm9_val, f"{path}/val.pt")
    torch.save(qm9_test, f"{path}/test.pt")

    stats = {"mu": mu_np, "std": std_np, "channels": ["C","N","O"], "note": "dataset-wide per-channel mean/std computed on train split"}
    with open(os.path.join(path, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization stats to {os.path.join(path, 'norm_stats.json')}")

    print(f"Saved RGB images under {img_root} and {init_root} (train/test/val)")


def get_dataloaders(config: DataConfig, dev: bool = False):
    """Returns two dataloaders, train and test."""

    dataset_type = "dev" if dev else "prod"
    path = f"{config.root}/qm9/{dataset_type}"

    train_ds = TensorDataset(torch.load(f"{path}/train.pt").cpu())
    val_ds = TensorDataset(torch.load(f"{path}/val.pt").cpu())
    test_ds = TensorDataset(torch.load(f"{path}/test.pt").cpu())

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=not dev,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    return train_dl, val_dl, test_dl


def _resize_pil(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.BICUBIC)

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.uint8, copy=True)
    ten = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return ten


class ImagePairFolderDataset(Dataset):
    def __init__(self, root: str, split: str, resolution: int):
        self.init_dir = os.path.join(root, "init_images", split)
        self.tgt_dir  = os.path.join(root, "images", split)
        self.resolution = resolution   # <— add this

        init_paths = sorted(glob.glob(os.path.join(self.init_dir, "*.png")))
        tgt_paths  = sorted(glob.glob(os.path.join(self.tgt_dir,  "*.png")))
        init_map = {os.path.basename(p): p for p in init_paths}
        self.pairs = [(init_map[name], p) for p in tgt_paths if (name := os.path.basename(p)) in init_map]
        if not self.pairs:
            raise RuntimeError(f"No pairs found under {self.init_dir} and {self.tgt_dir}")

        # remove self.tf definition entirely

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        init_path, tgt_path = self.pairs[idx]
        init_img = Image.open(init_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")

        init_img = _resize_pil(init_img, self.resolution)
        tgt_img = _resize_pil(tgt_img, self.resolution)

        init_t = _pil_to_tensor(init_img)  # [3,H,W] in [0,1]
        tgt_t = _pil_to_tensor(tgt_img)

        return {"init": init_t, "target": tgt_t, "path": os.path.basename(tgt_path)}


def get_img2img_dataloaders(config: DataConfig, dev: bool = False):
    """Returns train/test dataloaders for the exported PNG pairs (for SD img2img FT)."""
    dataset_type = "dev" if dev else "prod"
    root = f"{config.root}/qm9/{dataset_type}"
    train_ds = ImagePairFolderDataset(root, "train", config.resolution)
    test_ds  = ImagePairFolderDataset(root, "test", config.resolution)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=not dev,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return train_dl, test_dl


if __name__ == "__main__":
    config = DataConfig(root="./data", raw="./data/raw", batch_size=64, resolution=256)
    #create_dataset(config, dev=True, force_reload=False)
    create_dataset(config, dev=False, force_reload=False)
