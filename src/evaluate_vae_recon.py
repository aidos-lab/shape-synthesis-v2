#!/usr/bin/env python3
import os, math, argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import csv

from diffusers import AutoencoderKL

# ---- IO helpers ----
def list_images(path: Path, exts=(".png", ".jpg", ".jpeg", ".bmp")) -> List[Path]:
    if path.is_file():
        return [path]
    return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in exts]

def load_rgb_01(p: Path, target_res: int = 0) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    if target_res > 0 and (img.width != target_res or img.height != target_res):
        img = img.resize((target_res, target_res), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)
    ten = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # [3,H,W] in [0,1]
    return ten

# ---- SSIM (PyTorch, Gaussian window) ----
def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device="cpu"):
    coords = torch.arange(window_size, device=device).float() - (window_size - 1)/2
    gauss = torch.exp(-(coords**2)/(2*sigma*sigma))
    gauss = gauss / gauss.sum()
    w1d = gauss.view(1,1,1,-1)
    window = (w1d.transpose(2,3) @ w1d).squeeze(0).squeeze(0)  # [k,k]
    return window

def ssim_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    x,y: [B,3,H,W] in [0,1]
    returns: SSIM per-image [B]
    """
    C1 = (0.01)**2
    C2 = (0.03)**2
    b, c, h, w = x.shape
    device = x.device
    win = _gaussian_window(window_size, sigma, device=device).view(1,1,window_size,window_size)
    win = win.repeat(c, 1, 1, 1)  # [C,1,k,k] for groups conv

    def _conv(img):
        return torch.conv2d(img, win, padding=window_size//2, groups=c)

    mu_x  = _conv(x)
    mu_y  = _conv(y)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _conv(x * x) - mu_x2
    sigma_y2 = _conv(y * y) - mu_y2
    sigma_xy = _conv(x * y) - mu_xy

    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2))
    # average over H,W and channels, then over batch -> return per-image
    ssim_per_img = ssim_map.mean(dim=[1,2,3])
    return ssim_per_img

# ---- PSNR ----
def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    # mse: [B], images in [0,1]
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))

# ---- VAE round-trip ----
@torch.no_grad()
def vae_roundtrip(vae: AutoencoderKL, x01: torch.Tensor, scale: float = 0.18215) -> torch.Tensor:
    """
    x01: [B,3,H,W] in [0,1] -> returns recon in [0,1]
    deterministic: uses posterior mode for encoding
    """
    x_in = x01 * 2.0 - 1.0
    enc = vae.encode(x_in)                    # DiagonalGaussianDistribution
    z   = enc.latent_dist.mode() * scale      # [B,4,h,w]
    xhat = vae.decode(z / scale).sample       # [-1,1]
    xhat01 = (xhat * 0.5 + 0.5).clamp(0.0, 1.0)
    return xhat01

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, required=True, help="Path to a PNG or a directory")
    ap.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="SD checkpoint to pull VAE from")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=0, help="If >0, resize to square of this size before eval")
    ap.add_argument("--csv", type=str, default="", help="Optional path to write per-image metrics CSV")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true", help="Use float16 on GPU")
    args = ap.parse_args()

    paths = list_images(Path(args.images))
    if not paths:
        raise SystemExit(f"No images found under {args.images}")

    device = torch.device(args.device)
    dtype = torch.float16 if (device.type == "cuda" and args.fp16) else torch.float32

    print(f"[eval] Loading VAE from {args.model_id}/vae on {device} ({dtype})")
    load_root = Path(args.model_id)
    load_path = load_root / "vae"
    if not load_path.is_dir():
        load_path = load_root
    print(f"[eval] Loading VAE from {load_path}")
    vae = AutoencoderKL.from_pretrained(str(load_path))
    vae = vae.to(device, dtype=dtype)
    vae.eval()

    # batching
    B = args.batch_size
    mse_all, psnr_all, ssim_all = [], [], []
    rows: List[Tuple[str,float,float,float]] = []

    for i in range(0, len(paths), B):
        batch_paths = paths[i:i+B]
        x_list = [load_rgb_01(p, args.resolution) for p in batch_paths]
        x = torch.stack(x_list, dim=0).to(device=device, dtype=dtype)  # [B,3,H,W]

        with torch.no_grad():
            xhat = vae_roundtrip(vae, x)

        # metrics on float32 CPU
        x_cpu    = x.detach().float().cpu()
        xhat_cpu = xhat.detach().float().cpu()

        mse = F.mse_loss(xhat_cpu, x_cpu, reduction="none").mean(dim=[1,2,3])  # [B]
        psnr = psnr_from_mse(mse)
        ssim = ssim_torch(x_cpu, xhat_cpu)

        mse_all.append(mse)
        psnr_all.append(psnr)
        ssim_all.append(ssim)

        for j, p in enumerate(batch_paths):
            rows.append((str(p), float(mse[j]), float(psnr[j]), float(ssim[j])))

        if (i // B) % 10 == 0:
            print(f"[eval] {i + len(batch_paths)}/{len(paths)} done")

    mse_all  = torch.cat(mse_all)
    psnr_all = torch.cat(psnr_all)
    ssim_all = torch.cat(ssim_all)

    def summarize(name, t: torch.Tensor):
        q = torch.quantile(t, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
        print(f"{name}: mean={t.mean():.6f} | q=[{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}, {q[4]:.6f}]")

    print("\n=== VAE Reconstruction Quality ===")
    summarize("MSE",  mse_all)
    summarize("PSNR (dB)", psnr_all)
    summarize("SSIM", ssim_all)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "mse", "psnr_db", "ssim"])
            for row in rows:
                w.writerow(row)
        print(f"\nWrote per-image metrics to {args.csv}")

if __name__ == "__main__":
    main()