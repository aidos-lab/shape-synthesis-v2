# train_ect_img2img_unet.py
import os, math, argparse, random
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")  # avoids torchvision pull-in

from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

import math
import torch.nn as nn
from safetensors.torch import save_file as _safe_save


# ---------- Simple PNG folder dataset ----------
class PNGFolder(Dataset):
    def __init__(self, root_images_dir: str, resolution: int = 256, ext: str = ".png", limit: int = 0):
        self.root = Path(root_images_dir)
        self.res = resolution
        self.files = sorted([p for p in self.root.glob(f"*{ext}")])
        if limit > 0:
            self.files = self.files[:limit]
        if not self.files:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self): return len(self.files)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        # writable array
        arr = np.array(img, dtype=np.uint8, copy=True)
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if img.size != (self.res, self.res):
            img = img.resize((self.res, self.res), resample=Image.BICUBIC)
        ten = self._pil_to_tensor(img)  # [3,H,W] in [0,1]
        return ten


def make_grid(images: torch.Tensor, nrow: int = 4):
    """images: [B,3,H,W] in [0,1] -> [3,H*,W*] grid"""
    # Simple grid to avoid torchvision dependency
    b, c, h, w = images.shape
    nrow = min(nrow, b)
    ncol = (b + nrow - 1) // nrow
    grid = torch.zeros(c, ncol * h, nrow * w, dtype=images.dtype, device=images.device)
    for i in range(b):
        r = i // nrow
        cl = i % nrow
        grid[:, r*h:(r+1)*h, cl*w:(cl+1)*w] = images[i]
    return grid.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="./data/qm9/dev/images/train",
                    help="Folder with target ECT PNGs (RGB=CNO)")
    ap.add_argument("--val-root", type=str, default="./data/qm9/dev/images/test",
                    help="Folder with validation PNGs (optional)")
    ap.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--out-dir", type=str, default="./checkpoints/ect-sd15-unet")
    ap.add_argument("--log-dir", type=str, default="./runs/ect-sd15-unet")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=150_000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--fp16", action="store_true", help="mixed precision")
    ap.add_argument("--xformers", action="store_true", help="enable memory-efficient attention if installed")
    ap.add_argument("--val-limit", type=int, default=16, help="use up to N val samples for quick TB previews")

    # ===== VAE decoder fine-tuning (reconstruction mode; no LoRA) =====
    ap.add_argument("--vae-decoder-ft", action="store_true", help="If set, train VAE decoder on reconstruction loss (encoder frozen), instead of UNet noise prediction.")
    ap.add_argument("--vae-lr", type=float, default=1e-5, help="Learning rate when training VAE decoder")
    ap.add_argument("--vae-steps", type=int, default=20000, help="Max steps for VAE decoder training")
    ap.add_argument("--vae-log-every", type=int, default=200, help="Logging interval for VAE decoder mode")
    ap.add_argument("--l1-weight", type=float, default=1.0, help="Weight for L1 recon loss in VAE decoder mode")
    ap.add_argument("--ssim-weight", type=float, default=0.0, help="Weight for SSIM term (1-SSIM) in VAE decoder mode")
    ap.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for VAE decoder weights (0 disables EMA)")

    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # -------- TensorBoard --------
    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_text("hparams", str(vars(args)))

    # -------- Data --------
    train_ds = PNGFolder(args.data_root, args.resolution)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)

    # Small fixed validation batch for image previews
    val_ds = None
    val_batch = None
    if os.path.isdir(args.val_root):
        try:
            val_ds = PNGFolder(args.val_root, args.resolution, limit=max(args.val_limit, 0))
            val_loader = DataLoader(val_ds, batch_size=min(8, len(val_ds)), shuffle=False, num_workers=2, pin_memory=True)
            val_batch = next(iter(val_loader)).to(device)
        except Exception as e:
            print("[warn] could not build val set:", e)

    # -------- Model --------
    vae  = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    text = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    tok  = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)

    if args.xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print("[warn] xformers not available:", e)

    # Optional: checkpointing if you push res/batch
    # unet.enable_gradient_checkpointing()

    vae.requires_grad_(False); text.requires_grad_(False)
    unet.train()

    noise_sched = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.999))

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / max(1, args.warmup_steps)
        total = max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - args.warmup_steps) / total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Precompute dummy text embedding ("ECT")
    def get_text_emb(bsz):
        ids = tok(["ECT"] * bsz, padding="max_length",
                  max_length=tok.model_max_length, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            return text(ids)[0]  # [B,T,D]

    scaler = torch.amp.GradScaler(device="cuda", enabled=args.fp16)

    def vae_encode(x):  # x in [0,1], shape [B,3,H,W]
        with torch.no_grad():
            lat = vae.encode((x * 2 - 1)).latent_dist.sample()
        return lat * 0.18215

    # ===== SSIM helper for reconstruction mode =====
    # Minimal SSIM (on [0,1]) for an optional perceptual term
    def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device='cpu'):
        coords = torch.arange(window_size, device=device).float() - (window_size - 1)/2
        gauss = torch.exp(-(coords**2)/(2*sigma*sigma))
        gauss = gauss / gauss.sum()
        w1d = gauss.view(1,1,1,-1)
        window = (w1d.transpose(2,3) @ w1d).squeeze(0).squeeze(0)
        return window

    def ssim_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
        C1 = (0.01)**2
        C2 = (0.03)**2
        b, c, h, w = x.shape
        device = x.device
        win = _gaussian_window(window_size, sigma, device=device).view(1,1,window_size,window_size)
        win = win.repeat(c, 1, 1, 1)
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
        return ssim_map.mean(dim=[1,2,3])

    # ---------- Training ----------
    if not args.vae_decoder_ft:
        # ===== Original UNet noise-prediction training =====
        step = 0
        while step < args.max_steps:
            for x in train_dl:
                if step >= args.max_steps:
                    break
                x = x.to(device, non_blocking=True)

                with torch.no_grad():
                    z   = vae_encode(x)  # [B,4,h,w]
                    eps = torch.randn_like(z)
                    t   = torch.randint(0, noise_sched.config.num_train_timesteps, (z.shape[0],), device=device).long()
                    zt  = noise_sched.add_noise(z, eps, t)
                    enc = get_text_emb(z.shape[0])

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                    pred = unet(zt, t, encoder_hidden_states=enc).sample
                    loss = F.mse_loss(pred, eps)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                sched.step()

                if step % args.log_every == 0:
                    writer.add_scalar("train/loss", loss.item(), step)
                    writer.add_scalar("train/lr",   opt.param_groups[0]["lr"], step)

                if val_batch is not None and step % args.eval_every == 0 and step > 0:
                    unet.eval()
                    with torch.no_grad():
                        z_val  = vae_encode(val_batch)
                        t_val  = torch.full((z_val.shape[0],), int(0.6 * noise_sched.config.num_train_timesteps),
                                            device=device, dtype=torch.long)
                        noise  = torch.randn_like(z_val)
                        z_noisy = noise_sched.add_noise(z_val, noise, t_val)
                        enc_v  = get_text_emb(z_val.shape[0])
                        # Quick sampler (debug preview only)
                        z_sample = z_noisy.clone()
                        for _ in range(10):
                            t_cur = torch.clamp(t_val, min=0)
                            eps_hat = unet(z_sample, t_cur, encoder_hidden_states=enc_v).sample
                            alpha = noise_sched.alphas_cumprod[t_cur].view(-1,1,1,1)
                            sigma = (1 - alpha).sqrt()
                            z_sample = (z_sample - sigma * eps_hat) / (alpha.sqrt() + 1e-8)
                            t_val = torch.clamp(t_val - (noise_sched.config.num_train_timesteps // 10), min=0)
                        imgs_in  = val_batch[:8].detach().cpu()
                        imgs_rec = (vae.decode(z_sample / 0.18215).sample * 0.5 + 0.5).clamp(0,1).detach().cpu()
                        writer.add_image("val/input_grid",  make_grid(imgs_in,  nrow=4), step)
                        writer.add_image("val/recon_grid",  make_grid(imgs_rec, nrow=4), step)
                    unet.train()

                if step % args.save_every == 0 and step > 0:
                    unet.save_pretrained(f"{args.out_dir}/step_{step:06d}")
                step += 1

        unet.save_pretrained(args.out_dir)
        writer.close()
        print("Saved UNet to", args.out_dir)
    else:
        # ===== VAE decoder reconstruction training (encoder frozen) =====
        # Freeze encoder and text/UNet; unfreeze decoder only
        vae.encoder.requires_grad_(False)
        for p in vae.decoder.parameters():
            p.requires_grad = True
        vae.train()

        # Optional EMA for stability
        use_ema = args.ema_decay > 0 and args.ema_decay < 1
        if use_ema:
            import copy
            ema_decoder = copy.deepcopy(vae.decoder).to(device)
            for p in ema_decoder.parameters():
                p.requires_grad = False
            def ema_update(model, ema_model, decay):
                with torch.no_grad():
                    for p, p_ema in zip(model.parameters(), ema_model.parameters()):
                        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

        opt_dec = torch.optim.AdamW(vae.decoder.parameters(), lr=args.vae_lr, betas=(0.9, 0.999), weight_decay=0.0)

        step = 0
        while step < args.vae_steps:
            for x in train_dl:
                if step >= args.vae_steps:
                    break
                x = x.to(device, non_blocking=True)  # [B,3,H,W] in [0,1]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                    x_in = x * 2 - 1
                    with torch.no_grad():
                        enc = vae.encode(x_in)
                        z   = enc.latent_dist.mode() * 0.18215
                    xhat = vae.decode(z / 0.18215).sample
                    xhat01 = (xhat * 0.5 + 0.5).clamp(0,1)
                    l1 = F.l1_loss(xhat01, x)
                    if args.ssim_weight > 0:
                        ssim = ssim_torch(x, xhat01)
                        loss = args.l1_weight * l1 + args.ssim_weight * (1 - ssim.mean())
                    else:
                        loss = args.l1_weight * l1

                opt_dec.zero_grad(set_to_none=True)
                loss.backward()
                opt_dec.step()

                if use_ema:
                    ema_update(vae.decoder, ema_decoder, args.ema_decay)

                if step % args.vae_log_every == 0:
                    writer.add_scalar("vae_decoder/loss", float(loss.detach().cpu()), step)
                    writer.add_scalar("vae_decoder/l1", float(l1.detach().cpu()), step)
                    if args.ssim_weight > 0:
                        writer.add_scalar("vae_decoder/ssim", float(ssim.mean().detach().cpu()), step)
                    grid_in  = make_grid(x.detach().cpu(), nrow=4)
                    grid_out = make_grid(xhat01.detach().cpu(), nrow=4)
                    writer.add_image("vae_decoder/input", grid_in, step)
                    writer.add_image("vae_decoder/recon", grid_out, step)

                step += 1
                if step >= args.vae_steps:
                    break

        # Save fine-tuned VAE (optionally swap in EMA weights before saving)
        os.makedirs(args.out_dir, exist_ok=True)
        if use_ema:
            # Copy EMA weights into the live decoder before saving
            for p, p_ema in zip(vae.decoder.parameters(), ema_decoder.parameters()):
                p.data.copy_(p_ema.data)
        vae.save_pretrained(args.out_dir)
        writer.close()
        print(f"Saved VAE (decoder-finetuned) to {args.out_dir}")


if __name__ == "__main__":
    main()