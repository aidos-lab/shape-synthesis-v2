# infer_img2img_pretrained_dff.py
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")  # keep transformers from importing torchvision

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
import numpy as np

from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from safetensors.torch import load_file


def list_images(path: Path, exts=(".png", ".jpg", ".jpeg", ".bmp")) -> List[Path]:
    if path.is_file():
        return [path]
    files = []
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in exts:
            files.append(p)
    return files


essential_prompt = "ECT"  # same token used during training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to an image or a directory of images to reconstruct")
    ap.add_argument("--out-dir", type=str, default="./outputs/img2img", help="Where to save reconstructions")
    ap.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--unet-dir", type=str, required=True, help="Folder with fine-tuned UNet (save_pretrained) or a .safetensors file")
    ap.add_argument("--vae-lora", type=str, default="", help="Optional path to a VAE decoder LoRA .safetensors file to load")
    ap.add_argument("--vae-lora-scale", type=float, default=1.0, help="Scale factor for VAE LoRA (alpha multiplier)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=256, help="Images will be resized to a square of this size before inference")
    ap.add_argument("--limit", type=int, default=0, help="Optionally cap number of images processed")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(inp)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found under {inp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Build pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.safety_checker = None  # speed; not needed for scientific ECT images
    pipe.enable_attention_slicing()

    # ===== Optional: Load VAE decoder LoRA =====
    import torch.nn as nn
    from safetensors.torch import load_file as _safe_load

    class _LoRAConv(nn.Module):
        def __init__(self, in_ch, out_ch, r=4, alpha=4):
            super().__init__()
            self.r = r
            self.alpha = alpha
            self.scale = alpha / max(1, r)
            self.lora_A = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
            self.lora_B = nn.Conv2d(r, out_ch, kernel_size=1, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)) if hasattr(nn.init, 'kaiming_uniform_') else None
            nn.init.zeros_(self.lora_B.weight)
            self.dropout = nn.Identity()

        def forward(self, x):
            return self.lora_B(self.lora_A(self.dropout(x))) * self.scale

    def _attach_lora_to_decoder(vae, rank_default=4, alpha_default=4, scale_mult=1.0):
        """Attach LoRA modules (LoCon-style 1x1) to every Conv2d in the VAE.decoder and return a dict mapping names->modules."""
        loras = {}
        for name, module in vae.decoder.named_modules():
            if isinstance(module, nn.Conv2d):
                l = _LoRAConv(module.in_channels, module.out_channels, r=rank_default, alpha=alpha_default)
                # Register as a child so it lands at e.g. decoder.*.lora
                module.__dict__["_has_lora"] = True
                setattr(module, "lora", l)
                # Patch forward: y = conv(x) + lora(x)
                orig_forward = module.forward
                def _patched_forward(x, _orig=orig_forward, _l=l):
                    return _orig(x) + _l(x)
                module.forward = _patched_forward
                loras[f"decoder.{name}.lora"] = l
        # scale multiplier (allows runtime scaling)
        for l in loras.values():
            l.scale *= float(scale_mult)
        return loras

    def _load_lora_safetensors_into_decoder(vae, path, scale_mult=1.0):
        sd = _safe_load(path)
        # Infer rank/alpha from shapes; group keys by module prefix ending with '.lora'
        # Expected keys like: 'decoder.xxx.lora.lora_A.weight', 'decoder.xxx.lora.lora_B.weight', optionally 'alpha'/'scale'
        # First, build a map of module prefixes
        prefixes = set()
        for k in sd.keys():
            if k.endswith("lora_A.weight"):
                prefixes.add(k[:-len("lora_A.weight")] )
        # Attach modules as needed and load weights
        import re
        for prefix in sorted(prefixes):
            # Parse module path: e.g. 'decoder.xxx.lora.' -> target conv at 'decoder.xxx'
            target = prefix[:-len("lora.")]  # remove trailing 'lora.'
            # Find the target module
            m = vae
            for attr in target.split('.'):
                m = getattr(m, attr)
            # Determine rank from weight shape
            A_key = prefix + "lora_A.weight"
            B_key = prefix + "lora_B.weight"
            A_w = sd[A_key]
            B_w = sd[B_key]
            r = A_w.shape[0]
            in_ch = A_w.shape[1]
            out_ch = B_w.shape[0]
            # Attach lora if missing or mismatched
            if not hasattr(m, 'lora') or not isinstance(m.lora, _LoRAConv) or m.lora.r != r or m.lora.lora_A.in_channels != in_ch or m.lora.lora_B.out_channels != out_ch:
                # Default alpha = r (if not stored separately)
                l = _LoRAConv(in_ch, out_ch, r=r, alpha=r)
                setattr(m, 'lora', l)
                orig_forward = m.forward
                def _patched_forward(x, _orig=orig_forward, _l=l):
                    return _orig(x) + _l(x)
                m.forward = _patched_forward
            # Load weights
            with torch.no_grad():
                m.lora.lora_A.weight.copy_(A_w)
                m.lora.lora_B.weight.copy_(B_w)
                # Optional per-module scale
                if (prefix + "scale") in sd:
                    m.lora.scale = float(sd[prefix + "scale"]) * float(scale_mult)
                else:
                    m.lora.scale *= float(scale_mult)
        return True

    # If user supplied a LoRA file, load it
    if args.vae_lora:
        try:
            _load_lora_safetensors_into_decoder(pipe.vae, args.vae_lora, scale_mult=args.vae_lora_scale)
            print(f"[infer] Loaded VAE decoder LoRA from {args.vae_lora} (scale×{args.vae_lora_scale})")
        except Exception as e:
            print(f"[infer] Failed to load VAE LoRA: {e}")

    # Load fine-tuned UNet
    unet_path = Path(args.unet_dir)
    loaded = False
    try:
        ft_unet = UNet2DConditionModel.from_pretrained(unet_path)
        ft_unet.to(device, dtype=dtype)
        pipe.unet = ft_unet
        loaded = True
        print(f"[infer] loaded UNet via from_pretrained: {unet_path}")
    except Exception as e:
        print(f"[infer] from_pretrained failed: {e}\nTrying to load state dict…")
        try:
            # Expect a unet.safetensors with a bare state_dict
            sd_file = unet_path if unet_path.suffix == ".safetensors" else (unet_path / "unet.safetensors")
            sd = load_file(str(sd_file))
            pipe.unet.load_state_dict(sd, strict=True)
            pipe.unet.to(device, dtype=dtype)
            loaded = True
            print(f"[infer] loaded UNet state dict from {sd_file}")
        except Exception as e2:
            raise SystemExit(f"Failed to load UNet from {args.unet_dir}: {e2}")

    assert loaded

    # Set RNG
    generator = torch.Generator(device=device)
    if args.seed >= 0:
        generator.manual_seed(args.seed)

    def load_pil(path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if args.resolution > 0 and (img.width != args.resolution or img.height != args.resolution):
            img = img.resize((args.resolution, args.resolution), resample=Image.BICUBIC)
        return img

    batch_size = args.batch_size
    n = len(images)

    for i in range(0, n, batch_size):
        batch_paths = images[i : i + batch_size]
        batch_imgs = [load_pil(p) for p in batch_paths]

        results = pipe(
            prompt=[essential_prompt] * len(batch_imgs),
            image=batch_imgs,
            strength=args.strength,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=generator,
        )
        out_imgs = results.images

        for j, out_img in enumerate(out_imgs):
            p = batch_paths[j]
            # mirror input folder structure in out_dir
            rel = p.name if inp.is_file() else str(p.relative_to(inp))
            out_path = out_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_img.save(out_path)
            idx = i + j
            if idx % 10 == 0:
                print(f"[infer] {idx+1}/{n} -> {out_path}")

    print(f"Done. Wrote {n} images to {out_dir}")


if __name__ == "__main__":
    main()
