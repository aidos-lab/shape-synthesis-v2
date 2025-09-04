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
