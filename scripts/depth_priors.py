#!/usr/bin/env python3
"""Run a monocular depth model over a dataset's images and save one disparity map per image.

Output: <out_dir>/<image stem>.npy, float16, relative disparity (larger = closer), resized to --width
wide. These feed the depth and normal losses in room_priors.py.

Example: python scripts/depth_priors.py rooms/room1/dataset/images rooms/room1/dataset/priors
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODEL = "depth-anything/Depth-Anything-V2-Large-hf"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images")
    ap.add_argument("out_dir")
    ap.add_argument("--width", type=int, default=512, help="width of the saved prior maps")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            disp = model(pixel_values=inputs["pixel_values"].half()).predicted_depth  # [1, h', w']
        size = (round(h * args.width / w), args.width)
        disp = F.interpolate(disp[:, None].float(), size=size, mode="bilinear", align_corners=False)[0, 0]
        np.save(out / (path.stem + ".npy"), disp.cpu().numpy().astype(np.float16))
        if (i + 1) % 25 == 0 or i + 1 == len(paths):
            print(f"depth priors: {i + 1}/{len(paths)}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
