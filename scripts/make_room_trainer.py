#!/usr/bin/env python3
"""Generate gsplat's examples/room_trainer.py: simple_trainer.py plus the depth/normal priors in room_priors.py.

gsplat's trainer is one long training loop with no hooks, so this applies a few small, anchored
edits to a copy of it instead of maintaining a fork. Each anchor must match exactly once, which
holds for the pinned gsplat version in setup_fedora.sh; on any other version it stops with a clear
error rather than producing a silently broken trainer.

Usage: python scripts/make_room_trainer.py [GSPLAT_DIR]
"""
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

EDITS = [
    # New config fields, next to the existing depth-loss settings.
    (
        "    depth_lambda: float = 1e-2\n",
        "    depth_lambda: float = 1e-2\n"
        "    # Room priors (room_priors.py): folder of per-image monocular disparity maps, or None.\n"
        "    prior_dir: Optional[str] = None\n"
        "    prior_depth_lambda: float = 0.1\n"
        "    prior_normal_lambda: float = 0.05\n"
        "    prior_start: int = 500\n",
    ),
    # Load the priors once before the loop.
    (
        "        pbar = tqdm.tqdm(range(init_step, max_steps))\n",
        "        priors = None\n"
        "        if cfg.prior_dir:\n"
        f"            sys.path.insert(0, {str(SCRIPTS_DIR)!r})\n"
        "            from room_priors import load_priors, prior_losses\n"
        "            priors = load_priors(cfg.prior_dir, self.parser.image_names, device)\n"
        "        pbar = tqdm.tqdm(range(init_step, max_steps))\n",
    ),
    # Render expected depth alongside colour whenever priors are on.
    (
        'render_mode="RGB+ED" if cfg.depth_loss else "RGB",',
        'render_mode="RGB+ED" if (cfg.depth_loss or cfg.prior_dir) else "RGB",',
    ),
    # Add the prior losses to the photometric loss.
    (
        "            if cfg.use_bilateral_grid:\n                tvloss = 10",
        "            if priors is not None and step >= cfg.prior_start:\n"
        "                name = self.parser.image_names[self.trainset.indices[int(image_ids[0])]]\n"
        "                p_depth, p_normal = prior_losses(depths[0, ..., 0], alphas[0, ..., 0], priors[name], Ks[0])\n"
        "                loss += cfg.prior_depth_lambda * p_depth + cfg.prior_normal_lambda * p_normal\n"
        "            if cfg.use_bilateral_grid:\n                tvloss = 10",
    ),
    ("import json\n", "import json\nimport sys\n"),
]


def main():
    gsplat_dir = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GSPLAT_DIR", SCRIPTS_DIR.parent.parent / "gsplat"))
    src = gsplat_dir / "examples" / "simple_trainer.py"
    text = src.read_text()
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"make_room_trainer: expected 1 match in {src}, found {n}, for:\n{old}\n"
                     "This gsplat version differs from the pinned one; re-run scripts/setup_fedora.sh.")
        text = text.replace(old, new)
    out = src.with_name("room_trainer.py")
    out.write_text(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
