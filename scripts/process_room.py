#!/usr/bin/env python3
"""Turn a phone video (or a folder of photos) of one room into a trained Flux-GS splat.

Stages, each skipped automatically if its output already exists (use --force to redo):
  1. frames  - extract frames from the video and keep the sharpest one in each time window
  2. poses   - COLMAP feature extraction + matching, GLOMAP/COLMAP mapping
  3. dataset - undistort to PINHOLE cameras in the layout train.py expects (images/, sparse/0/)
  4. priors  - monocular depth maps (Depth Anything V2) used to keep walls and ceilings flat
  5. train   - train a splat with gsplat (default) or this repo's Flux-GS trainer

Example:
  python scripts/process_room.py --video ~/Videos/kitchen.mp4 --name kitchen
  python scripts/process_room.py --images ~/Pictures/kitchen/ --name kitchen --stop-after poses
"""
import argparse
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
STAGES = ["frames", "poses", "dataset", "priors", "train"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def sharpness(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    # Score on a downscaled copy so the result does not depend on resolution and stays fast.
    img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.Laplacian(img, cv2.CV_64F).var()


def stage_frames(args, work):
    out = work / "input"
    if args.images:
        src = sorted(p for p in Path(args.images).expanduser().iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not src:
            sys.exit(f"No images found in {args.images}")
        out.mkdir(parents=True)
        for p in src:
            shutil.copy2(p, out / p.name)
        print(f"Copied {len(src)} images")
        return

    raw = work / "raw_frames"
    raw.mkdir(parents=True, exist_ok=True)
    # Pixel "10-bit HDR video" should be turned off when recording; tonemapping HLG here is lossy.
    vf = f"fps={args.extract_fps},scale='min({args.max_width},iw)':-2"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", Path(args.video).expanduser(),
         "-vf", vf, "-q:v", "2", raw / "f_%06d.jpg"])
    frames = sorted(raw.glob("*.jpg"))
    if not frames:
        sys.exit("ffmpeg produced no frames")

    # Split the timeline into equal windows and keep the sharpest frame in each, which drops
    # motion-blurred frames while keeping even coverage of the walk-around.
    n = min(args.frames, len(frames))
    scores = [sharpness(f) for f in frames]
    out.mkdir(parents=True)
    kept = []
    for i in range(n):
        lo, hi = i * len(frames) // n, max((i + 1) * len(frames) // n, i * len(frames) // n + 1)
        best = max(range(lo, hi), key=lambda j: scores[j])
        kept.append(scores[best])
        shutil.copy2(frames[best], out / f"{i:05d}.jpg")
    shutil.rmtree(raw)
    print(f"Kept {n} of {len(frames)} extracted frames "
          f"(median sharpness {sorted(kept)[len(kept) // 2]:.0f}, lowest {min(kept):.0f})")


def count_registered(model_dir):
    # images.bin starts with the number of registered images as a uint64. Read it directly rather than
    # importing scene/, whose package init pulls in the Flux-GS training stack (cupy, cuML).
    with open(model_dir / "images.bin", "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def stage_poses(args, work):
    db, images, sparse = work / "database.db", work / "input", work / "sparse_raw"
    if db.exists():
        db.unlink()
    # A previous interrupted run can leave sparse_raw behind; mapper output must start clean.
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    run(["colmap", "feature_extractor", "--database_path", db, "--image_path", images,
         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV"])
    if args.matcher == "exhaustive":
        run(["colmap", "exhaustive_matcher", "--database_path", db])
    else:
        run(["colmap", "sequential_matcher", "--database_path", db, "--SequentialMatching.overlap", "20"])

    if shutil.which("glomap") and not args.colmap_mapper:
        run(["glomap", "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse])
    else:
        run(["colmap", "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse])

    models = [d for d in sparse.iterdir() if (d / "images.bin").exists()]
    if not models:
        sys.exit("Pose estimation failed: no model was reconstructed. Try --matcher exhaustive or recapture "
                 "with slower movement and more overlap.")
    counts = {d: count_registered(d) for d in models}
    best = max(counts, key=counts.get)
    total = len(list(images.iterdir()))
    (work / "best_model.txt").write_text(str(best))
    print(f"Registered {counts[best]} of {total} images in the largest model ({len(models)} model(s) found)")
    if counts[best] < 0.8 * total:
        print("WARNING: under 80% of frames registered; parts of the room may be missing.")


def stage_dataset(args, work):
    best = Path((work / "best_model.txt").read_text())
    dataset = work / "dataset"
    run(["colmap", "image_undistorter", "--image_path", work / "input", "--input_path", best,
         "--output_path", dataset, "--output_type", "COLMAP"])
    # train.py reads sparse/0/, the undistorter writes sparse/ directly.
    target = dataset / "sparse" / "0"
    target.mkdir(exist_ok=True)
    for f in (dataset / "sparse").iterdir():
        if f.is_file():
            f.rename(target / f.name)


def stage_priors(args, work):
    if not use_priors(args):
        print("Depth priors off (--no-priors or --trainer fluxgs), skipping")
        return
    run([sys.executable, REPO_DIR / "scripts" / "depth_priors.py", work / "dataset" / "images", work / "dataset" / "priors"])


def use_priors(args):
    return args.trainer == "gsplat" and not args.no_priors


def stage_train(args, work):
    model = work / args.model_name
    if args.trainer == "gsplat":
        gsplat_dir = Path(os.environ.get("GSPLAT_DIR", REPO_DIR.parent / "gsplat"))
        if not (gsplat_dir / "examples" / "simple_trainer.py").exists():
            sys.exit(f"gsplat not found at {gsplat_dir}; run scripts/setup_fedora.sh or set GSPLAT_DIR")
        prior_args = []
        if use_priors(args):
            # Depth/normal priors keep plain walls and ceilings flat instead of cloudy (see room_priors.py).
            run([sys.executable, REPO_DIR / "scripts" / "make_room_trainer.py", gsplat_dir])
            trainer = gsplat_dir / "examples" / "room_trainer.py"
            prior_args = ["--prior_dir", work / "dataset" / "priors"]
        else:
            trainer = gsplat_dir / "examples" / "simple_trainer.py"
        # Frames are already capped at --max-width, so train at full resolution (data_factor 1).
        # The default strategy grows without bound on a 4K room capture (5M+ Gaussians, out of memory on 16 GB);
        # MCMC keeps the count capped. Expandable segments reduce allocator fragmentation as the model grows.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        run([sys.executable, trainer, "mcmc", "--data_dir", work / "dataset", "--data_factor", "1",
             "--result_dir", model, "--save_ply", "--disable_viewer",
             "--strategy.cap-max", str(args.max_gaussians), *prior_args, *args.train_args], cwd=gsplat_dir / "examples")
        plys = sorted((model / "ply").glob("*.ply"), key=lambda p: p.stat().st_mtime)
        print(f"\nTrained model: {model}\n  splat file: {plys[-1] if plys else model / 'ply'}"
              "\n  view it by dragging the .ply into https://superspl.at/editor")
    else:
        run([sys.executable, "train.py", "-s", work / "dataset", "-m", model,
             "--densification_interval", "500", "--optimizer_type", "default",
             "--highfeature_lr", "0.02", "--grad_abs_thresh", "0.0008", *args.train_args], cwd=REPO_DIR)
        print(f"\nTrained model: {model}\n  point cloud: {model}/point_cloud/\n  web viewer file: {model}/comp.json")


def stage_done(stage, work, args):
    if stage == "train":
        model = work / args.model_name
        return any((model / "ply").glob("*.ply")) or (model / "comp.json").exists()
    if stage == "priors":
        priors, images = work / "dataset" / "priors", work / "dataset" / "images"
        return not use_priors(args) or (priors.exists() and len(list(priors.glob("*.npy"))) == len(list(images.iterdir())))
    marker = {"frames": "input", "poses": "best_model.txt", "dataset": "dataset/sparse/0"}
    return (work / marker[stage]).exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="walk-around video of the room")
    src.add_argument("--images", help="folder of photos instead of a video")
    ap.add_argument("--name", required=True, help="room name, used for the output folder")
    ap.add_argument("--out", default=str(REPO_DIR / "rooms"), help="parent folder for room outputs")
    ap.add_argument("--frames", type=int, default=300, help="number of frames to keep from the video")
    ap.add_argument("--extract-fps", type=float, default=6, help="frames per second sampled before selection")
    ap.add_argument("--max-width", type=int, default=1920, help="downscale frames wider than this")
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive",
                    help="exhaustive is slower but handles revisiting parts of the room")
    ap.add_argument("--colmap-mapper", action="store_true", help="use COLMAP's mapper even if GLOMAP is installed")
    ap.add_argument("--trainer", choices=["gsplat", "fluxgs"], default="gsplat",
                    help="gsplat (default) or this repo's Flux-GS trainer (needs WITH_FLUXGS=1 setup)")
    ap.add_argument("--max-gaussians", type=int, default=3_000_000,
                    help="cap on splat count for gsplat (MCMC strategy); lower it if training runs out of GPU memory")
    ap.add_argument("--no-priors", action="store_true",
                    help="train gsplat without the monocular depth/normal priors that flatten walls and ceilings")
    ap.add_argument("--model-name", default="model",
                    help="output folder for the trained model, so variants of one room can sit side by side")
    ap.add_argument("--stop-after", choices=STAGES, default="train")
    ap.add_argument("--force", choices=STAGES, help="rerun from this stage, deleting its output and later ones")
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="extra train.py args after --")
    args = ap.parse_args()
    args.train_args = [a for a in args.train_args if a != "--"]

    work = Path(args.out).expanduser().resolve() / args.name
    work.mkdir(parents=True, exist_ok=True)
    if args.force:
        cleanup = {"frames": ["input", "raw_frames"], "poses": ["database.db", "sparse_raw", "best_model.txt"],
                   "dataset": ["dataset"], "priors": ["dataset/priors"],
                   "train": [args.model_name]}
        for stage in STAGES[STAGES.index(args.force):]:
            for name in cleanup[stage]:
                p = work / name
                shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)

    funcs = {"frames": stage_frames, "poses": stage_poses, "dataset": stage_dataset, "priors": stage_priors,
             "train": stage_train}
    for stage in STAGES:
        if stage_done(stage, work, args):
            print(f"== {stage}: already done, skipping")
        else:
            print(f"== {stage}")
            funcs[stage](args, work)
        if stage == args.stop_after:
            break


if __name__ == "__main__":
    main()
