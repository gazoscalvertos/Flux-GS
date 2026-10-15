# Room capture to splat (Fedora + RTX 50-series)

Step 1 of the mobile room-scanning project: get one room from a phone video to a trained Flux-GS model
on a local GPU, before the dedicated Android capture app exists.

## 1. Set up the machine (once)

1. Install the NVIDIA driver (570 or newer for Blackwell cards) from RPM Fusion and check `nvidia-smi` works.
2. Install [Miniforge](https://github.com/conda-forge/miniforge).
3. From the repo root run `bash scripts/setup_fedora.sh`.

The script builds a `flux-gs` conda env with CUDA 12.8, GCC 13 (Fedora's own GCC is too new for nvcc),
COLMAP, GLOMAP if available, PyTorch cu128 and [gsplat](https://github.com/nerfstudio-project/gsplat)
compiled for `sm_120`. It ends by rendering a few test Gaussians on the GPU (`scripts/check_gpu.py`),
so a clean finish means training will run. `WITH_FLUXGS=1` also builds Flux-GS's mobile-compression
extras (tiny-cuda-nn, cuML, tmc3), which only the `--trainer fluxgs` option needs.

## 2. Capture a room with the phone

* Camera settings: 4K at 30 fps, **10-bit HDR off**, video stabilisation off if the option is shown.
* Lock exposure if you can (tap and hold on a mid-brightness wall) and turn on all the room's lights.
* Walk slowly around the edge of the room facing inwards, then do a second lap facing the walls and corners
  at a different height (chest, then waist). Overlap matters more than duration. Aim for 1 to 3 minutes.
* Avoid fast turns on the spot, which give blurred frames with no parallax.

## 3. Process it

```shell
conda activate flux-gs
python scripts/process_room.py --video ~/Videos/kitchen.mp4 --name kitchen
```

Output goes to `rooms/kitchen/`: `input/` (selected frames), `dataset/` (undistorted COLMAP scene),
`model/` (trained splat; `model/ply/*.ply` opens in the [SuperSplat](https://superspl.at/editor) web viewer). Each stage is skipped when its
output exists, so you can rerun after a failure; `--force poses` redoes poses onwards.

Useful options: `--stop-after poses` to check registration before training, `--frames 400` for larger
rooms, `--matcher sequential` for very long videos, and anything after `--` is passed to the trainer
(for example `-- --steps_scaler 0.25` for a quick low-quality test run).

For a quick end-to-end check of the Flux-GS trainer (about a minute of training), shorten the run and move
the compression stages inside it; `--nn_iter` must not be later than `--net_itr`:

```shell
python scripts/process_room.py --video ~/Videos/kitchen.mp4 --name kitchen_test --frames 100 --trainer fluxgs -- \
  --iterations 3000 --save_iterations 3000 --test_iterations 3000 --checkpoint_iterations 3000 \
  --svq_itr 2500 --net_itr 1500 --nn_iter 1500 --densify_until_iter 2000
```

If fewer than 80% of frames register, the script warns you. The usual fix is recapturing with slower
movement and more overlap; plain walls with no texture are the hardest case.
