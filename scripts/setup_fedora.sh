#!/usr/bin/env bash
# One-shot environment setup for Flux-GS on Fedora with an RTX 50-series (Blackwell, sm_120) GPU.
#
# Prerequisites (one-time, needs sudo, not done by this script):
#   * NVIDIA proprietary driver 570 or newer (RPM Fusion: `sudo dnf install akmod-nvidia`), check with `nvidia-smi`.
#   * Miniforge / conda (https://github.com/conda-forge/miniforge).
#
# Everything else (CUDA 12.8 toolkit, a CUDA-compatible GCC, COLMAP, ffmpeg) is installed into a
# conda env, so Fedora's newer system GCC never has to be used with nvcc.
#
# Usage: bash scripts/setup_fedora.sh            (env name defaults to "flux-gs")
#        ENV_NAME=myenv bash scripts/setup_fedora.sh
set -euo pipefail

ENV_NAME=${ENV_NAME:-flux-gs}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMC_DIR=${TMC_DIR:-"$REPO_DIR/../mpeg-pcc-tmc13"}

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first."; exit 1; }
command -v conda >/dev/null || { echo "conda not found: install Miniforge first."; exit 1; }
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Creating conda env $ENV_NAME"
  conda create -y -n "$ENV_NAME" --override-channels -c conda-forge \
    python=3.11 cuda-version=12.8 cuda-nvcc cuda-cudart-dev cuda-libraries-dev \
    gcc_linux-64=13 gxx_linux-64=13 cmake ninja git ffmpeg colmap
fi
# conda's activation hooks (e.g. cuda-nvcc's) reference unset variables, so relax `set -u` around them.
set +u
conda activate "$ENV_NAME"
set -u

# GLOMAP is a much faster global mapper than COLMAP's incremental one; optional.
conda install -y --override-channels -c conda-forge glomap || echo "(glomap not available, COLMAP mapper will be used)"

export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="12.0"
export TCNN_CUDA_ARCHITECTURES=120

echo "==> PyTorch (CUDA 12.8 build, required for Blackwell)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "==> Python dependencies"
pip install websockets tqdm plyfile icecream dahuffman opencv-python-headless

echo "==> Building CUDA extensions for sm_120"
pip install --no-build-isolation "$REPO_DIR/submodules/diff-gaussian-rasterization_fluxgs"
pip install --no-build-isolation "$REPO_DIR/submodules/simple-knn"
pip install --no-build-isolation "$REPO_DIR/submodules/fused-ssim"
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

echo "==> RAPIDS cuML (k-means used for compression); 25.4+ supports Blackwell"
pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12>=25.4"

echo "==> MPEG G-PCC codec (tmc3) for compressed output"
if [ ! -x "$TMC_DIR/build/tmc3/tmc3" ]; then
  [ -d "$TMC_DIR" ] || git clone --depth 1 https://github.com/MPEGGroup/mpeg-pcc-tmc13.git "$TMC_DIR"
  cmake -S "$TMC_DIR" -B "$TMC_DIR/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$TMC_DIR/build" -j"$(nproc)"
fi
conda env config vars set FLUXGS_TMC3="$TMC_DIR/build/tmc3/tmc3" >/dev/null

echo "==> Checking the install"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("GPU:", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
import diff_gaussian_rasterization_fluxgs, simple_knn._C, fused_ssim, tinycudann, cuml  # noqa: F401
print("All CUDA extensions import OK")
EOF
command -v colmap >/dev/null && echo "COLMAP: $(command -v colmap)"
echo
echo "Done. Run 'conda activate $ENV_NAME' (re-activate once so FLUXGS_TMC3 is set), then see docs/room-pipeline.md."
