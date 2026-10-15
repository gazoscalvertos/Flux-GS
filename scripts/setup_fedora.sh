#!/usr/bin/env bash
# Environment setup for room scanning on Fedora with an RTX 50-series (Blackwell, sm_120) GPU.
#
# Prerequisites (one-time, needs sudo, not done by this script):
#   * NVIDIA proprietary driver 570 or newer (RPM Fusion: `sudo dnf install akmod-nvidia`), check with `nvidia-smi`.
#   * conda: Miniforge (https://github.com/conda-forge/miniforge), or Anaconda/Miniconda (only conda-forge is used).
#
# Everything else (CUDA 12.8 toolkit, a CUDA-compatible GCC, COLMAP, ffmpeg) is installed into a
# conda env, so Fedora's newer system GCC never has to be used with nvcc.
#
# The default trainer is gsplat (https://github.com/nerfstudio-project/gsplat). Flux-GS's own
# mobile-compression extras (tiny-cuda-nn, cuML, tmc3) are only built with WITH_FLUXGS=1.
#
# Usage: bash scripts/setup_fedora.sh                 (env name defaults to "flux-gs")
#        WITH_FLUXGS=1 bash scripts/setup_fedora.sh
set -euo pipefail

ENV_NAME=${ENV_NAME:-flux-gs}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GSPLAT_DIR=${GSPLAT_DIR:-"$REPO_DIR/../gsplat"}
GSPLAT_VERSION=v1.5.3
WITH_FLUXGS=${WITH_FLUXGS:-0}
TMC_DIR=${TMC_DIR:-"$REPO_DIR/../mpeg-pcc-tmc13"}
TCNN_DIR=${TCNN_DIR:-"$REPO_DIR/../tiny-cuda-nn"}

# GitHub connections sometimes stall for minutes, so clone with retries instead of letting pip clone.
clone() {  # clone <url> <dir> [extra git args]
  local url=$1 dir=$2; shift 2
  for i in 1 2 3 4 5; do
    rm -rf "$dir"
    timeout 600 git clone --depth 1 "$@" "$url" "$dir" && return 0
    echo "git clone of $url failed (attempt $i), retrying"; sleep 5
  done
  return 1
}

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

echo "==> gsplat $GSPLAT_VERSION (CUDA kernels compiled now for sm_120, so build errors show up here)"
[ -f "$GSPLAT_DIR/setup.py" ] || \
  clone https://github.com/nerfstudio-project/gsplat.git "$GSPLAT_DIR" --branch "$GSPLAT_VERSION" --recursive --shallow-submodules
pip install --no-build-isolation "$GSPLAT_DIR"
pip install --no-build-isolation -r "$GSPLAT_DIR/examples/requirements.txt"
conda env config vars set GSPLAT_DIR="$GSPLAT_DIR" >/dev/null

if [ "$WITH_FLUXGS" = 1 ]; then
  echo "==> Flux-GS extensions and compression tools"
  pip install websockets tqdm plyfile icecream dahuffman opencv-python-headless
  pip install --no-build-isolation "$REPO_DIR/submodules/diff-gaussian-rasterization_fluxgs"
  pip install --no-build-isolation "$REPO_DIR/submodules/simple-knn"
  pip install --no-build-isolation "$REPO_DIR/submodules/fused-ssim"
  [ -f "$TCNN_DIR/dependencies/cutlass/CMakeLists.txt" ] || \
    clone https://github.com/NVlabs/tiny-cuda-nn.git "$TCNN_DIR" --recursive --shallow-submodules
  pip install --no-build-isolation "$TCNN_DIR/bindings/torch"
  pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12>=25.4"
  if [ ! -x "$TMC_DIR/build/tmc3/tmc3" ]; then
    [ -f "$TMC_DIR/CMakeLists.txt" ] || clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git "$TMC_DIR"
    # tmc3 declares cmake_minimum_required < 3.5, which CMake 4 refuses without this.
    cmake -S "$TMC_DIR" -B "$TMC_DIR/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build "$TMC_DIR/build" -j"$(nproc)"
  fi
  conda env config vars set FLUXGS_TMC3="$TMC_DIR/build/tmc3/tmc3" >/dev/null
fi

echo "==> Checking the install"
WITH_FLUXGS=$WITH_FLUXGS python "$REPO_DIR/scripts/check_gpu.py"
command -v colmap >/dev/null && echo "COLMAP: $(command -v colmap)"
echo
echo "Done. Run 'conda deactivate && conda activate $ENV_NAME' once so GSPLAT_DIR is set, then see docs/room-pipeline.md."
