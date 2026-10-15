"""Check that the GPU stack works: PyTorch sees the card and gsplat's compiled kernels render on it."""
import os

import torch

assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("GPU:", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))

# Render a few random Gaussians so the kernels are exercised on this GPU, not just imported.
from gsplat import rasterization

n = 100
dev = "cuda"
_, alphas, _ = rasterization(
    means=torch.randn(n, 3, device=dev) + torch.tensor([0, 0, 5.0], device=dev),
    quats=torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1),
    scales=torch.rand(n, 3, device=dev) * 0.1,
    opacities=torch.rand(n, device=dev),
    colors=torch.rand(n, 3, device=dev),
    viewmats=torch.eye(4, device=dev)[None],
    Ks=torch.tensor([[[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]], device=dev),
    width=320,
    height=240,
)
assert alphas.sum() > 0, "gsplat rendered an empty image"
import fused_ssim, pycolmap  # noqa: F401,E401  (used by the gsplat trainer)
print("gsplat renders on the GPU OK")

if os.environ.get("WITH_FLUXGS") == "1":
    import cuml, diff_gaussian_rasterization_fluxgs, simple_knn._C, tinycudann  # noqa: F401,E401
    print("Flux-GS extensions import OK")
