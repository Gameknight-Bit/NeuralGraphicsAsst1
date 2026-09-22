import torch

from pathlib import Path

import numpy as np
from PIL import Image

from texture import load_texture
from s3tc import DXT1Texture

from feature_grid import FeatureGrid

root = Path(__file__).resolve().parent
output_dir = root / "outputs"
output_dir.mkdir(exist_ok=True)

# S3TC Baseline Analysis/Generation
for filename in ("gradient.png", "bricks.png", "clouds.png"):
    texture = load_texture(root / "textures" / filename)
    sampler = DXT1Texture().compress(texture)

    h, w = texture.shape[:2]
    u = ((np.arange(w) + 0.5) / w)[None, :]
    v = ((np.arange(h) + 0.5) / h)[:, None]
    reconstruction = sampler.sample(u, v)

    pixels = np.rint(np.clip(reconstruction, 0, 1) * 255).astype(np.uint8)
    output_path = output_dir / f"{Path(filename).stem}_dxt1.png"
    Image.fromarray(pixels).save(output_path)

    size = sampler.blocks.nbytes
    print(f"{filename}: {size:,} compressed bytes")
    print(f"  Ratio: {texture.nbytes / size:.2f}:1")
    print(f"  Bits/texel: {8 * size / (h * w):.2f}")

# Helper for torch device settings
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Testing
device = get_device()
feature_grid = FeatureGrid().to(device)

uv = torch.rand(1024, 2, device=device)
features = feature_grid(uv)

print(features.shape)  # torch.Size([1024, 8])