import torch
from pathlib import Path
import numpy as np
from PIL import Image

#Local Lib Imports
from texture import load_texture
from s3tc import DXT1Texture
from nncomp import FeatureGrid, ColorMLP, NeuralTexture

# Helper for torch device settings
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

root = Path(__file__).resolve().parent
output_dir = root / "outputs"
output_dir.mkdir(exist_ok=True)
device = get_device()

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

#Neural texture training on test textures
for filename in ("gradient.png", "bricks.png", "clouds.png"):
    print(f"\nTraining on {filename}")
    texture = load_texture(root / "textures" / filename)

    # A fresh model for each texture.
    model = NeuralTexture().to(device)
    model.compress(texture)

    h, w = texture.shape[:2]
    u = ((np.arange(w) + 0.5) / w)[None, :]
    v = ((np.arange(h) + 0.5) / h)[:, None]
    reconstruction = model.sample(u, v)

    target = texture.astype(np.float32) / 255.0
    mse = np.mean((reconstruction - target) ** 2)
    psnr = float("inf") if mse == 0 else -10 * np.log10(mse)
    print(f"Full-image MSE: {mse:.6f} | PSNR: {psnr:.2f} dB")

    pixels = np.rint(reconstruction * 255).astype(np.uint8)
    Image.fromarray(pixels).save(
        output_dir / f"{Path(filename).stem}_neural.png"
    )