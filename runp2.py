import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from texture import load_texture
from s3tc import DXT1Texture


root = Path(__file__).resolve().parent
output_dir = root / "outputs/s3tc_results"
output_dir.mkdir(exist_ok=True)
filenames = ("gradient.png", "bricks.png", "clouds.png")
results = []
fig, axes = plt.subplots(len(filenames), 3, figsize=(15, 14), layout="constrained")

for row, filename in enumerate(filenames):
    texture = load_texture(root / "textures" / filename)
    sampler = DXT1Texture().compress(texture)
    h, w = texture.shape[:2]

    # These texel centers match sample's x = u*W - 0.5 mapping.
    u = ((np.arange(w) + 0.5) / w)[None, :]
    v = ((np.arange(h) + 0.5) / h)[:, None]
    reconstruction = sampler.sample(u, v)

    # Average over every pixel AND all three channels. MAX = 1.
    target = texture.astype(np.float32) / 255.0
    squared_error = (reconstruction - target) ** 2
    mse = float(np.mean(squared_error))
    psnr = float("inf") if mse == 0 else -10.0 * np.log10(mse)

    compressed_bytes = sampler.blocks.nbytes  # Includes any padded edge blocks.
    raw_bytes = texture.nbytes               # H * W * 3 for RGB8.
    ratio = compressed_bytes / raw_bytes     # Assignment's requested ratio.
    factor = raw_bytes / compressed_bytes    # Reciprocal, e.g. 6x smaller.
    bits_per_texel = 8 * compressed_bytes / (h * w)
    results.append([filename, psnr, mse, compressed_bytes, raw_bytes,
                    compressed_bytes / 1000, raw_bytes / 1000, ratio, factor, bits_per_texel])

    pixels = np.rint(np.clip(reconstruction, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(output_dir / f"{Path(filename).stem}_dxt1.png")

    axes[row, 0].imshow(texture, interpolation="nearest")
    axes[row, 0].set_title(f"{filename}: original\n{raw_bytes / 1000:.3f} KB")
    axes[row, 1].imshow(reconstruction, interpolation="nearest")
    axes[row, 1].set_title(f"S3TC: {psnr:.2f} dB | {compressed_bytes / 1000:.3f} KB\n"
                         f"Compressed / raw = {ratio:.6f}")

    # Brighter locations have larger reconstruction errors. Each row has
    # its own labeled scale; this is an error map, not another reconstruction.
    error_map = np.sqrt(np.mean(squared_error, axis=2))
    heatmap = axes[row, 2].imshow(error_map, cmap="magma", interpolation="nearest",
                                  vmin=0, vmax=max(float(error_map.max()), 1e-8))
    axes[row, 2].set_title("Error locations (RGB RMSE)")
    fig.colorbar(heatmap, ax=axes[row, 2], label="Normalized RGB error")
    for ax in axes[row]:
        ax.set_axis_off()

    print(f"{filename}: PSNR {psnr:.3f} dB | {compressed_bytes:,} compressed bytes")
    print(f"  Compressed / original: {ratio:.6f} ({100 * ratio:.2f}%)")
    print(f"  Compression factor: {factor:.2f}x | Bits/texel: {bits_per_texel:.2f}")

fig.savefig(output_dir / "s3tc_comparison.png", dpi=150)
plt.close(fig)
with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["texture", "psnr_db", "mse", "compressed_bytes", "raw_bytes",
                     "compressed_kb", "raw_kb", "compressed_over_raw", "compression_factor", "bits_per_texel"])
    writer.writerows(results)
print(f"\nSaved reconstructions, comparison, and measurements to {output_dir}")