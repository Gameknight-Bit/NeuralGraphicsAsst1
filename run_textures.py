"""Train, quantize, and compare any textures in textures.

Examples (run from the project folder):
    python run_textures.py gradient.png bricks.png clouds.png --output p6_p7_results
    python run_textures.py wood.jpg patterns/fabric.png gravel.png --output p8_results
    python run_textures.py wood.jpg --quantize-mlp

Requires texture.py, s3tc.py, and nncomp.py.
Defaults: fresh Small/Medium/Large fits, 2000 steps, batch 16384, Adam lr=0.01.
Images are converted to RGB without resizing. Quote paths containing spaces.
The output folder contains CSVs, combined plots, and one folder per texture.
Sizes count encoded arrays and quantization metadata, excluding file headers.
Frequency plots are qualitative aids; explain the results using the actual
texture and measured PSNR. Their power scales are normalized independently.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without opening windows during training.
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from texture import load_texture
from s3tc import DXT1Texture
from nncomp import NeuralTexture, make_training_data, quantize_model


MODELS = [("Small", (64,), 2), ("Medium", (16, 32, 64), 2),
          ("Large", (16, 32, 64, 128), 4)]


def psnr(prediction, target):
    """Full-image PSNR, averaged over all pixels and RGB channels."""
    mse = np.mean((prediction - target) ** 2)
    return float("inf") if mse == 0 else float(-10 * np.log10(mse))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("textures", nargs="+", help="Paths relative to textures/.")
    parser.add_argument("--output", default="results", help="Output folder, relative to this script.")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per model (default: 2000).")
    parser.add_argument("--batch-size", type=int, default=16_384, help="Texels per batch (default: 16384).")
    parser.add_argument("--quantize-mlp", action="store_true", help="Also quantize MLP weights and biases.")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive.")

    root = Path(__file__).resolve().parent
    textures_dir = (root / "textures").resolve()
    paths = []
    for name in args.textures:
        relative = Path(name)
        if relative.parts and relative.parts[0] == "textures":
            relative = Path(*relative.parts[1:])  # Also accept textures/wood.jpg.
        path = (textures_dir / relative).resolve()
        if textures_dir not in path.parents or not path.is_file():
            parser.error(f"Expected an image inside {textures_dir}: {name}")
        if path not in paths:
            paths.append(path)
    # Check every input before starting any expensive training.
    for path in paths:
        with Image.open(path) as image:
            image.verify()

    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | {len(paths)} textures | {3 * len(paths)} fresh fits")
    quantized_label = "uint8 grids + MLP" if args.quantize_mlp else "uint8 grids; float32 MLP"
    with (output_dir / "settings.json").open("w", encoding="utf-8") as file:
        json.dump({"textures": [p.relative_to(textures_dir).as_posix() for p in paths],
                   "steps": args.steps, "batch_size": args.batch_size, "learning_rate": 0.01,
                   "quantize_mlp": args.quantize_mlp, "models": MODELS, "device": device,
                   "initialization_seed": 0, "minibatch_seed": 1}, file, indent=2)

    results, history = [], []
    columns = min(3, len(paths))
    rows = (len(paths) + columns - 1) // columns
    size_fig, size_axes = plt.subplots(rows, columns, squeeze=False,
                                      figsize=(6 * columns, 5 * rows), layout="constrained")
    train_fig, train_axes = plt.subplots(rows, columns, squeeze=False,
                                        figsize=(6 * columns, 5 * rows), layout="constrained")
    size_fig.suptitle(f"Size vs. quality | quantized: {quantized_label}")
    train_fig.suptitle("Full-image PSNR during training | x = post-training quantization")
    for axes in (size_axes, train_axes):
        for ax in axes.flat[len(paths):]:
            ax.set_visible(False)

    for texture_index, path in enumerate(paths):
        texture_name = path.relative_to(textures_dir).as_posix()
        # Numbering prevents collisions between images that share a filename.
        texture_dir = output_dir / f"{texture_index + 1:02d}_{path.stem}"
        texture_dir.mkdir(exist_ok=True)
        texture = load_texture(path)
        h, w = texture.shape[:2]
        target_rgb = texture.astype(np.float32) / 255.0
        raw_bytes = texture.nbytes
        Image.fromarray(texture).save(texture_dir / "original.png")
        coords, target = make_training_data(texture, device)
        u = ((np.arange(w) + 0.5) / w)[None, :]
        v = ((np.arange(h) + 0.5) / h)[:, None]
        print(f"\n{texture_name}: {w} x {h}, raw {raw_bytes / 1000:.3f} KB")

        size_ax, train_ax = size_axes.flat[texture_index], train_axes.flat[texture_index]
        comparison_fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
        comparison_fig.suptitle(f"{texture_name} | quantized: {quantized_label}")
        for ax in axes.flat:
            ax.set_axis_off()
        axes[0, 0].imshow(texture, interpolation="nearest")
        axes[0, 0].set_title(f"Original\n{raw_bytes / 1000:.3f} KB")

        # Classical baseline. blocks.nbytes includes padding for partial 4x4 blocks.
        baseline = DXT1Texture().compress(texture)
        baseline_rgb = baseline.sample(u, v)
        baseline_psnr, baseline_bytes = psnr(baseline_rgb, target_rgb), baseline.blocks.nbytes
        baseline.blocks.tofile(texture_dir / "dxt1_blocks.bin")
        Image.fromarray(np.rint(baseline_rgb * 255).clip(0, 255).astype(np.uint8)).save(texture_dir / "dxt1.png")
        axes[1, 0].imshow(baseline_rgb, interpolation="nearest")
        axes[1, 0].set_title(f"DXT1\n{baseline_psnr:.2f} dB | {baseline_bytes / 1000:.3f} KB")
        texture_results = [("DXT1", "BC1", baseline_psnr, baseline_bytes, "")]
        float_sizes, float_psnrs, quant_sizes, quant_psnrs = [], [], [], []

        for model_index, (model_name, resolutions, feat_dim) in enumerate(MODELS):
            torch.manual_seed(0)
            model = NeuralTexture(resolutions, feat_dim).to(device)  # Always a fresh fit.
            model.height, model.width = h, w
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            torch.manual_seed(1)  # Same minibatch sequence across architectures.
            plot_steps, plot_psnrs = [], []

            # The training loop from P5, with full-image evaluation every 100 steps.
            for step in range(args.steps):
                indices = torch.randint(len(coords), (args.batch_size,), device=device)
                prediction = model(coords[indices])
                loss = F.mse_loss(prediction, target[indices])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
                    before = model.sample(u, v, batch_size=args.batch_size)
                    psnr_before = psnr(before, target_rgb)
                    plot_steps.append(step + 1)
                    plot_psnrs.append(psnr_before)
                    history.append([texture_name, model_name, step + 1, psnr_before])
                    print(f"  {model_name}: step {step + 1}/{args.steps}, PSNR {psnr_before:.3f} dB")

            model.eval()
            name = model_name.lower()
            bytes_before = sum(p.numel() * p.element_size() for p in model.parameters())
            torch.save({"state_dict": model.state_dict(), "resolutions": resolutions,
                        "feat_dim": feat_dim, "height": h, "width": w}, texture_dir / f"{name}_float32.pt")

            # Save q, lo, scale, and any unquantized MLP arrays. Decode via x_hat.
            stored = quantize_model(model, quantize_mlp=args.quantize_mlp)
            after = model.sample(u, v, batch_size=args.batch_size)
            psnr_after = psnr(after, target_rgb)
            psnr_loss = 0.0 if psnr_before == psnr_after else psnr_before - psnr_after
            bytes_after = sum(t.numel() * t.element_size()
                              for array in stored.values() for t in array.values())
            torch.save({"parameters": stored, "resolutions": resolutions, "feat_dim": feat_dim,
                        "height": h, "width": w, "quantize_mlp": args.quantize_mlp},
                       texture_dir / f"{name}_uint8.pt")
            texture_results.extend([(model_name, "float32", psnr_before, bytes_before, ""),
                                    (model_name, quantized_label, psnr_after, bytes_after, psnr_loss)])
            print(f"    Quantized: {psnr_after:.3f} dB (loss {psnr_loss:.3f}), "
                  f"{bytes_before / 1000:.3f} -> {bytes_after / 1000:.3f} KB, "
                  f"raw/compressed {raw_bytes / bytes_after:.3f}x")

            for row, (rgb, encoding, quality, size) in enumerate([
                    (before, "float32", psnr_before, bytes_before),
                    (after, "uint8", psnr_after, bytes_after)]):
                pixels = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(pixels).save(texture_dir / f"{name}_{encoding}.png")
                axes[row, model_index + 1].imshow(rgb, interpolation="nearest")
                axes[row, model_index + 1].set_title(
                    f"{model_name} {encoding}\n{quality:.2f} dB | {size / 1000:.3f} KB")
            train_ax.plot(plot_steps, plot_psnrs, color=f"C{model_index}", label=model_name)
            train_ax.plot(args.steps, psnr_after, "x", color=f"C{model_index}", markersize=8)
            float_sizes.append(bytes_before / 1000)
            float_psnrs.append(psnr_before)
            quant_sizes.append(bytes_after / 1000)
            quant_psnrs.append(psnr_after)

        comparison_fig.savefig(texture_dir / "comparison.png", dpi=150)
        plt.close(comparison_fig)
        size_ax.plot(float_sizes, float_psnrs, "o-", label="Neural float32")
        size_ax.plot(quant_sizes, quant_psnrs, "x--", label="Neural quantized")
        for sizes, qualities, offset in [(float_sizes, float_psnrs, 8), (quant_sizes, quant_psnrs, -14)]:
            for (model_name, _, _), size, quality in zip(MODELS, sizes, qualities):
                if np.isfinite(quality):
                    size_ax.annotate(model_name[0], (size, quality),
                                     xytext=(-5 if model_name == "Small" else 5, offset),
                                     ha="right" if model_name == "Small" else "left", textcoords="offset points")
        if np.isfinite(baseline_psnr):
            size_ax.plot(baseline_bytes / 1000, baseline_psnr, "s", color="gray", label="DXT1")
            train_ax.axhline(baseline_psnr, color="gray", linestyle="--", label="DXT1")
        # Infinite PSNR cannot be drawn at a finite y position; report it explicitly.
        exact = [f"{method} ({encoding})" for method, encoding, quality, _, _ in texture_results
                 if np.isinf(quality)]
        if exact:
            size_ax.text(0.02, 0.98, "Infinite PSNR: " + ", ".join(exact),
                         transform=size_ax.transAxes, va="top", fontsize=8, wrap=True)
        size_ax.set(title=f"{texture_name}\nS=Small, M=Medium, L=Large",
                    xlabel="Stored size (KB)", ylabel="Full-image PSNR (dB)")
        train_ax.set(title=texture_name, xlabel="Training step", ylabel="Full-image PSNR (dB)")
        for ax in (size_ax, train_ax):
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            ax.margins(x=0.12, y=0.15)

        # P8 visual aid: remove each channel's mean and taper image boundaries
        # with a Hann window, so artificial seams do not dominate the spectrum.
        wy = np.hanning(h) if h > 2 else np.ones(h)
        wx = np.hanning(w) if w > 2 else np.ones(w)
        rgb64 = target_rgb.astype(np.float64)  # Avoid residual roundoff on constant images.
        centered = (rgb64 - rgb64.mean(axis=(0, 1))) * (wy[:, None] * wx[None, :])[:, :, None]
        spectrum = np.fft.fftshift(np.fft.fft2(centered, axes=(0, 1)), axes=(0, 1))
        power = np.mean(np.abs(spectrum) ** 2, axis=2)
        peak = power.max()
        power_db = 10 * np.log10(np.maximum(power / peak, 1e-6)) if peak > 0 else np.full_like(power, -60)
        frequency_fig, frequency_axes = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
        frequency_axes[0].imshow(texture, interpolation="nearest")
        frequency_axes[0].set_title("Original")
        heatmap = frequency_axes[1].imshow(power_db, cmap="magma", vmin=-60, vmax=0)
        frequency_axes[1].set_title("RGB power spectrum\nMean removed; Hann window")
        for ax in frequency_axes:
            ax.set_axis_off()
        frequency_fig.colorbar(heatmap, ax=frequency_axes[1], label="Power relative to this image's peak (dB)")
        frequency_fig.suptitle(texture_name + " | Center: low frequencies; edges: high frequencies")
        frequency_fig.savefig(texture_dir / "frequency.png", dpi=150)
        plt.close(frequency_fig)

        for method, encoding, quality, size, psnr_loss in texture_results:
            results.append([texture_name, texture_dir.name, w, h, method, encoding, quality,
                            size, size / 1000, raw_bytes, raw_bytes / 1000,
                            size / raw_bytes, raw_bytes / size, psnr_loss])
        # Save completed textures as we go. KB = 1000 bytes; headers excluded.
        with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["texture", "output_folder", "width", "height", "model", "encoding", "psnr_db",
                             "size_bytes", "size_kb", "raw_bytes", "raw_kb", "compressed_over_raw",
                             "compression_factor", "psnr_loss_db"])
            writer.writerows(results)
        with (output_dir / "psnr_history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["texture", "model", "step", "full_image_psnr_db"])
            writer.writerows(history)

    size_fig.savefig(output_dir / "psnr_vs_size.png", dpi=150)
    train_fig.savefig(output_dir / "psnr_training.png", dpi=150)
    plt.close(size_fig)
    plt.close(train_fig)
    print(f"\nDone. Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
