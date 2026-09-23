"""P7: quantize the nine trained P6 models and compare quality and size.

Run after run_p6.py: python run_p7.py
Requires textures/, p6_results/*.pt, results.csv, and psnr_history.csv.
Sizes count stored arrays (including lo/scale), not .pt container overhead.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from texture import load_texture
from nncomp import NeuralTexture, quantize_model


TEXTURES = ["gradient", "bricks", "clouds"]
MODELS = ["Small", "Medium", "Large"]
QUANTIZE_MLP = False  # The assignment permits leaving the small MLP in float32.

root = Path(__file__).resolve().parent
p6_dir = root / "outputs/p6_results"
output_dir = root / ("outputs/p7_results_all_uint8" if QUANTIZE_MLP else "outputs/p7_results")
output_dir.mkdir(exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Device:", device)

with (p6_dir / "results.csv").open(newline="") as file:
    p6_results = list(csv.DictReader(file))
with (p6_dir / "psnr_history.csv").open(newline="") as file:
    history = list(csv.DictReader(file))

results = []
size_fig, size_axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
training_fig, training_axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
quantized_label = "uint8 grids + MLP" if QUANTIZE_MLP else "uint8 grids; float32 MLP"
size_fig.suptitle(f"P6 vs. P7: {quantized_label}")
training_fig.suptitle(f"P6 training with P7 post-training endpoints: {quantized_label}")

for texture_index, texture_name in enumerate(TEXTURES):
    texture = load_texture(root / "textures" / f"{texture_name}.png")
    h, w = texture.shape[:2]
    target = texture.astype(np.float32) / 255.0
    raw_bytes = texture.nbytes
    u = ((np.arange(w) + 0.5) / w)[None, :]
    v = ((np.arange(h) + 0.5) / h)[:, None]
    size_ax = size_axes[texture_index]
    training_ax = training_axes[texture_index]

    for model_index, model_name in enumerate(MODELS):
        stem = f"{texture_name}_{model_name.lower()}"
        checkpoint = torch.load(p6_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
        model = NeuralTexture(checkpoint["resolutions"], checkpoint["feat_dim"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.height, model.width = checkpoint["height"], checkpoint["width"]
        if (model.height, model.width) != (h, w):
            raise ValueError(f"{stem}: texture dimensions differ from the P6 checkpoint.")
        model.eval()

        # Measure every texel before and after quantization, without retraining.
        before = model.sample(u, v)
        mse_before = np.mean((before - target) ** 2)
        psnr_before = float("inf") if mse_before == 0 else -10 * np.log10(mse_before)
        bytes_before = sum(p.numel() * p.element_size() for p in model.parameters())

        stored = quantize_model(model, quantize_mlp=QUANTIZE_MLP)
        after = model.sample(u, v)
        mse_after = np.mean((after - target) ** 2)
        psnr_after = float("inf") if mse_after == 0 else -10 * np.log10(mse_after)
        psnr_loss = 0.0 if psnr_before == psnr_after else psnr_before - psnr_after

        # Includes one byte per q and 8 bytes of lo/scale per quantized array,
        # plus 4 bytes per MLP value when QUANTIZE_MLP is False.
        bytes_after = sum(t.numel() * t.element_size()
                          for array in stored.values() for t in array.values())
        results.append([texture_name, model_name, psnr_before, psnr_after, psnr_loss,
                        bytes_before / 1000, bytes_after / 1000, raw_bytes / 1000,
                        bytes_before / raw_bytes, bytes_after / raw_bytes,
                        raw_bytes / bytes_after])
        print(f"{texture_name} / {model_name}: {psnr_before:.3f} -> {psnr_after:.3f} dB "
              f"(loss {psnr_loss:.3f} dB), {bytes_before / 1000:.3f} -> {bytes_after / 1000:.3f} KB, "
              f"compressed/raw {bytes_after / raw_bytes:.4f}, factor {raw_bytes / bytes_after:.2f}x")

        # Save the actual uint8 encoding, not the dequantized model.state_dict().
        torch.save({"parameters": stored, "resolutions": checkpoint["resolutions"],
                    "feat_dim": checkpoint["feat_dim"], "height": h, "width": w,
                    "quantize_mlp": QUANTIZE_MLP}, output_dir / f"{stem}_uint8.pt")
        pixels = np.rint(np.clip(after, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(output_dir / f"{stem}_uint8.png")

        color = f"C{model_index}"
        size_ax.plot([bytes_before / 1000, bytes_after / 1000],
                     [psnr_before, psnr_after], color=color, alpha=0.4)
        size_ax.plot(bytes_before / 1000, psnr_before, "o", color=color, label=f"{model_name} float32")
        size_ax.plot(bytes_after / 1000, psnr_after, "x", color=color, markersize=8,
                     label=f"{model_name} quantized")

        curve = [row for row in history
                 if row["texture"] == texture_name and row["model"] == model_name]
        steps = [int(row["step"]) for row in curve]
        training_ax.plot(steps, [float(row["full_image_psnr_db"]) for row in curve],
                         color=color, label=model_name)
        # The x marker is post-training quantization, not another gradient step.
        training_ax.plot(steps[-1], psnr_after, "x", color=color, markersize=9,
                         label="Quantized final" if model_index == 0 else None)

    baseline = next(row for row in p6_results
                    if row["texture"] == texture_name and row["model"] == "DXT1")
    baseline_psnr = float(baseline["psnr_db"])
    if np.isfinite(baseline_psnr):
        size_ax.plot(float(baseline["compressed_kb"]), baseline_psnr, "s", color="gray", label="DXT1")
        training_ax.axhline(baseline_psnr, color="gray", linestyle="--", label="DXT1")
    else:
        size_ax.text(0.02, 0.98, "DXT1: exact reconstruction (infinite PSNR)",
                     transform=size_ax.transAxes, va="top", fontsize=8)
    size_ax.set(title=texture_name.capitalize(), xlabel="Stored size (KB)", ylabel="Full-image PSNR (dB)")
    training_ax.set(title=texture_name.capitalize(), xlabel="Training step", ylabel="Full-image PSNR (dB)")
    for ax in (size_ax, training_ax):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

size_fig.savefig(output_dir / "psnr_vs_size.png", dpi=150)
training_fig.savefig(output_dir / "psnr_training_quantized.png", dpi=150)
plt.close(size_fig)
plt.close(training_fig)

# KB = 1000 bytes. Ratio = compressed/raw; factor = raw/compressed.
with (output_dir / "results.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["texture", "model", "psnr_before_db", "psnr_after_db", "psnr_loss_db",
                     "before_kb", "after_kb", "raw_kb", "before_over_raw", "after_over_raw",
                     "compression_factor"])
    writer.writerows(results)

print(f"\nDone. Quantized models, images, plots, and results are in {output_dir}")
