import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from texture import load_texture
from s3tc import DXT1Texture
from nncomp import NeuralTexture


TEXTURES = ["gradient", "bricks", "clouds"]
MODELS = [
    ("Small", (64,), 2),
    ("Medium", (16, 32, 64), 2),
    ("Large", (16, 32, 64, 128), 4),
]
STEPS = 2000
BATCH_SIZE = 16384

root = Path(__file__).resolve().parent
output_dir = root / "outputs/p6_results"
output_dir.mkdir(exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Device:", device)

results = []
history = []
curves_fig, curves_axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

for texture_index, texture_name in enumerate(TEXTURES):
    texture = load_texture(root / "textures" / f"{texture_name}.png")
    h, w = texture.shape[:2]
    raw_bytes = texture.nbytes  # H * W * 3 for RGB8

    # Build every texel-center coordinate and its target color once.
    u = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    v = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    coords = torch.stack((uu, vv), dim=-1).reshape(-1, 2)
    target = torch.tensor(texture, device=device, dtype=torch.float32).reshape(-1, 3) / 255.0

    # Each row shows the original next to DXT1, Small, Medium, or Large.
    comparison_fig, axes = plt.subplots(4, 2, figsize=(9, 15), layout="constrained")
    comparison_fig.suptitle(texture_name.capitalize())
    for ax in axes.flat:
        ax.set_axis_off()
    for ax in axes[:, 0]:
        ax.imshow(texture, interpolation="nearest")
        ax.set_title(f"Original | {raw_bytes / 1000:.3f} KB")

    # P2 baseline, measured at the same texel centers.
    baseline = DXT1Texture().compress(texture)
    u_np = ((np.arange(w) + 0.5) / w)[None, :]
    v_np = ((np.arange(h) + 0.5) / h)[:, None]
    baseline_rgb = baseline.sample(u_np, v_np)
    baseline_mse = np.mean((baseline_rgb - texture.astype(np.float32) / 255.0) ** 2)
    baseline_psnr = float("inf") if baseline_mse == 0 else -10 * np.log10(baseline_mse)
    baseline_bytes = baseline.blocks.nbytes
    results.append([texture_name, "DXT1", baseline_psnr, baseline_bytes / 1000,
                    raw_bytes / 1000, baseline_bytes / raw_bytes])
    axes[0, 1].imshow(baseline_rgb, interpolation="nearest")
    axes[0, 1].set_title(f"DXT1 | {baseline_psnr:.2f} dB | {baseline_bytes / 1000:.3f} KB\n"
                         f"Compressed / raw: {baseline_bytes / raw_bytes:.4f}")
    print(f"\n{texture_name}: DXT1 PSNR = {baseline_psnr:.2f} dB")

    for row, (model_name, resolutions, feat_dim) in enumerate(MODELS, start=1):
        torch.manual_seed(0)
        model = NeuralTexture(resolutions=resolutions, feat_dim=feat_dim).to(device)
        model.height, model.width = h, w
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.manual_seed(1)  # Same minibatch sequence for each architecture.
        plot_steps, plot_psnr = [], []

        # This is the training loop from the assignment.
        for step in range(STEPS):
            indices = torch.randint(len(coords), (BATCH_SIZE,), device=device)
            prediction = model(coords[indices])
            loss = F.mse_loss(prediction, target[indices])
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Every 100 steps, measure full-image PSNR for the plot.
            if step == 0 or (step + 1) % 100 == 0 or step + 1 == STEPS:
                with torch.no_grad():
                    full_prediction = torch.cat([model(batch) for batch in coords.split(BATCH_SIZE)])
                    mse = F.mse_loss(full_prediction, target)
                    psnr = (-10 * torch.log10(mse)).item()
                plot_steps.append(step + 1)
                plot_psnr.append(psnr)
                history.append([texture_name, model_name, step + 1, psnr])
                print(f"{texture_name} / {model_name}: step {step + 1}, PSNR {psnr:.2f} dB")

        model.eval()
        curves_axes[texture_index].plot(plot_steps, plot_psnr, label=model_name)

        # The final evaluation above uses the fully trained model.
        reconstruction = full_prediction.reshape(h, w, 3).cpu().numpy()
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        ratio = size_bytes / raw_bytes
        results.append([texture_name, model_name, psnr, size_bytes / 1000,
                        raw_bytes / 1000, ratio])

        axes[row, 1].imshow(reconstruction, interpolation="nearest")
        axes[row, 1].set_title(f"{model_name} | {psnr:.2f} dB | {size_bytes / 1000:.3f} KB\n"
                              f"Compressed / raw: {ratio:.4f}")
        pixels = np.rint(np.clip(reconstruction, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(output_dir / f"{texture_name}_{model_name.lower()}.png")
        torch.save({"state_dict": model.state_dict(), "resolutions": resolutions,
                    "feat_dim": feat_dim, "height": h, "width": w},
                   output_dir / f"{texture_name}_{model_name.lower()}.pt")

    comparison_fig.savefig(output_dir / f"{texture_name}_comparison.png", dpi=150)
    plt.close(comparison_fig)
    ax = curves_axes[texture_index]
    if np.isfinite(baseline_psnr):
        ax.axhline(baseline_psnr, color="gray", linestyle="--", label="DXT1")
    ax.set(title=texture_name.capitalize(), xlabel="Training step", ylabel="Full-image PSNR (dB)")
    ax.legend()
    ax.grid(alpha=0.2)

curves_fig.savefig(output_dir / "psnr_training.png", dpi=150)
plt.close(curves_fig)

# KB = 1000 bytes. Compression ratio = compressed bytes / raw RGB8 bytes.
with (output_dir / "results.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["texture", "model", "psnr_db", "compressed_kb", "raw_kb", "compressed_over_raw"])
    writer.writerows(results)
with (output_dir / "psnr_history.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["texture", "model", "step", "full_image_psnr_db"])
    writer.writerows(history)

print(f"\nDone. Images, plots, and results are in {output_dir}")