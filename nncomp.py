import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Grid to extract features from 
class FeatureGrid(nn.Module):
    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2):
        super().__init__()

        #Generate initially with random grid of vals.
        self.grids = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(1, feat_dim, r, r))
            for r in resolutions
        ])
        self.out_dim = feat_dim * len(resolutions)

    def forward(self, uv):
        # uv: (N, 2), with columns (u, v).
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError("Expected uv with shape (N, 2).")

        # Convert coordinates to [-1, 1].
        coords = (2.0 * uv - 1.0).reshape(1, -1, 1, 2)

        features = []
        for grid in self.grids:
            sampled = F.grid_sample(
                grid,
                coords,
                mode="bilinear", #Required setting
                padding_mode="border", #Required setting
                align_corners=False, #Required setting
            )  # (1, feat_dim, N, 1)

            # Convert to (N, feat_dim).
            features.append(sampled[0, :, :, 0].transpose(0, 1))

        return torch.cat(features, dim=1)  # (N, out_dim)

# Maps concatenated features to (r,g,b) color.
class ColorMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        #Just what P4 describes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

#make data for training on particular texture
def make_training_data(texture, device):
    #Pair every RGB8 texel with its fixed center coordinate, in row order.
    texture = np.asarray(texture)
    if texture.ndim != 3 or texture.shape[2] != 3:
        raise ValueError("Expected an RGB array with shape (H, W, 3).")
    if texture.shape[0] == 0 or texture.shape[1] == 0:
        raise ValueError("Texture height and width must be positive.")
    if texture.dtype != np.uint8:
        raise TypeError("Expected uint8 RGB texels in [0, 255].")

    h, w = texture.shape[:2]
    u = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    v = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    # Both reshapes visit row 0 left-to-right, then row 1, etc.
    coords = torch.stack((uu, vv), dim=-1).reshape(-1, 2)
    target = torch.tensor(texture, device=device, dtype=torch.float32)
    target = target.reshape(-1, 3) / 255.0
    return coords, target

#Parent class for neural texture pipeline
class NeuralTexture(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = FeatureGrid()
        self.mlp  = ColorMLP(self.grid.out_dim)
        self.height = None
        self.width = None

    def forward(self, uv):
        return self.mlp(self.grid(uv))

    def compress(self, texture, steps=2000, batch_size=16384,
                 lr=1e-2, log_every=200):
        """
        Move the model to the desired device with .to(device) first.
        Create a NEW NeuralTexture for each new texture. Calling compress
        again on this instance continues from its existing learned weights.
        """
        if steps < 1 or batch_size < 1:
            raise ValueError("steps and batch_size must be positive.")

        device = next(self.parameters()).device
        # Build the entire fixed training set ONCE, before the loop.
        coords, target = make_training_data(texture, device)
        self.height, self.width = np.asarray(texture).shape[:2]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        for step in range(steps):
            # Sample texel IDs with replacement, not new continuous UVs.
            indices = torch.randint(
                coords.shape[0], (batch_size,), device=device
            )
            opt.zero_grad(set_to_none=True)
            prediction = self(coords[indices])
            loss = F.mse_loss(prediction, target[indices])
            loss.backward()
            opt.step()

            #Logging for testing purposes
            if log_every and (step == 0 or (step + 1) % log_every == 0
                              or step + 1 == steps):
                psnr = -10.0 * torch.log10(loss.detach())
                print(
                    f"Step {step + 1:4d}/{steps} | "
                    f"batch MSE {loss.item():.6f} | "
                    f"batch PSNR {psnr.item():.2f} dB"
                )

        self.eval()
        # Training data and optimizer are local, not part of the encoding.
        return self

    @torch.no_grad()
    def sample(self, u, v, batch_size=16384):
        #Uses batches to limit GPU memory usage when reconstructing the texture. And should be compatible with
        #texture class written in P1.

        if self.height is None:
            raise RuntimeError("Call compress(texture) before sample(u, v).")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        u, v = np.broadcast_arrays(
            np.asarray(u, dtype=np.float32),
            np.asarray(v, dtype=np.float32),
        )
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            raise ValueError("Coordinates must be finite.")
        coords = np.stack((u, v), axis=-1).reshape(-1, 2)
        coords = np.clip(coords, 0.0, 1.0)

        device = next(self.parameters()).device
        colors = np.empty((len(coords), 3), dtype=np.float32)
        for start in range(0, len(coords), batch_size):
            batch = torch.as_tensor(
                coords[start:start + batch_size], device=device
            )
            colors[start:start + batch_size] = self(batch).cpu().numpy()
        return colors.reshape(u.shape + (3,))