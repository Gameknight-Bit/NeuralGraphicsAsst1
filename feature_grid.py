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