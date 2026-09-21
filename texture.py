import numpy as np
from PIL import Image


def load_texture(path):
    """Load an image as an (H, W, 3) uint8 RGB array without resizing it."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


class UncompressedTexture():
    """Store every RGB8 texel and sample it using bilinear interpolation."""

    def __init__(self):
        self.texels = None

    def compress(self, texture):
        texture = np.asarray(texture)
        if texture.ndim != 3 or texture.shape[2] != 3:
            raise ValueError("Expected an RGB array with shape (H, W, 3).")
        if texture.shape[0] == 0 or texture.shape[1] == 0:
            raise ValueError("Texture height and width must be positive.")
        if texture.dtype != np.uint8:
            raise TypeError("Expected uint8 RGB texels in [0, 255].")

        # The reference does no compression. Own a copy of every texel.
        self.texels = texture.copy()
        self.height, self.width = self.texels.shape[:2]
        return self

    def sample(self, u, v):
        if self.texels is None:
            raise RuntimeError("Call compress(texture) before sample(u, v).")

        # Broadcasting supports both single samples and batches.
        u, v = np.broadcast_arrays(
            np.asarray(u, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
        )
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            raise ValueError("Coordinates must be finite.")

        # Map normalized coordinates to continuous texel indices.
        x = np.clip(u, 0.0, 1.0) * (self.width - 1)
        y = np.clip(v, 0.0, 1.0) * (self.height - 1)

        x0 = np.floor(x).astype(np.intp)
        y0 = np.floor(y).astype(np.intp)
        x1 = np.minimum(x0 + 1, self.width - 1)
        y1 = np.minimum(y0 + 1, self.height - 1)

        # Add an RGB axis so each weight applies to all three channels.
        s = np.expand_dims(x - x0, axis=-1)
        t = np.expand_dims(y - y0, axis=-1)

        # Bilinear Interpolation
        c00 = self.texels[y0, x0].astype(np.float32) / 255.0
        c10 = self.texels[y0, x1].astype(np.float32) / 255.0
        c01 = self.texels[y1, x0].astype(np.float32) / 255.0
        c11 = self.texels[y1, x1].astype(np.float32) / 255.0

        top = (1.0 - s) * c00 + s * c10
        bottom = (1.0 - s) * c01 + s * c11
        return ((1.0 - t) * top + t * bottom).astype(np.float32)
