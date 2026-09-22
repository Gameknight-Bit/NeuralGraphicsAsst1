"""A simple opaque DXT1/BC1 compressor and bilinear texture sampler."""

import numpy as np


#Weird cool way to have bit-replication promotion
# https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm
def pack_rgb565(color):
    """Quantize a float RGB color in [0, 1] to one 16-bit integer."""
    r, g, b = np.rint(np.clip(color, 0.0, 1.0) * [31, 63, 31]).astype(int)
    return int((r << 11) | (g << 5) | b)


# TODO: Change this to whatever we want c0 and c1 to have!!
def unpack_rgb565(code):
    """Decode one packed RGB565 endpoint, or an array of endpoints."""
    code = np.asarray(code, dtype=np.uint16)
    r = (code >> 11) & 31
    g = (code >> 5) & 63
    b = code & 31
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    return rgb / np.array([31, 63, 31], dtype=np.float32)


def make_palette(code0, code1):
    """Build the four-color palette from quantized endpoints (code0 > code1)."""
    c0 = unpack_rgb565(code0)
    c1 = unpack_rgb565(code1)
    c2 = (2.0 * c0 + c1) / 3.0
    c3 = (c0 + 2.0 * c1) / 3.0
    return np.stack([c0, c1, c2, c3], axis=-2)


class DXT1Texture:
    """Input: (H, W, 3) uint8 RGB. Output: float32 RGB in [0, 1].
    UV endpoints correspond to corner texel centers; v increases downward.
    """

    def __init__(self):
        self.blocks = None

    def compress(self, texture):
        texture = np.asarray(texture)
        if texture.ndim != 3 or texture.shape[2] != 3:
            raise ValueError("Expected an RGB array with shape (H, W, 3).")
        if texture.shape[0] == 0 or texture.shape[1] == 0:
            raise ValueError("Texture height and width must be positive.")
        if texture.dtype != np.uint8:
            raise TypeError("Expected uint8 RGB texels in [0, 255].")

        self.height, self.width = texture.shape[:2]
        blocks_y = (self.height + 3) // 4
        blocks_x = (self.width + 3) // 4

        # Repeat edge texels to fill partial blocks. Keep the original size
        # for UV mapping, so padding never changes the sampling convention.
        padded = np.pad(
            texture,
            ((0, blocks_y * 4 - self.height),
             (0, blocks_x * 4 - self.width), (0, 0)),
            mode="edge",
        )

        # Each structured array entry is exactly 8 bytes: 2 + 2 + 4.
        # '<' specifies little-endian storage, as used by BC1 blocks.
        self.blocks = np.empty(
            (blocks_y, blocks_x),
            dtype=[("c0", "<u2"), ("c1", "<u2"), ("indices", "<u4")],
        )

        for by in range(blocks_y):
            for bx in range(blocks_x):
                block = padded[4 * by:4 * by + 4, 4 * bx:4 * bx + 4]
                colors = block.reshape(16, 3).astype(np.float32) / 255.0

                # PCA: the largest eigenvector gives the direction of
                # greatest color variation. Pick its two extreme texels.
                centered = colors - colors.mean(axis=0)
                _, axes = np.linalg.eigh(centered.T @ centered)
                projection = centered @ axes[:, -1]
                code0 = pack_rgb565(colors[np.argmax(projection)])
                code1 = pack_rgb565(colors[np.argmin(projection)])

                # BC1 requires code0 > code1 for the four-color mode.
                if code0 < code1:
                    code0, code1 = code1, code0
                if code0 == code1:
                    # Preserve one endpoint exactly for constant blocks.
                    if code0 < 65535:
                        code0 += 1
                    else:
                        code1 -= 1

                # Choose indices AFTER quantizing and ordering endpoints.
                palette = make_palette(code0, code1)
                errors = np.sum(
                    (colors[:, None, :] - palette[None, :, :]) ** 2, # squared L2 Norm
                    axis=-1,
                )
                indices = np.argmin(errors, axis=1)

                # Texel i = 4 * local_y + local_x occupies bits 2*i..2*i+1.
                packed_indices = 0
                for i in range(16):
                    packed_indices |= int(indices[i]) << (2 * i)
                self.blocks[by, bx] = (code0, code1, packed_indices)

        return self

    def _decode_texel(self, x, y):
        """Read compressed texels at integer pixel coordinates."""
        block = self.blocks[y // 4, x // 4]
        palette = make_palette(block["c0"], block["c1"])

        local_index = 4 * (y % 4) + (x % 4)
        shift = np.asarray(2 * local_index, dtype=np.uint32)
        index = ((block["indices"] >> shift) & 3).astype(np.intp)

        # Select one of the four palette entries for each requested texel.
        return np.take_along_axis(
            palette, index[..., None, None], axis=-2
        )[..., 0, :]

    def sample(self, u, v):
        if self.blocks is None:
            raise RuntimeError("Call compress(texture) before sample(u, v).")

        u, v = np.broadcast_arrays(
            np.asarray(u, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
        )
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            raise ValueError("Coordinates must be finite.")

        # The same coordinate mapping and bilinear weights as P1.
        x = np.clip(u * self.width - 0.5, 0, self.width - 1)
        y = np.clip(v * self.height - 0.5, 0, self.height - 1)

        x0 = np.floor(x).astype(np.intp)
        y0 = np.floor(y).astype(np.intp)
        x1 = np.minimum(x0 + 1, self.width - 1)
        y1 = np.minimum(y0 + 1, self.height - 1)
        s = np.expand_dims(x - x0, axis=-1)
        t = np.expand_dims(y - y0, axis=-1)

        # Each neighbor may belong to a DIFFERENT compressed block.
        c00 = self._decode_texel(x0, y0)
        c10 = self._decode_texel(x1, y0)
        c01 = self._decode_texel(x0, y1)
        c11 = self._decode_texel(x1, y1)

        top = (1.0 - s) * c00 + s * c10
        bottom = (1.0 - s) * c01 + s * c11
        return ((1.0 - t) * top + t * bottom).astype(np.float32)
