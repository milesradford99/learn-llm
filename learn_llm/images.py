"""Small images on the page: digits and weight maps, as PNGs built without extra packages."""

import base64
import struct
import zlib

import numpy as np


def png(rgb: np.ndarray) -> bytes:
    """Encode an (h, w, 3) uint8 array as a PNG file."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _ink(image: np.ndarray) -> np.ndarray:
    """0 = white paper, 255 = black ink, as in the dataset."""
    grey = 255 - np.asarray(image, dtype=np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def _signed(weights: np.ndarray) -> np.ndarray:
    """Positive weights blue, negative red, zero white; scaled by the largest size."""
    w = np.asarray(weights, dtype=float)
    top = np.abs(w).max() or 1.0
    t = np.clip(w / top, -1, 1)[:, :, None]
    white, blue, red = np.array([255, 255, 255]), np.array([37, 99, 235]), np.array([220, 38, 38])
    rgb = np.where(t >= 0, white + t * (blue - white), white + (-t) * (red - white))
    return rgb.astype(np.uint8)


def _tag(rgb: np.ndarray, caption: str, size: int) -> str:
    data = base64.b64encode(png(rgb)).decode()
    return (
        '<figure style="margin:0;text-align:center">'
        f'<img src="data:image/png;base64,{data}" width="{size}" height="{size}" '
        f'style="image-rendering:pixelated;border:1px solid var(--border,#e2e8f0);border-radius:4px" alt="{caption}">'
        f'<figcaption style="font-size:0.8rem;color:var(--muted-foreground,#64748b)">{caption}</figcaption></figure>'
    )


def row(tiles: list[str]):
    import marimo as mo

    return mo.Html('<div style="display:flex;flex-wrap:wrap;gap:10px">' + "".join(tiles) + "</div>")


def digits(images, captions, size: int = 84):
    """A row of digit images with a caption under each."""
    return row([_tag(_ink(img), str(c), size) for img, c in zip(images, captions)])


def weight_maps(columns, captions, size: int = 84):
    """Weight columns of length 784, each shown as a 28 × 28 map (blue +, red −)."""
    return row([_tag(_signed(np.reshape(col, (28, 28))), str(c), size) for col, c in zip(columns, captions)])


def maps(arrays, captions, size: int = 84):
    """Grids of any shape, shown as maps (blue +, red −)."""
    return row([_tag(_signed(a), str(c), size) for a, c in zip(arrays, captions)])
