
from __future__ import annotations
import numpy as np
from PIL import Image

TRYON6_PALETTE = np.array(
    [
        [0, 0, 0], # bg
        [153, 0, 204], # hair
        [255, 224, 189], # face (skin-like)
        [0, 128, 255], # torso
        [0, 200, 0], # arms
        [255, 0, 0], # legs
    ],
    dtype=np.uint8,
)


def colorize(mask_np: np.ndarray, palette: str = "tryon6") -> Image.Image:
    if palette == "tryon6":
        pal = TRYON6_PALETTE
    else:
        raise ValueError("unknown palette")
    h, w = mask_np.shape
    rgb = pal[mask_np.clip(0, len(pal) - 1)]
    return Image.fromarray(rgb)
