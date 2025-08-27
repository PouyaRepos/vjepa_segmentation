
from __future__ import annotations
import random
from typing import Tuple
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF




def _letterbox(img: Image.Image, target: Tuple[int, int], fill=(0, 0, 0)):
    w, h = img.size
    tw, th = target
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (tw, th), fill)
    px = (tw - nw) // 2
    py = (th - nh) // 2
    canvas.paste(img_resized, (px, py))
    return canvas, (px, py, nw, nh)




def preprocess_image(img: Image.Image, size=(512, 512), flip=False, color_jitter=0.0):
    # basic aug
    if flip and random.random() < 0.5:
        img = TF.hflip(img)
    if color_jitter > 0:
    # simple brightness jitter
        b = 1.0 + random.uniform(-color_jitter, color_jitter)
        img = TF.adjust_brightness(img, b)
    img, _ = _letterbox(img, size)
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img




def preprocess_mask(mask: Image.Image, size=(512, 512)):
    mask = mask.convert("L")
    mask = mask.resize(size, Image.NEAREST)
    return np.array(mask, dtype=np.int64)

