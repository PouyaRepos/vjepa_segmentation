
from __future__ import annotations
import os
from typing import Callable, Dict, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from . .transforms import preprocess_image, preprocess_mask




class CIHPRemap:
    """Remap original CIHP 20-class labels to 6 classes used for try-on.
    Provide a mapping dict: original_id -> new_id (0..5). Unmapped -> 0 (bg).
    """


    def __init__(self, mapping: Dict[int, int]):
        self.mapping = mapping


    def __call__(self, mask_np):
        out = mask_np.copy()
        # vectorized remap
        for k, v in self.mapping.items():
            out[mask_np == k] = v
        # safety: clamp to [0..5]
        out = out.clip(0, 5)
        return out




class CIHPDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size=(512, 512),
        class_map: Dict[str, int] | None = None,
        cihp_to_six: Dict[int, int] | None = None,
        aug: Dict | None = None,
    ):
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, "Images", split)
        self.ann_dir = os.path.join(root, "Annotations", split)
        self.ids = [f[:-4] for f in os.listdir(self.img_dir) if f.endswith(".jpg")]
        self.img_size = tuple(img_size)
        self.remap = CIHPRemap(cihp_to_six or {})
        a = aug or {}
        self.flip = bool(a.get("flip", True))
        self.cj = float(a.get("color_jitter", 0.2))


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, i):
        id_ = self.ids[i]
        img = Image.open(os.path.join(self.img_dir, id_ + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.ann_dir, id_ + ".png"))
        x = preprocess_image(img, self.img_size, flip=self.flip, color_jitter=self.cj)
        y = preprocess_mask(mask, self.img_size)
        y = self.remap(y)
        y = torch.from_numpy(y).long()
        return x, y