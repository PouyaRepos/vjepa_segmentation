
from __future__ import annotations
import os
import random
import yaml
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import torch




def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




@dataclass
class Cfg:
    data: Dict[str, Any]
    model: Dict[str, Any]
    train: Dict[str, Any]
    eval: Dict[str, Any]
    infer: Dict[str, Any]




def load_cfg(path: str) -> Cfg:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Cfg(**cfg)




def auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




class AvgMeter:
    def __init__(self):
        self.n = 0
        self.s = 0.0


def update(self, val: float, k: int = 1):
    self.s += val * k
    self.n += k


@property
def avg(self) -> float:
    return self.s / max(1, self.n)