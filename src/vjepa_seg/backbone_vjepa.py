
from __future__ import annotations
import torch
import torch.nn as nn




class DummyFrozenBackbone(nn.Module):
    """A placeholder backbone that mimics a ViT-L/16 spatial downsample of 16x.
    Output: B x C x H/16 x W/16 feature map.
    """


def __init__(self, out_dim: int = 1024):
    super().__init__()
    self.out_dim = out_dim
    self.conv = nn.Conv2d(3, out_dim, kernel_size=16, stride=16, padding=0, bias=False)
    for p in self.parameters():
        p.requires_grad = False


def forward(self, x: torch.Tensor) -> torch.Tensor:
# x: B x 3 x H x W
    return self.conv(x) # B x C x H/16 x W/16




class VJEPAFeatureBackbone(nn.Module):
    """Wrapper for a frozen V-JEPA ViT-L/16 backbone.


    For now, this class defaults to DummyFrozenBackbone unless `impl="vjepa"` and
    actual V-JEPA weights are wired up by the user.
    """


def __init__(self, impl: str = "dummy", out_dim: int = 1024):
    super().__init__()
    if impl == "dummy":
     self.backbone = DummyFrozenBackbone(out_dim)
    elif impl == "vjepa":
        # TODO: hook actual V-JEPA model loading here.
        # Raise for clarity to prompt user to connect weights.
        raise NotImplementedError(
    "V-JEPA weights not wired. Use impl='dummy' for smoke tests or add loader."
    )
    else:
        raise ValueError(f"Unknown backbone impl: {impl}")


    for p in self.backbone.parameters():
        p.requires_grad = False


def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.backbone(x)

