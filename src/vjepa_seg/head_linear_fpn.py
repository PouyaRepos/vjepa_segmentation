from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

class LinearFPNHead(nn.Module):
    """Lightweight head: 1x1 reduce -> 3x3 refine -> 1x1 logits -> upsample."""
    def __init__(self, in_dim: int, fpn_dim: int, num_classes: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, fpn_dim, kernel_size=1, bias=False)
        self.refine = ConvBNReLU(fpn_dim, fpn_dim, k=3, s=1, p=1)
        self.classifier = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        x = self.reduce(feats)
        x = self.refine(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x  # B x C x H x W
