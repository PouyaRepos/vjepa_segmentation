
# ===========================
# tests/test_shapes.py
# ===========================
import torch
from vjepa_seg.backbone_vjepa import VJEPAFeatureBackbone
from vjepa_seg.head_linear_fpn import LinearFPNHead

def test_forward_shapes():
    b, c, h, w = 2, 3, 512, 512
    x = torch.randn(b, c, h, w)
    bb = VJEPAFeatureBackbone(impl="dummy", out_dim=1024)
    feats = bb(x)
    assert feats.shape == (b, 1024, h // 16, w // 16)
    head = LinearFPNHead(1024, 256, 6)
    logits = head(feats, out_size=(h, w))
    assert logits.shape == (b, 6, h, w)
