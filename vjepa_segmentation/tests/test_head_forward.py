
# ===========================
# tests/test_head_forward.py
# ===========================
import torch
from vjepa_seg.head_linear_fpn import LinearFPNHead

def test_head_only():
    feats = torch.randn(1, 1024, 32, 32)
    head = LinearFPNHead(1024, 256, 6)
    logits = head(feats, out_size=(512, 512))
    assert logits.shape == (1, 6, 512, 512)
