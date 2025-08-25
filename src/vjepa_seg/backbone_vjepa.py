from __future__ import annotations
import torch
import torch.nn as nn

class DummyFrozenBackbone(nn.Module):
    """Stand-in: outputs B x C x (H/16) x (W/16)."""
    def __init__(self, out_dim: int = 1024):
        super().__init__()
        self.conv = nn.Conv2d(3, out_dim, kernel_size=16, stride=16, padding=0, bias=False)
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class VJEPAFeatureBackbone(nn.Module):
    """
    impl='dummy' -> conv stand-in.
    impl='vjepa' -> import V-JEPA ViT and expose B x C x H/16 x W/16 features.
    """
    def __init__(self, impl: str = "dummy", out_dim: int = 1024, vjepa_ckpt: str | None = None,
                 patch_size: int = 16, expect_frames: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.impl = impl
        self.vjepa_ckpt = vjepa_ckpt

        if impl == "dummy":
            self.backbone = DummyFrozenBackbone(out_dim)
        elif impl == "vjepa":
            # Try to import from vjepa/vjepa2 repo (make sure PYTHONPATH points to the repo root)
            try:
                from src.models.vision_transformer import VisionTransformer  # vjepa2
                vt_class = VisionTransformer
            except Exception:
                try:
                    # some repos name it vit.py
                    from src.models.vit import VisionTransformer as VT2
                    vt_class = VT2
                except Exception as e:
                    raise ImportError(
                        "Could not import VisionTransformer from your V-JEPA repo. "
                        "Ensure PYTHONPATH includes the repo root (containing src/models/...)."
                    ) from e

            if not vjepa_ckpt:
                raise ValueError("When impl='vjepa', you must pass --vjepa_ckpt /abs/path/to/ckpt.pth")

            # Build a ViT-L-ish model. Adjust dims if your ckpt differs.
            self.vit = vt_class(
                img_size=224,           # not strictly used for reshape
                patch_size=patch_size,
                num_frames=max(1, expect_frames),
                tubelet_size=1,
                in_chans=3,
                embed_dim=out_dim,      # 1024 typical for ViT-L
                depth=24,
                num_heads=16,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                norm_layer=nn.LayerNorm,
            )

            state = torch.load(vjepa_ckpt, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.vit.load_state_dict(state, strict=False)
            self.vit.eval()
            for p in self.vit.parameters():
                p.requires_grad = False
            self.backbone = self.vit
        else:
            raise ValueError(f"Unknown backbone impl: {impl}")

        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl != "vjepa":
            return self.backbone(x)

        B, _, H, W = x.shape
        Ht, Wt = H // self.patch_size, W // self.patch_size

        # If JEPA expects (B, C, T, H, W), tile a single frame across T
        num_frames = getattr(self.backbone, "num_frames", 1)
        x_in = x.unsqueeze(2).repeat(1, 1, num_frames, 1, 1) if num_frames > 1 else x

        # Try common forward methods
        if hasattr(self.backbone, "forward_features"):
            tokens = self.backbone.forward_features(x_in)   # B x N x C (often) or B x C x h x w
        else:
            tokens = self.backbone(x_in)

        if tokens.dim() == 3:  # B x N x C
            B, N, C = tokens.shape
            # drop CLS if present
            if N == (Ht * Wt + 1):
                tokens = tokens[:, 1:, :]
                N -= 1
            tokens = tokens.transpose(1, 2).contiguous()  # B x C x N
            feats = tokens.view(B, C, Ht, Wt)
            return feats
        elif tokens.dim() == 4:  # already B x C x h x w
            return tokens
        else:
            raise RuntimeError(f"Unexpected V-JEPA feature shape: {tokens.shape}")
