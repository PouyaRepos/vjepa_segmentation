# src/vjepa_seg/backbone_vjepa.py
from __future__ import annotations
import torch
import torch.nn as nn

class DummyFrozenBackbone(nn.Module):
    """Stand-in: B x C x (H/16) x (W/16)."""
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
    impl='vjepa' -> fetch V-JEPA2 ViT via torch.hub (code only), load your local checkpoint,
                    return B x C x (H/16) x (W/16) features.
    """
    def __init__(self, impl: str = "dummy", out_dim: int = 1024,
                 vjepa_ckpt: str | None = None, patch_size: int = 16, expect_frames: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.impl = impl

        if impl == "dummy":
            self.backbone = DummyFrozenBackbone(out_dim)
        elif impl == "vjepa":
            if not vjepa_ckpt:
                raise ValueError("When impl='vjepa', pass --vjepa_ckpt /abs/path/to/checkpoint.pt")

            # 1) Try local import; else fall back to torch.hub (code only, no weights).
            vt_class = None
            try:
                from src.models.vision_transformer import VisionTransformer  # type: ignore[import-not-found]
                vt_class = VisionTransformer
            except Exception:
                vt_class = None

            if vt_class is None:
                obj = torch.hub.load(
                    "facebookresearch/vjepa2",
                    "vjepa2_vit_large",   # ViT-L/16 entry; hub may return a module OR a tuple/dict
                    pretrained=False,
                    trust_repo=True,
                )

                # Coerce whatever Hub returned into an nn.Module
                def coerce_to_module(x):
                    if isinstance(x, nn.Module):
                        return x
                    if isinstance(x, (tuple, list)):
                        for it in x:
                            if isinstance(it, nn.Module):
                                return it
                    if isinstance(x, dict):
                        for key in ["model", "encoder", "backbone", "net", "vit", "module"]:
                            v = x.get(key)
                            if isinstance(v, nn.Module):
                                return v
                        # last resort: first nn.Module value
                        for v in x.values():
                            if isinstance(v, nn.Module):
                                return v
                    raise TypeError(f"Hub returned unsupported type: {type(x)}")

                self.vit = coerce_to_module(obj)
            else:
                # Build a ViT-L/16-ish skeleton if local class exists
                self.vit = vt_class(
                    img_size=224, patch_size=patch_size,
                    num_frames=max(1, expect_frames), tubelet_size=1,
                    in_chans=3, embed_dim=out_dim, depth=24, num_heads=16, mlp_ratio=4.0,
                    qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, norm_layer=nn.LayerNorm,
                )
            if hasattr(self.vit, "num_frames"):
                try:
                    prev = int(self.vit.num_frames)
                except Exception:
                    prev = None
                self.vit.num_frames = 1
                if prev and prev != 1:
                    print(f"[VJEPA] Overriding num_frames {prev} -> 1 to avoid large attention buffers.")

            # 2) Load your checkpoint with tolerant key-matching
            state = torch.load(vjepa_ckpt, map_location="cpu")
            # unwrap common wrappers
            for k in ("state_dict", "model"):
                if isinstance(state, dict) and k in state and isinstance(state[k], dict):
                    state = state[k]

            # strip common prefixes & keep only matching shapes
            msd = self.vit.state_dict()
            def strip_prefix(name: str) -> str:
                for pref in ("module.", "model.", "encoder.", "backbone.", "net."):
                    if name.startswith(pref):
                        name = name[len(pref):]
                return name

            filtered = {}
            for k, v in state.items():
                k2 = strip_prefix(k)
                if k2 in msd and msd[k2].shape == v.shape:
                    filtered[k2] = v

            missing, unexpected = self.vit.load_state_dict(filtered, strict=False)
            if unexpected:
                print(f"[VJEPA] Ignoring {len(unexpected)} unexpected keys from checkpoint.")
            if missing:
                print(f"[VJEPA] {len(missing)} model keys missing in checkpoint (ok if heads differ).")

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

        # Always pass T=1 to the video model
        x_in = x.unsqueeze(2)  # (B, C, 1, H, W)
        if getattr(self.backbone, "tubelet_size", 2) == 2 and x_in.shape[2] == 1:
            x_in = x_in.repeat(1, 1, 2, 1, 1)  # (B, C, 2, H, W)

        # Prefer forward_features if available
        if hasattr(self.backbone, "forward_features"):
            tokens = self.backbone.forward_features(x_in)
        else:
            tokens = self.backbone(x_in)

        # Case A: B x N x C tokens (flattened space [+ optional CLS])
        if tokens.dim() == 3:
            B, N, C = tokens.shape
            n_per_frame = Ht * Wt
            # Drop single CLS if present
            if N % n_per_frame == 1:
                tokens = tokens[:, 1:, :]
                N -= 1
            if N % n_per_frame != 0:
                raise RuntimeError(f"Unexpected token length N={N}, not divisible by Ht*Wt={n_per_frame}")
            # T == 1 now (we forced it), but keep robust:
            T = max(1, N // n_per_frame)
            tokens = tokens.view(B, T, n_per_frame, C).mean(dim=1)  # time-mean (no-op when T=1)
            feats = tokens.transpose(1, 2).contiguous().view(B, C, Ht, Wt)
            return feats

        # Case B: already spatial B x C x h x w
        if tokens.dim() == 4:
            return tokens

        # Case C: B x C x T x h x w -> mean over time
        if tokens.dim() == 5:
            return tokens.mean(dim=2)

        raise RuntimeError(f"Unexpected V-JEPA feature shape: {tokens.shape}")
