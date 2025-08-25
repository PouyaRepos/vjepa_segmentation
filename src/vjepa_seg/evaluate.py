
from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader
from .utils import load_cfg, auto_device
from .backbone_vjepa import VJEPAFeatureBackbone
from .head_linear_fpn import LinearFPNHead
from .train import evaluate as eval_fn
from .datasets.cihp import CIHPDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dummy_backbone", type=int, default=1)
    ap.add_argument("--ckpt", default="ckpts/head_linear_fpn_best.pt")
    ap.add_argument("--vjepa_ckpt", type=str, default=None)

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = auto_device()

    val_set = CIHPDataset(
        root=cfg.data["root"],
        split="val",
        img_size=tuple(cfg.data.get("img_size", [512, 512])),
        cihp_to_six=cfg.data.get("cihp_to_six", {}),
        aug={"flip": False, "color_jitter": 0.0},
    )
    val_loader = DataLoader(val_set, batch_size=cfg.train.get("batch_size", 8), shuffle=False, num_workers=4)

    # impl = "dummy" if args.dummy_backbone else "vjepa"
    # model_b = VJEPAFeatureBackbone(impl=impl, out_dim=cfg.model.get("backbone_out_dim", 1024)).to(device)
    impl = "dummy" if args.dummy_backbone else "vjepa"
    model_b = VJEPAFeatureBackbone(
        impl=impl,
        out_dim=cfg.model.get("backbone_out_dim", 1024),
        vjepa_ckpt=args.vjepa_ckpt,
    ).to(device)
    model_h = LinearFPNHead(
        in_dim=cfg.model.get("backbone_out_dim", 1024),
        fpn_dim=cfg.model.get("fpn_dim", 256),
        num_classes=int(cfg.data.get("num_classes", 6)),
    ).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model_h.load_state_dict(state["model_h"])

    miou, acc = eval_fn(model_b, model_h, val_loader, device, int(cfg.data.get("num_classes", 6)))
    print(f"val mIoU={miou:.3f} | acc={acc:.3f}")


if __name__ == "__main__":
    main()
