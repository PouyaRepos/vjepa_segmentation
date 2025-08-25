


from __future__ import annotations
import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .utils import load_cfg, set_seed, auto_device, AvgMeter
from .backbone_vjepa import VJEPAFeatureBackbone
from .head_linear_fpn import LinearFPNHead
from .datasets.cihp import CIHPDataset


def miou(pred, target, num_classes: int):
    # pred: BxHxW (int), target: BxHxW
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return sum(ious) / max(1, len(ious))


def train_one_epoch(model_b, model_h, loader, criterion, optimizers, device, num_classes):
    model_b.eval()
    model_h.train()
    opt = optimizers
    loss_meter = AvgMeter()
    miou_meter = AvgMeter()
    acc_meter = AvgMeter()

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            feats = model_b(x)
        logits = model_h(feats, out_size=(x.shape[2], x.shape[3]))
        loss = criterion(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(1)
            m = miou(pred, y, num_classes)
            acc = (pred == y).float().mean().item()
        loss_meter.update(loss.item(), x.size(0))
        miou_meter.update(m, x.size(0))
        acc_meter.update(acc, x.size(0))

    return loss_meter.avg, miou_meter.avg, acc_meter.avg


def evaluate(model_b, model_h, loader, device, num_classes):
    model_b.eval()
    model_h.eval()
    miou_meter = AvgMeter()
    acc_meter = AvgMeter()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=False):
            x = x.to(device)
            y = y.to(device)
            feats = model_b(x)
            logits = model_h(feats, out_size=(x.shape[2], x.shape[3]))
            pred = logits.argmax(1)
            m = miou(pred, y, num_classes)
            acc = (pred == y).float().mean().item()
            miou_meter.update(m, x.size(0))
            acc_meter.update(acc, x.size(0))
    return miou_meter.avg, acc_meter.avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dummy_backbone", type=int, default=0)
    ap.add_argument("--vjepa_ckpt", type=str, default=None)

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(42)
    device = auto_device()

    # dataset
    train_set = CIHPDataset(
        root=cfg.data["root"],
        split="train",
        img_size=tuple(cfg.data.get("img_size", [512, 512])),
        cihp_to_six=cfg.data.get("cihp_to_six", {}),
        aug=cfg.data.get("aug", {}),
    )
    val_set = CIHPDataset(
        root=cfg.data["root"],
        split="val",
        img_size=tuple(cfg.data.get("img_size", [512, 512])),
        cihp_to_six=cfg.data.get("cihp_to_six", {}),
        aug={"flip": False, "color_jitter": 0.0},
    )

    num_classes = int(cfg.data.get("num_classes", 6))

    train_loader = DataLoader(train_set, batch_size=cfg.train.get("batch_size", 8), shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=cfg.train.get("batch_size", 8), shuffle=False, num_workers=4)

    # models
    # impl = "dummy" if args.dummy_backbone else "vjepa"
    # model_b = VJEPAFeatureBackbone(impl=impl, out_dim=cfg.model.get("backbone_out_dim", 1024)).to(device)
    # if impl == "vjepa":
    #     # until wired, this raises NotImplementedError
    #     try:
    #         _ = model_b(torch.zeros(1, 3, *cfg.data.get("img_size", [512, 512])).to(device))
    #     except NotImplementedError as e:
    #         print(str(e))
    #         print("Falling back to dummy backbone. Use --dummy_backbone 1 for CI/quickstart.")
    #         model_b = VJEPAFeatureBackbone(impl="dummy", out_dim=cfg.model.get("backbone_out_dim", 1024)).to(device)
    
    impl = "dummy" if args.dummy_backbone else "vjepa"
    model_b = VJEPAFeatureBackbone(
        impl=impl,
        out_dim=cfg.model.get("backbone_out_dim", 1024),
        vjepa_ckpt=args.vjepa_ckpt,  # <— NEW: pass ckpt
    ).to(device)

    model_h = LinearFPNHead(
        in_dim=cfg.model.get("backbone_out_dim", 1024),
        fpn_dim=cfg.model.get("fpn_dim", 256),
        num_classes=num_classes,
    ).to(device)

    model_h = LinearFPNHead(
        in_dim=cfg.model.get("backbone_out_dim", 1024),
        fpn_dim=cfg.model.get("fpn_dim", 256),
        num_classes=num_classes,
    ).to(device)

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    opt = optim.AdamW(model_h.parameters(), lr=cfg.train.get("lr", 3e-4), weight_decay=cfg.train.get("weight_decay", 1e-4))

    best_miou = -1.0
    epochs = cfg.train.get("epochs", 10)
    for ep in range(1, epochs + 1):
        tr_loss, tr_miou, tr_acc = train_one_epoch(model_b, model_h, train_loader, criterion, opt, device, num_classes)
        va_miou, va_acc = evaluate(model_b, model_h, val_loader, device, num_classes)
        print(f"epoch {ep:03d} | loss {tr_loss:.4f} | tr_mIoU {tr_miou:.3f} | tr_acc {tr_acc:.3f} | val_mIoU {va_miou:.3f} | val_acc {va_acc:.3f}")
        if va_miou > best_miou:
            best_miou = va_miou
            os.makedirs("ckpts", exist_ok=True)
            torch.save({"model_h": model_h.state_dict()}, os.path.join("ckpts", "head_linear_fpn_best.pt"))
            print(f"saved ckpts/head_linear_fpn_best.pt (mIoU={best_miou:.3f})")


if __name__ == "__main__":
    main()
