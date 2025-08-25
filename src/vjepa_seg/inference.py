
from __future__ import annotations
import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from .utils import load_cfg, auto_device
from .backbone_vjepa import VJEPAFeatureBackbone
from .head_linear_fpn import LinearFPNHead
from .transforms import preprocess_image
from .viz import colorize


def ema_update(prev, cur, alpha):
    return alpha * cur + (1 - alpha) * prev


def run_image(cfg, image_path, out_path, use_dummy, ckpt_path, vjepa_ckpt):
    device = auto_device()
    impl = "dummy" if use_dummy else "vjepa"
    model_b = VJEPAFeatureBackbone(
        impl=impl,
        out_dim=cfg.model.get("backbone_out_dim", 1024),
        vjepa_ckpt=vjepa_ckpt,
    ).to(device)
    model_h = LinearFPNHead(
        in_dim=cfg.model.get("backbone_out_dim", 1024),
        fpn_dim=cfg.model.get("fpn_dim", 256),
        num_classes=int(cfg.data.get("num_classes", 6)),
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    model_h.load_state_dict(state["model_h"])

    model_b.eval(); model_h.eval()

    img = Image.open(image_path).convert("RGB")
    x = preprocess_image(img, tuple(cfg.data.get("img_size", [512, 512])), flip=False, color_jitter=0.0)
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model_b(x)
        logits = model_h(feats, out_size=(x.shape[2], x.shape[3]))
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    color = colorize(pred)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    color.save(out_path)
    print(f"wrote {out_path}")


def run_video(cfg, video_path, out_path, use_dummy, ckpt_path, vjepa_ckpt, ema_alpha=0.0):
    device = auto_device()
    impl = "dummy" if use_dummy else "vjepa"
    model_b = VJEPAFeatureBackbone(
        impl=impl,
        out_dim=cfg.model.get("backbone_out_dim", 1024),
        vjepa_ckpt=vjepa_ckpt,
    ).to(device)
    model_h = LinearFPNHead(
        in_dim=cfg.model.get("backbone_out_dim", 1024),
        fpn_dim=cfg.model.get("fpn_dim", 256),
        num_classes=int(cfg.data.get("num_classes", 6)),
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model_h.load_state_dict(state["model_h"])
    model_b.eval(); model_h.eval()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    prev_logits = None
    target_size = tuple(cfg.data.get("img_size", [512, 512]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = preprocess_image(img, target_size, flip=False, color_jitter=0.0)
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model_b(x)
            logits = model_h(feats, out_size=(x.shape[2], x.shape[3]))[0].cpu().numpy()
        if ema_alpha > 0 and prev_logits is not None:
            logits = ema_update(prev_logits, logits, ema_alpha)
        prev_logits = logits
        pred = logits.argmax(0).astype(np.uint8)
        color = colorize(pred)
        color_bgr = cv2.cvtColor(np.array(color), cv2.COLOR_RGB2BGR)
        color_bgr = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
        out.write(color_bgr)

    cap.release(); out.release()
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="ckpts/head_linear_fpn_best.pt")
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--vjepa_ckpt", type=str, default=None, help="Path to V-JEPA(2) checkpoint .pt/.pth")


    args = ap.parse_args()

    cfg = load_cfg(args.config)

    if args.image:
        run_image(cfg, args.image, args.out, bool(args.dummy_backbone), args.ckpt, args.vjepa_ckpt)
    elif args.video:
        run_video(cfg, args.video, args.out, bool(args.dummy_backbone), args.ckpt, args.vjepa_ckpt, args.ema)
    else:
        raise SystemExit("Provide --image or --video")


if __name__ == "__main__":
    main()
