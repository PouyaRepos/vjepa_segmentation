
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
import torch.nn.functional as F

PALETTE = np.array([
    [0, 0, 0],        # 0 bg
    [203, 64, 255],   # 1 hair
    [255, 204, 153],  # 2 face
    [0, 153, 255],    # 3 torso
    [255, 153, 0],    # 4 arms
    [0, 255, 255],    # 5 legs
], dtype=np.uint8)

def preprocess_bgr(frame_bgr, target_hw):
    """frame_bgr: HxWx3 uint8, target_hw: (H, W)"""
    Ht, Wt = target_hw
    resized = cv2.resize(frame_bgr, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # 1x3xHxW
    # simple [-1,1] normalization (match your train if you used something else)
    x = x * 2.0 - 1.0
    return x

@torch.no_grad()
def predict_mask_from_bgr(frame_bgr, model_b, model_h, device, num_classes, ema_state=None, ema_alpha=0.0):
    """Returns (mask_uint8, new_ema_state)."""
    H0, W0 = frame_bgr.shape[:2]
    # preprocess to model input size from config
    inp_h, inp_w = model_h.input_size if hasattr(model_h, "input_size") else (256, 256)
    x = preprocess_bgr(frame_bgr, (inp_h, inp_w)).to(device)
    feats = model_b(x)                       # BxCxhxw
    # upsample logits to original resolution
    logits_up = model_h(feats, out_size=(H0, W0))  # BxKxH0xW0


    # optional EMA smoothing on probabilities
    if ema_alpha > 0.0:
        probs = torch.softmax(logits_up, dim=1)
        if ema_state is None:
            ema_state = probs.clone()
        else:
            ema_state = ema_alpha * ema_state + (1.0 - ema_alpha) * probs
        mask = ema_state.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return mask, ema_state
    else:
        mask = logits_up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return mask, None

def overlay_bgr(frame_bgr, mask_uint8):
    colors = PALETTE[mask_uint8]            # HxWx3 (RGB-like)
    # palette produced in RGB order; convert to BGR to blend with frame_bgr
    colors_bgr = colors[..., ::-1].copy()
    out = cv2.addWeighted(frame_bgr, 0.6, colors_bgr, 0.4, 0.0)
    return out


def ema_update(prev, cur, alpha):
    return alpha * cur + (1 - alpha) * prev


def run_image(cfg, image_path, out_path, use_dummy, ckpt, vjepa_ckpt):
    from vjepa_seg.utils import auto_device
    from vjepa_seg.backbone_vjepa import VJEPAFeatureBackbone
    from vjepa_seg.head_linear_fpn import LinearFPNHead

    device = auto_device()
    print("Using device:", device)

    num_classes = int(cfg.data.get("num_classes", 6))
    img_size    = cfg.data.get("img_size", [256, 256])
    fpn_dim     = int(cfg.model.get("fpn_dim", 128))
    out_dim     = int(cfg.model.get("backbone_out_dim", 1024))

    # normalize img_size to (H, W)
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        inp_h, inp_w = int(img_size[0]), int(img_size[1])
    else:
        inp_h = inp_w = int(img_size)


    impl = "dummy" if use_dummy else "vjepa"
    model_b = VJEPAFeatureBackbone(impl=impl, out_dim=out_dim, vjepa_ckpt=vjepa_ckpt).to(device)
    model_h = LinearFPNHead(in_dim=out_dim, fpn_dim=fpn_dim, num_classes=num_classes).to(device)
    model_h.input_size = (inp_h, inp_w)

    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model_h.load_state_dict(state, strict=False)
        print(f"[load] head weights: {ckpt}")
    else:
        print(f"[warn] no head checkpoint at {ckpt} — results may be blank.")

    model_b.eval(); model_h.eval()

    frame_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    mask, _ = predict_mask_from_bgr(frame_bgr, model_b, model_h, device, num_classes)
    print("labels present:", np.unique(mask).tolist())

    overlay = overlay_bgr(frame_bgr, mask)
    cv2.imwrite(out_path, overlay)
    print(f"wrote {out_path}")


def run_video(cfg, video_path, out_path, use_dummy, ckpt, vjepa_ckpt, ema_alpha=0.0):
    from vjepa_seg.utils import auto_device
    from vjepa_seg.backbone_vjepa import VJEPAFeatureBackbone
    from vjepa_seg.head_linear_fpn import LinearFPNHead

    device = auto_device()
    print("Using device:", device)

    num_classes = int(cfg.data.get("num_classes", 6))
    img_size    = cfg.data.get("img_size", [256, 256])
    fpn_dim     = int(cfg.model.get("fpn_dim", 128))
    out_dim     = int(cfg.model.get("backbone_out_dim", 1024))

    # normalize img_size to (H, W)
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        inp_h, inp_w = int(img_size[0]), int(img_size[1])
    else:
        inp_h = inp_w = int(img_size)

    impl = "dummy" if use_dummy else "vjepa"
    model_b = VJEPAFeatureBackbone(impl=impl, out_dim=out_dim, vjepa_ckpt=vjepa_ckpt).to(device)
    model_h = LinearFPNHead(in_dim=out_dim, fpn_dim=fpn_dim, num_classes=num_classes).to(device)
    model_h.input_size = (inp_h, inp_w)

    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model_h.load_state_dict(state, strict=False)
        print(f"[load] head weights: {ckpt}")
    else:
        print(f"[warn] no head checkpoint at {ckpt} — results may be blank.")

    model_b.eval(); model_h.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    ema_state = None
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        mask, ema_state = predict_mask_from_bgr(frame_bgr, model_b, model_h, device, num_classes, ema_state, ema_alpha)
        if idx < 5:
            print(f"[frame {idx}] labels:", np.unique(mask).tolist())
        overlay = overlay_bgr(frame_bgr, mask)
        writer.write(overlay.astype(np.uint8))
        idx += 1

    cap.release()
    writer.release()
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
    ap.add_argument("--dummy_backbone", type=int, default=0,
                help="1 = use dummy conv backbone, 0 = use V-JEPA2 (default)")



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
