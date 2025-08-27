
# ===========================
# scripts/demo_video.sh
# ===========================
#!/usr/bin/env bash
python -m vjepa_seg.inference --config configs/cihp_linear_fpn.yaml --video samples/clip.mp4 --out out.mp4 --ema 0.6 --dummy_backbone 1
