
# ===========================
# scripts/demo_image.sh
# ===========================
#!/usr/bin/env bash
python -m vjepa_seg.inference --config configs/cihp_linear_fpn.yaml --image samples/person.jpg --out out.png --dummy_backbone 1

