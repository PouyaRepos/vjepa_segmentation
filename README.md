# vjepa_segmentation

Fast human-parsing (bg, hair, face, torso, arms, legs) using **frozen V-JEPA features** and a **light FPN head**.
Train supervised on CIHP (remapped to 6 classes).
Due to computation limit, it is 500(train)/100(val) images only.
Inference works on images and videos; optional EMA smoothing for minimal temporal flicker.


```bash
# install
python -m venv .venv && source .venv/bin/activate
pip install -e .


# train
python -m vjepa_seg.train --config configs/cihp_linear_fpn.yaml


# evaluate
python -m vjepa_seg.evaluate --config configs/cihp_linear_fpn.yaml


# inference (image)
python -m vjepa_seg.inference --image path.jpg --out out.png


# inference (video)
python -m vjepa_seg.inference --video in.mp4 --out out.mp4 --ema 0.6
```


> Note: `backbone_vjepa.py` exposes a clean interface for V-JEPA features. If you don't have V-JEPA weights handy, set `--dummy_backbone 1` to use a random frozen backbone for smoke tests.
