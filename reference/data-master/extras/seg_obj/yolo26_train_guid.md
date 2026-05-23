# YOLO26 Training Guide

This guide is for training a YOLO26 segmentation model on:

`/home/imp/imp/data/object_library/barel2/barel2_dataset`

It is written for a Linux machine with an NVIDIA Blackwell GPU.

## 1. Dataset

This dataset is already in YOLO segmentation format and looks valid:

- `train/images` and `train/labels`
- `valid/images` and `valid/labels`
- `test/images` and `test/labels`
- `data.yaml` at `/home/imp/imp/data/object_library/barel2/barel2_dataset/data.yaml`

Current summary:

- `train`: 60 images / 60 labels
- `valid`: 3 images / 3 labels
- `test`: 2 images / 2 labels
- classes: 1
- class id used in labels: `0`

## 2. Create Conda Env

```bash
conda create -y -n seg python=3.11
conda activate seg
python -m pip install --upgrade pip
```

## 3. Install PyTorch With CUDA

Use the current official PyTorch selector if you want the newest recommended command:

https://pytorch.org/get-started/locally/

At the time of writing on April 2, 2026, the stable install page showed PyTorch `2.7.0` with a CUDA `12.8` pip option, and PyTorch's 2.7 release notes explicitly mention Blackwell support.

Official references:

- https://pytorch.org/get-started/locally/
- https://pytorch.org/blog/pytorch-2-7/

Install with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 4. Install YOLO26 Stack

```bash
pip install "ultralytics>=8.4.33" labelme opencv-python-headless
```

If `opencv-python` gets installed as a dependency and later causes Qt issues for `labelme`, remove it and keep only headless OpenCV:

```bash
pip uninstall -y opencv-python
pip install opencv-python-headless
```

Ultralytics docs:

- https://docs.ultralytics.com/

## 5. Verify GPU

Run:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY
```

Expected result on the Blackwell machine:

- `cuda_available: True`
- `device_count: 1` or more
- device name should show your NVIDIA GPU

If CUDA is `False`, do not start training yet.

## 6. Verify Dataset Once

```bash
yolo checks
```

Optional quick sanity check:

```bash
python - <<'PY'
from pathlib import Path
base = Path("/home/imp/imp/data/object_library/barel2/barel2_dataset")
for split in ["train", "valid", "test"]:
    images = sorted((base / split / "images").glob("*"))
    labels = sorted((base / split / "labels").glob("*.txt"))
    print(split, "images=", len(images), "labels=", len(labels))
PY
```

## 7. Train YOLO26 Medium Segmentation

This is the main training command:

```bash
conda activate seg
yolo segment train \
  model=yolo26m-seg.pt \
  data=/home/imp/imp/data/object_library/barel2/barel2_dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  amp=True \
  cache=False \
  pretrained=True \
  project=/home/imp/imp/data/object_library/barel2/runs/segment \
  name=barel2_yolo26m
```

Notes:

- `model=yolo26m-seg.pt` means YOLO26 medium segmentation weights
- `device=0` means first CUDA GPU
- `batch=16` is a starting point, not a rule
- if you hit CUDA OOM, lower batch to `8`, then `4`

## 8. Match Your Roboflow Training Setup

From your Roboflow run:

- model type: `YOLO26 Instance Segmentation (Fast)`
- input size: `640x640`
- train/valid/test: `60 / 3 / 2`
- rotation: `-15` to `+15`
- exposure: `-15%` to `+15%`
- blur: up to `1px`

The closest local Ultralytics command is:

```bash
conda activate seg
yolo segment train \
  model=yolo26m-seg.pt \
  data=/home/imp/imp/data/object_library/barel2/barel2_dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  amp=True \
  degrees=15 \
  hsv_v=0.15 \
  blur=1.0 \
  scale=0.0 \
  translate=0.0 \
  shear=0.0 \
  perspective=0.0 \
  fliplr=0.0 \
  flipud=0.0 \
  mosaic=0.0 \
  mixup=0.0 \
  erasing=0.0 \
  project=/home/imp/imp/data/object_library/barel2/runs/segment \
  name=barel2_yolo26m_rf_like
```

Mapping notes:

- Roboflow `Resize: Stretch to 640x640` is already handled by `imgsz=640`
- Roboflow `Rotation -15 to +15` maps to `degrees=15`
- Roboflow `Exposure +/-15%` is approximated with `hsv_v=0.15`
- Roboflow `Blur up to 1px` maps to `blur=1.0`
- the other augmentation values are set to `0.0` to keep the run close to your Roboflow setup

If you want the closest possible reproduction, use this command instead of the more general augmented one later in this file.

## 9. Recommended Variants

If your Blackwell GPU has plenty of VRAM:

```bash
yolo segment train \
  model=yolo26l-seg.pt \
  data=/home/imp/imp/data/object_library/barel2/barel2_dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=8 \
  device=0 \
  workers=8 \
  amp=True \
  project=/home/imp/imp/data/object_library/barel2/runs/segment \
  name=barel2_yolo26l
```

If you want a faster first pass:

```bash
yolo segment train \
  model=yolo26s-seg.pt \
  data=/home/imp/imp/data/object_library/barel2/barel2_dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=24 \
  device=0 \
  workers=8 \
  amp=True \
  project=/home/imp/imp/data/object_library/barel2/runs/segment \
  name=barel2_yolo26s
```

## 10. Watch Metrics

Training outputs will go under:

`/home/imp/imp/data/object_library/barel2/runs/segment/barel2_yolo26m`

Key files:

- `results.png`
- `results.csv`
- `weights/best.pt`
- `weights/last.pt`

## 11. Validate Best Model

```bash
yolo segment val \
  model=/home/imp/imp/data/object_library/barel2/runs/segment/barel2_yolo26m/weights/best.pt \
  data=/home/imp/imp/data/object_library/barel2/barel2_dataset/data.yaml \
  device=0
```

## 12. Quick Prediction Test

```bash
yolo segment predict \
  model=/home/imp/imp/data/object_library/barel2/runs/segment/barel2_yolo26m/weights/best.pt \
  source=/home/imp/imp/data/object_library/barel2/barel2_dataset/test/images \
  device=0 \
  conf=0.25 \
  save=True
```

## 13. Copy Final Model To Object Folder

If the result looks good:

```bash
cp /home/imp/imp/data/object_library/barel2/runs/segment/barel2_yolo26m/weights/best.pt \
   /home/imp/imp/data/object_library/barel2/barel2.pt
```

If you want to keep both old and new models:

```bash
cp /home/imp/imp/data/object_library/barel2/runs/segment/barel2_yolo26m/weights/best.pt \
   /home/imp/imp/data/object_library/barel2/barel2_yolo26m.pt
```

## 14. Resume Training

```bash
yolo segment train resume \
  model=/home/imp/imp/data/object_library/barel2/runs/segment/barel2_yolo26m/weights/last.pt
```

## 15. If CUDA Is Not Detected

Check these in order:

1. `nvidia-smi`
2. `python -c "import torch; print(torch.cuda.is_available())"`
3. reinstall PyTorch from the official selector page
4. verify NVIDIA driver is installed and loaded on the target machine

## 16. Suggested First Run

For the first serious run on the Blackwell machine, use:

```bash
yolo segment train \
  model=yolo26m-seg.pt \
  data=/home/imp/imp/data/object_library/barel2/barel2_dataset/data.yaml \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  amp=True \
  degrees=5 \
  translate=0.05 \
  scale=0.10 \
  fliplr=0.5 \
  mosaic=0.2 \
  project=/home/imp/imp/data/object_library/barel2/runs/segment \
  name=barel2_yolo26m_aug
```

This is a reasonable starting point for your small industrial dataset.
