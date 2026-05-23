# Segmentation Training Guide on Ubuntu

This is the updated Linux workflow for training a segmentation model for:

```text
/home/imp/imp/data/object_library/barel2
```

The capture step uses the same RGB exposure settings from:

```text
/home/imp/imp/dexsent-vgr-camera-core-2.5d/config/cam_realsense_d435i.yaml
```

Right now that means the capture script will pick up the D435i color settings from the YAML, including:

- `enable_auto_exposure: false`
- `exposure: 400`
- `gain: 40`
- `enable_auto_white_balance: true`

If you change those values in the camera YAML, the capture script will use the updated values next time.

## 1. Set paths

```bash
export OBJECT_DIR=/home/imp/imp/data/object_library/barel2
export DATASET_DIR=$OBJECT_DIR/dataset
export LABELME_DIR=$DATASET_DIR/rgb
export YOLO_DIR=$OBJECT_DIR/yolo_dataset
export CAM_CONFIG=/home/imp/imp/dexsent-vgr-camera-core-2.5d/config/cam_realsense_d435i.yaml
export RUN_NAME=barel2_seg
```

## 2. Capture a new dataset from RealSense

Use the camera env for capture:

```bash
conda activate vgr-camera
cd /home/imp/imp
python seg_obj/capture_from_realsense.py \
  --output "$DATASET_DIR" \
  --camera-config "$CAM_CONFIG"
```

While the preview is open:

- press `s` to save each frame
- press `q` to quit

Optional: if you still want the older manual depth tuning during capture, add:

```bash
--no-depth-auto-exposure --depth-exposure 25000 --depth-gain 32
```

## 3. Copy RGB images into the Labelme folder

```bash
mkdir -p "$LABELME_DIR"
cp "$DATASET_DIR"/rgb_*.png "$LABELME_DIR"/
```

## 4. Label images in Labelme

Use the vision env for labeling, conversion, and training:

```bash
conda activate vgr-vision
```

If Labelme fails with an `xcb` plugin error on Ubuntu, install the missing XCB runtime once:

```bash
sudo apt-get update
sudo apt-get install -y libxcb-xinerama0
```

If `labelme` is not installed yet:

```bash
pip install labelme
```

Then open the folder:

```bash
cd /home/imp/imp
labelme "$LABELME_DIR"
```

If the command is not found:

```bash
cd /home/imp/imp
python -m labelme "$LABELME_DIR"
```

Use one class name only:

```text
barel2
```

## 5. Convert Labelme polygons to YOLO segmentation format

```bash
cd /home/imp/imp
python seg_obj/convert_labelme_to_yolo.py \
  --input "$LABELME_DIR" \
  --output "$YOLO_DIR" \
  --val-ratio 0.2 \
  --seed 42
```

This creates:

```text
$YOLO_DIR/data.yaml
```

## 6. Train the segmentation model

CPU-safe command:

```bash
cd /home/imp/imp
yolo segment train \
  model=yolov8n-seg.pt \
  data="$YOLO_DIR/data.yaml" \
  epochs=100 \
  imgsz=640 \
  batch=8 \
  device=cpu \
  workers=8 \
  project="$OBJECT_DIR/runs/segment" \
  name="$RUN_NAME" \
  hsv_h=0 \
  hsv_s=0 \
  hsv_v=0.2 \
  degrees=30 \
  translate=0 \
  scale=0 \
  shear=0 \
  perspective=0 \
  fliplr=0 \
  flipud=0 \
  mosaic=0 \
  mixup=0 \
  cutmix=0 \
  copy_paste=0
```

If you hit worker or memory issues, use:

```bash
cd /home/imp/imp
yolo segment train \
  model=yolov8n-seg.pt \
  data="$YOLO_DIR/data.yaml" \
  epochs=100 \
  imgsz=640 \
  batch=4 \
  device=cpu \
  workers=0 \
  project="$OBJECT_DIR/runs/segment" \
  name="$RUN_NAME" \
  hsv_h=0 \
  hsv_s=0 \
  hsv_v=0.2 \
  degrees=30 \
  translate=0 \
  scale=0 \
  shear=0 \
  perspective=0 \
  fliplr=0 \
  flipud=0 \
  mosaic=0 \
  mixup=0 \
  cutmix=0 \
  copy_paste=0
```

If your `vgr-vision` env has CUDA ready, you can switch `device=cpu` to `device=0`.

## 7. Copy the trained model into the object folder

After training finishes:

```bash
cp "$OBJECT_DIR/runs/segment/$RUN_NAME/weights/best.pt" "$OBJECT_DIR/barel2.pt"
```

That overwrites the current segmentation model for `barel2`.

## 8. Final trained model path

Training run output:

```text
/home/imp/imp/data/object_library/barel2/runs/segment/barel2_seg/weights/best.pt
```

Model used by the vision pipeline:

```text
/home/imp/imp/data/object_library/barel2/barel2.pt
```

## Quick barel2 flow

```bash
export OBJECT_DIR=/home/imp/imp/data/object_library/barel2
export DATASET_DIR=$OBJECT_DIR/dataset
export LABELME_DIR=$DATASET_DIR/rgb
export YOLO_DIR=$OBJECT_DIR/yolo_dataset
export CAM_CONFIG=/home/imp/imp/dexsent-vgr-camera-core-2.5d/config/cam_realsense_d435i.yaml
export RUN_NAME=barel2_seg

conda activate vgr-camera
cd /home/imp/imp
python seg_obj/capture_from_realsense.py --output "$DATASET_DIR" --camera-config "$CAM_CONFIG"

mkdir -p "$LABELME_DIR"
cp "$DATASET_DIR"/rgb_*.png "$LABELME_DIR"/

conda activate vgr-vision
labelme "$LABELME_DIR"

cd /home/imp/imp
python seg_obj/convert_labelme_to_yolo.py --input "$LABELME_DIR" --output "$YOLO_DIR" --val-ratio 0.2 --seed 42
yolo segment train model=yolov8n-seg.pt data="$YOLO_DIR/data.yaml" epochs=100 imgsz=640 batch=8 device=cpu workers=8 project="$OBJECT_DIR/runs/segment" name="$RUN_NAME" hsv_h=0 hsv_s=0 hsv_v=0.2 degrees=30 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0 mosaic=0 mixup=0 cutmix=0 copy_paste=0
cp "$OBJECT_DIR/runs/segment/$RUN_NAME/weights/best.pt" "$OBJECT_DIR/barel2.pt"
```
