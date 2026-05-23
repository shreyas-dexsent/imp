import argparse
import base64
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def image_from_labelme(data, json_path):
    if data.get("imageData"):
        image_bytes = base64.b64decode(data["imageData"])
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image_path = Path(json_path).parent / data["imagePath"]
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image for {json_path}")
    return image


def normalize_points(points, width, height):
    normalized = []
    for x, y in points:
        normalized.append(x / width)
        normalized.append(y / height)
    return normalized


def convert_json(json_path, classes):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image = image_from_labelme(data, json_path)
    height, width = image.shape[:2]

    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"].strip()
        if label not in classes:
            classes[label] = len(classes)

        points = shape.get("points", [])
        if len(points) < 3:
            continue

        flat = normalize_points(points, width, height)
        line = f"{classes[label]} " + " ".join(f"{value:.6f}" for value in flat)
        lines.append(line)

    return image, lines, data.get("imagePath")


def main():
    parser = argparse.ArgumentParser(description="Convert Labelme polygons to YOLO segmentation dataset.")
    parser.add_argument("--input", required=True, help="Folder containing Labelme JSON files and images")
    parser.add_argument("--output", required=True, help="Output YOLO dataset folder")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    random.seed(args.seed)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No Labelme JSON files found in {input_dir}")

    classes = {}
    samples = []
    for json_file in json_files:
        image, yolo_lines, image_name = convert_json(json_file, classes)
        if not image_name:
            image_name = json_file.with_suffix(".png").name
        samples.append((json_file, image, yolo_lines, image_name))

    random.shuffle(samples)
    split_index = int(len(samples) * (1 - args.val_ratio))
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    for subset in ("train", "val"):
        (output_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    for subset_name, subset_samples in (("train", train_samples), ("val", val_samples)):
        for _, image, yolo_lines, image_name in subset_samples:
            image_out = output_dir / "images" / subset_name / image_name
            label_out = output_dir / "labels" / subset_name / f"{Path(image_name).stem}.txt"
            cv2.imwrite(str(image_out), image)
            with open(label_out, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

    # Copy unlabeled RGB images into train/val as empty-label negatives if they exist.
    labeled_images = {Path(sample[3]).name for sample in samples}
    unlabeled_images = sorted(input_dir.glob("rgb_*.png"))
    unlabeled_images = [img for img in unlabeled_images if img.name not in labeled_images]
    for idx, image_path in enumerate(unlabeled_images):
        subset_name = "val" if (idx % max(1, int(1 / max(args.val_ratio, 1e-6))) == 0) else "train"
        image_out = output_dir / "images" / subset_name / image_path.name
        label_out = output_dir / "labels" / subset_name / f"{image_path.stem}.txt"
        shutil.copy2(image_path, image_out)
        label_out.write_text("", encoding="utf-8")

    names = [None] * len(classes)
    for label, index in classes.items():
        names[index] = label.replace(" ", "_")

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {str((output_dir / 'images' / 'train').resolve())}\n")
        f.write(f"val: {str((output_dir / 'images' / 'val').resolve())}\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for idx, name in enumerate(names):
            f.write(f"  {idx}: {name}\n")

    print(f"[DONE] YOLO dataset written to: {output_dir}")
    print(f"[DONE] Classes: {names}")
    print(f"[DONE] Train samples: {len(train_samples)} labeled + {len([p for p in unlabeled_images if p])} unlabeled mixed in")
    print(f"[DONE] Val samples: {len(val_samples)}")


if __name__ == "__main__":
    main()
