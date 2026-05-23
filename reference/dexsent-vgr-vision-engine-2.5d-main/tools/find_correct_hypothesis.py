"""Rank every SO(3)-grid hypothesis by how well its rendered silhouette
matches the real segmentation mask, to find the IDs the coarse network
SHOULD have picked.

For each of 576 hypotheses:
  * render the CAD at TCO_init (same as coarse model sees)
  * compute mask_iou with the real segmentation mask (from pose.json)
  * optionally compute depth MAE if --scene-ply is provided

Print the top-N ranked by mask_iou (or combined iou*depth_score).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_SRC = REPO_ROOT / "vision_engine/modules/megapose_bin_picking/vendor_runtime/third_party/src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))
os.environ.setdefault("MEGAPOSE_DATA_DIR", str(VENDOR_SRC / "megapose/data"))

import numpy as _np_compat
if not hasattr(_np_compat, "float_"):
    _np_compat.float_ = _np_compat.float64
if not hasattr(_np_compat, "int_"):
    _np_compat.int_ = _np_compat.int64

import torch

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.lib3d.cosypose_ops import TCO_init_from_boxes_autodepth_with_R
from megapose.lib3d.rigid_mesh_database import MeshDataBase
from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from megapose.panda3d_renderer.types import Panda3dLightData
from megapose.utils import transform_utils


def build_mask_from_contour(contour_uv, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.asarray(contour_uv, dtype=np.int32)
    if pts.ndim == 2 and pts.size >= 6:
        cv2.fillPoly(mask, [pts], 255)
    return mask.astype(bool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose-json", type=Path, required=True)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--mesh-units", type=str, default="mm")
    p.add_argument("--grid", type=int, default=576)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--top-n", type=int, default=15)
    args = p.parse_args()

    with open(args.pose_json) as f:
        pose_json = json.load(f)
    match = (pose_json.get("matches") or [{}])[0]
    bbox = np.asarray(match["bbox_xyxy"], dtype=np.float32)
    intr = match["camera_intrinsics"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    width = intr["resolution"]["width"]
    height = intr["resolution"]["height"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    contour = match.get("segmentation_contour_uv")
    if not contour:
        contours = match.get("segmentation_contours_uv") or []
        contour = contours[0] if contours else []
    real_mask = build_mask_from_contour(contour, (height, width))
    real_pixels = int(real_mask.sum())
    print(f"real mask: {real_pixels} pixels")

    rerank_ids = []
    rerank = match.get("rerank") or {}
    for c in rerank.get("candidates") or []:
        try:
            rerank_ids.append(int(c["hypothesis_id"]))
        except Exception:
            pass
    print(f"runtime rerank top-5 = {rerank_ids}")

    # Build renderer
    ds = RigidObjectDataset([RigidObject(label=args.label, mesh_path=args.mesh, mesh_units=args.mesh_units, scaling_factor=1.0)])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh_db = MeshDataBase.from_object_ds(ds).batched(n_sym=1).to(device)
    renderer = Panda3dBatchRenderer(
        object_dataset=ds, n_workers=0, preload_cache=False, split_objects=False, device=device,
    )

    rotations = transform_utils.load_SO3_grid(args.grid)
    N = rotations.shape[0]
    mesh_points = mesh_db.select([args.label]).points
    if mesh_points.dim() == 2:
        mesh_points = mesh_points.unsqueeze(0)

    boxes = torch.as_tensor(bbox, dtype=torch.float32).repeat(N, 1)
    K_rep = torch.as_tensor(K, dtype=torch.float32).repeat(N, 1, 1)
    pts_rep = mesh_points.expand(N, -1, -1).cpu()
    TCO = TCO_init_from_boxes_autodepth_with_R(boxes, pts_rep, K_rep, rotations)

    # Render each hypothesis at FULL scene resolution using original K
    ious = np.zeros(N, dtype=np.float32)
    for start in range(0, N, args.batch):
        stop = min(N, start + args.batch)
        b = stop - start
        TCO_b = TCO[start:stop]
        K_t = torch.as_tensor(K, dtype=torch.float32).repeat(b, 1, 1)
        light_datas = [[Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))]] * b
        out = renderer.render(
            labels=[args.label] * b,
            TCO=TCO_b,
            K=K_t,
            light_datas=light_datas,
            resolution=(height, width),
            render_depth=True,
            render_mask=False,
            render_normals=False,
        )
        depths = out.depths.detach().cpu().numpy()
        for local_i in range(b):
            d = depths[local_i]
            if d.ndim == 3 and d.shape[0] == 1:
                d = d[0]
            rendered_mask = (d > 0) & np.isfinite(d)
            inter = int(np.count_nonzero(rendered_mask & real_mask))
            union = int(np.count_nonzero(rendered_mask | real_mask))
            ious[start + local_i] = inter / union if union else 0.0
        print(f"{stop}/{N}")

    order = np.argsort(-ious)
    print(f"\nTop {args.top_n} hypotheses by mask_iou against real segmentation:")
    print(f"{'rank':>4} {'id':>5} {'iou':>7}  {'in_rerank_top5':>15}")
    print("-" * 40)
    for r in range(min(args.top_n, N)):
        hid = int(order[r])
        tag = " <-- yes" if hid in set(rerank_ids) else ""
        print(f"{r:>4} {hid:>5} {ious[hid]:>7.4f}{tag}")

    print(f"\nruntime rerank top-5 (iou):")
    for hid in rerank_ids:
        print(f"  id={hid:4d} iou={ious[hid]:.4f} (rank {int(np.where(order == hid)[0][0]) if hid < N else 'N/A'})")


if __name__ == "__main__":
    main()
