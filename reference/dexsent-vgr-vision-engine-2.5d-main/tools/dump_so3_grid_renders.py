"""Render every SO(3)-grid orientation of a CAD object using the exact same
Panda3D renderer + material/shader that the MegaPose coarse model sees at
runtime.

For each hypothesis_id in the grid, we compute TCO with
``TCO_init_from_boxes_autodepth_with_R`` using a real bbox + K, render the
full scene at camera resolution, annotate the tile with id, and write:

  * ``individual/hyp_<id>.png`` — one image per hypothesis
  * ``grid_all.png`` — a single contact-sheet of every hypothesis
  * ``grid_coarse_top_<n>.png`` — a contact-sheet of the top-N by
    coarse_logit, if ``--coarse-logit-json`` is provided

Usage:

  python tools/dump_so3_grid_renders.py \\
    --mesh /home/imp/imp/data/stations/station-1/assets/asset-1/objects/barel2/barel2.obj \\
    --label barel \\
    --bbox-xyxy 470 483 654 678 \\
    --fx 665.4837 --fy 666.8171 --cx 634.1397 --cy 356.5756 \\
    --width 1280 --height 720 \\
    --grid 576 \\
    --output /tmp/grid_barel2

The bbox and K above come from
``data/runs/run-20260422_123943/vision/.../pose.json``; use whatever run you
are debugging.
"""

from __future__ import annotations

import argparse
import math
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

import numpy as _np_compat  # noqa: E402
if not hasattr(_np_compat, "float_"):
    _np_compat.float_ = _np_compat.float64
if not hasattr(_np_compat, "int_"):
    _np_compat.int_ = _np_compat.int64

import torch  # noqa: E402
import panda3d.core as p3d  # noqa: E402

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset  # noqa: E402
from megapose.lib3d.cosypose_ops import TCO_init_from_boxes_autodepth_with_R  # noqa: E402
from megapose.lib3d.rigid_mesh_database import MeshDataBase  # noqa: E402
from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer  # noqa: E402
from megapose.panda3d_renderer.types import Panda3dLightData  # noqa: E402
from megapose.utils import transform_utils  # noqa: E402


def load_so3_grid(grid_size: int) -> torch.Tensor:
    return transform_utils.load_SO3_grid(grid_size)


def build_renderer(mesh_path: Path, label: str, mesh_units: str):
    object_dataset = RigidObjectDataset([
        RigidObject(
            label=label,
            mesh_path=mesh_path,
            mesh_units=mesh_units,
            scaling_factor=1.0,
        ),
    ])
    mesh_db = MeshDataBase.from_object_ds(object_dataset)
    mesh_db_batched = mesh_db.batched(n_sym=1).to("cuda" if torch.cuda.is_available() else "cpu")
    renderer = Panda3dBatchRenderer(
        object_dataset=object_dataset,
        n_workers=0,
        preload_cache=False,
        split_objects=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return renderer, mesh_db_batched, object_dataset


def compute_TCO(
    rotations: torch.Tensor,
    bbox_xyxy: np.ndarray,
    K: np.ndarray,
    mesh_points_3d: torch.Tensor,
) -> torch.Tensor:
    n = rotations.shape[0]
    boxes = torch.as_tensor(bbox_xyxy, dtype=torch.float32).repeat(n, 1)
    K_t = torch.as_tensor(K, dtype=torch.float32).repeat(n, 1, 1)
    pts = mesh_points_3d.repeat(n, 1, 1)
    return TCO_init_from_boxes_autodepth_with_R(boxes, pts, K_t, rotations)


def render_batch(renderer, label: str, TCO: torch.Tensor, K: np.ndarray, resolution: tuple[int, int]):
    n = TCO.shape[0]
    K_t = torch.as_tensor(K, dtype=torch.float32).repeat(n, 1, 1)
    light_datas = [[Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))]] * n
    return renderer.render(
        labels=[label] * n,
        TCO=TCO,
        K=K_t,
        light_datas=light_datas,
        resolution=resolution,
        render_depth=False,
        render_mask=False,
        render_normals=False,
    )


def annotate_tile(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 200, 255), 2)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return out


def make_contact_sheet(tiles: list[np.ndarray], ncols: int) -> np.ndarray:
    if not tiles:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    h, w = tiles[0].shape[:2]
    nrows = math.ceil(len(tiles) / ncols)
    sheet = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, ncols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    return sheet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--label", type=str, default="object")
    p.add_argument("--mesh-units", type=str, default="mm", choices=["mm", "m"])
    p.add_argument("--bbox-xyxy", type=float, nargs=4, required=True, metavar=("xmin", "ymin", "xmax", "ymax"))
    p.add_argument("--fx", type=float, required=True)
    p.add_argument("--fy", type=float, required=True)
    p.add_argument("--cx", type=float, required=True)
    p.add_argument("--cy", type=float, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--grid", type=int, default=576, choices=[72, 512, 576, 4608])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--tile-width", type=int, default=240, help="resize each rendered scene to this width for the contact sheet")
    p.add_argument("--ncols", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--coarse-logit-json", type=Path, default=None,
                   help="Optional pose.json path; used to draw a second contact-sheet containing only the network's top-N hypotheses")
    p.add_argument("--top-n", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "individual").mkdir(exist_ok=True)

    K = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    bbox = np.asarray(args.bbox_xyxy, dtype=np.float32)
    resolution = (int(args.height), int(args.width))

    rotations = load_so3_grid(args.grid)
    print(f"SO(3) grid size = {rotations.shape[0]}")

    renderer, mesh_db_batched, _ = build_renderer(args.mesh, args.label, args.mesh_units)

    mesh_pts = mesh_db_batched.select([args.label]).points
    if mesh_pts.dim() == 2:
        mesh_pts = mesh_pts.unsqueeze(0)

    TCO_all = compute_TCO(rotations, bbox, K, mesh_pts.cpu())

    # Crop each render to a square region around the bbox so the contact
    # sheet actually shows the object and its orientation.
    bx0, by0, bx1, by1 = [int(round(v)) for v in bbox.tolist()]
    bw, bh = bx1 - bx0, by1 - by0
    pad = int(round(max(bw, bh) * 0.25))
    side = max(bw, bh) + 2 * pad
    cx = (bx0 + bx1) // 2
    cy = (by0 + by1) // 2
    crop_x0 = max(0, cx - side // 2)
    crop_y0 = max(0, cy - side // 2)
    crop_x1 = min(args.width, crop_x0 + side)
    crop_y1 = min(args.height, crop_y0 + side)

    all_tiles: list[tuple[int, np.ndarray]] = []
    total = TCO_all.shape[0]
    for start in range(0, total, args.batch):
        stop = min(total, start + args.batch)
        out = render_batch(renderer, args.label, TCO_all[start:stop], K, resolution)
        rgbs = out.rgbs.detach().cpu().numpy()
        for local_i in range(stop - start):
            hyp_id = start + local_i
            img = rgbs[local_i]
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(args.output / "individual" / f"hyp_{hyp_id:04d}.png"), bgr)
            cropped = bgr[crop_y0:crop_y1, crop_x0:crop_x1]
            scale = args.tile_width / cropped.shape[1]
            tile = cv2.resize(cropped, (args.tile_width, int(round(cropped.shape[0] * scale))))
            tile = annotate_tile(tile, f"id={hyp_id}")
            all_tiles.append((hyp_id, tile))
        print(f"rendered {stop}/{total}")

    sheet_all = make_contact_sheet([t for _, t in all_tiles], ncols=args.ncols)
    cv2.imwrite(str(args.output / "grid_all.png"), sheet_all)
    print(f"wrote {args.output / 'grid_all.png'}")

    if args.coarse_logit_json is not None and args.coarse_logit_json.is_file():
        import json
        with open(args.coarse_logit_json) as f:
            pose_json = json.load(f)
        rerank = (pose_json.get("matches") or [{}])[0].get("rerank") or {}
        candidates = rerank.get("candidates") or []
        ids_in_topk = [int(c.get("hypothesis_id", -1)) for c in candidates if int(c.get("hypothesis_id", -1)) >= 0]
        picked_tiles = [tile for hid, tile in all_tiles if hid in set(ids_in_topk)]
        if picked_tiles:
            sheet_top = make_contact_sheet(picked_tiles, ncols=min(len(picked_tiles), 5))
            cv2.imwrite(str(args.output / f"grid_rerank_top{len(picked_tiles)}.png"), sheet_top)
            print(f"wrote grid_rerank_top{len(picked_tiles)}.png (ids={ids_in_topk})")

    print(f"\nDone. See {args.output}/")


if __name__ == "__main__":
    main()
