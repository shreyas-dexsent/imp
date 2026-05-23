"""Dump the exact 8-channel RGBD tensor that the MegaPose coarse network sees
for every SO(3)-grid hypothesis.

The coarse forward pass (``forward_coarse`` in pose_rigid.py) does, per
hypothesis:

  1. Compute TCO_init from bbox + K + grid rotation.
  2. ``crop_inputs``: project the CAD's point cloud, take the smallest
     enclosing image bbox around the reprojection, auto-crop the real
     camera frame around that (same K_crop), resize to render_size (320 x 448).
  3. Render the CAD at TCO_init with K_crop into the same 320 x 448 canvas.
  4. ``normalize_images``: scale the depth channel by TCO[:,2,3] = Z.
  5. Concatenate → [B, 8, 320, 448] = [real_rgb(3), real_depth(1),
     rendered_rgb(3), rendered_depth(1)] → CNN → coarse_logit.

This tool reproduces exactly those five steps, then dumps:

  * ``real_crop/rgb.png``              the cropped+resized real RGB (one copy).
  * ``real_crop/depth_norm.png``       the normalized real depth channel
                                       (colour-mapped; one copy per hypothesis
                                       because K_crop and therefore the crop
                                       window depend on the hypothesis).
  * ``hyp_<id>/real_rgb.png``          cropped real RGB for this hypothesis.
  * ``hyp_<id>/real_depth.png``        normalized real depth for this hyp.
  * ``hyp_<id>/render_rgb.png``        rendered RGB.
  * ``hyp_<id>/render_depth.png``      normalized rendered depth.
  * ``hyp_<id>/side_by_side.png``      all four channels side-by-side for
                                       eyeballing; this is the single tile
                                       contained in ``grid_network_inputs.png``.
  * ``grid_network_inputs.png``        contact sheet over every hypothesis
                                       (``side_by_side`` per hyp).

Only the hypothesis IDs listed in ``--only-ids`` (default: rerank top-5 from
``--pose-json``) get per-hypothesis dumps by default. Pass ``--dump-all`` to
dump every hypothesis in the grid.

Example:

  python tools/dump_coarse_network_inputs.py \\
    --raw-rgb   /home/imp/imp/data/runs/run-.../cam_.../raw_rgb.png \\
    --raw-depth /home/imp/imp/data/runs/run-.../cam_.../scene_point_cloud.ply \\
    --pose-json /home/imp/imp/data/runs/run-.../cam_.../pose.json \\
    --mesh      /home/imp/imp/dexsent-vgr-vision-engine-2.5d/runtime_data/meshes/barel/barel.obj \\
    --label     barel \\
    --output    /tmp/coarse_inputs

If you do not have a ``raw_depth`` file handy, pass ``--no-depth`` and the
script will feed zeros; the RGB side is still faithful.
"""

from __future__ import annotations

import argparse
import json
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

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset  # noqa: E402
from megapose.lib3d.cosypose_ops import (  # noqa: E402
    TCO_init_from_boxes_autodepth_with_R,
)
from megapose.lib3d.camera_geometry import (  # noqa: E402
    boxes_from_uv,
    get_K_crop_resize,
    project_points_robust,
)
from megapose.lib3d.cropping import deepim_crops_robust  # noqa: E402
from megapose.lib3d.rigid_mesh_database import MeshDataBase  # noqa: E402
from megapose.panda3d_renderer.panda3d_batch_renderer import (  # noqa: E402
    Panda3dBatchRenderer,
)
from megapose.panda3d_renderer.types import Panda3dLightData  # noqa: E402
from megapose.utils import transform_utils  # noqa: E402


RENDER_SIZE = (320, 448)  # matches runtime default (H, W)
LAMB = 1.4                # matches forward_coarse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw-rgb", type=Path, required=True, help="PNG of the full camera frame at capture time")
    p.add_argument("--raw-depth", type=Path, default=None, help="Optional .npy/.png16 raw depth in metres (H x W). If omitted, depth is reconstructed from --scene-ply if provided, else filled with zeros.")
    p.add_argument("--scene-ply", type=Path, default=None, help="scene_point_cloud.ply / segmented_point_cloud.ply to reproject into a camera-frame depth map (matches what runtime fed in).")
    p.add_argument("--use-segmented-ply", action="store_true", help="If --scene-ply points at segmented_point_cloud.ply, also mask RGB by the resulting depth footprint (≈ what runtime's masked_rgb looks like).")
    p.add_argument("--pose-json", type=Path, required=True, help="pose.json for the run; provides bbox, K and rerank top-5 ids")
    p.add_argument("--mesh", type=Path, required=True, help="Runtime-prepared (recentered) barel.obj")
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--mesh-units", type=str, default="mm", choices=["mm", "m"])
    p.add_argument("--grid", type=int, default=576, choices=[72, 512, 576, 4608])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only-ids", type=int, nargs="*", default=None,
                   help="Hypothesis IDs to dump per-hyp. Default: rerank top-5 from pose.json.")
    p.add_argument("--dump-all", action="store_true", help="Dump every hypothesis in the grid (slow; produces many files).")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--tile-width", type=int, default=200)
    p.add_argument("--ncols", type=int, default=8)
    return p.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_mask_from_contour(contour_uv: list, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not contour_uv:
        return mask.astype(bool)
    pts = np.asarray(contour_uv, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask.astype(bool)


def depth_from_ply(ply_path: Path, K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    depth = np.zeros(shape, dtype=np.float32)
    try:
        with open(ply_path, "rb") as f:
            header_bytes = b""
            while b"end_header\n" not in header_bytes:
                header_bytes += f.read(4096)
                if not header_bytes:
                    break
            header = header_bytes.decode("utf-8", errors="ignore")
            n = 0
            fmt = "ascii"
            for line in header.splitlines():
                if line.startswith("element vertex"):
                    n = int(line.split()[-1])
                elif line.startswith("format"):
                    if "binary_little_endian" in line:
                        fmt = "bin_le"
                    elif "binary_big_endian" in line:
                        fmt = "bin_be"
            if n == 0:
                return depth
            if fmt == "ascii":
                body = b""
                body += f.read()
                lines = body.decode("utf-8").splitlines()
                xyz = np.asarray([[float(v) for v in ln.split()[:3]] for ln in lines[:n]], dtype=np.float32)
            else:
                dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]) if fmt == "bin_le" else np.dtype([("x", ">f4"), ("y", ">f4"), ("z", ">f4")])
                raw = np.frombuffer(f.read(n * 12), dtype=dt)
                xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
    except Exception as exc:
        print(f"depth_from_ply failed: {exc}")
        return depth

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = xyz[:, 2]
    good = np.isfinite(z) & (z > 0)
    xyz = xyz[good]
    u = (xyz[:, 0] * fx / xyz[:, 2] + cx).round().astype(np.int32)
    v = (xyz[:, 1] * fy / xyz[:, 2] + cy).round().astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], xyz[inside, 2]
    # take the nearest depth per pixel
    flat = v * w + u
    order = np.argsort(flat, kind="stable")
    flat, z = flat[order], z[order]
    prev = -1
    for idx in range(len(flat)):
        p = flat[idx]
        if p == prev:
            # keep minimum z for that pixel
            vv = p // w
            uu = p % w
            if z[idx] < depth[vv, uu]:
                depth[vv, uu] = z[idx]
            continue
        prev = p
        vv = p // w
        uu = p % w
        depth[vv, uu] = z[idx]
    return depth


def load_depth(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.float32)
    if path.suffix.lower() == ".npy":
        d = np.load(path).astype(np.float32)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        if raw.dtype == np.uint16:
            d = raw.astype(np.float32) / 1000.0  # mm → m
        else:
            d = raw.astype(np.float32)
    if d.shape[:2] != shape:
        d = cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return d


def build_renderer(mesh_path: Path, label: str, mesh_units: str):
    ds = RigidObjectDataset([RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units, scaling_factor=1.0)])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh_db = MeshDataBase.from_object_ds(ds).batched(n_sym=1).to(device)
    renderer = Panda3dBatchRenderer(
        object_dataset=ds,
        n_workers=0,
        preload_cache=False,
        split_objects=False,
        device=device,
    )
    return renderer, mesh_db, device


def visualize_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth) & (depth != 0.0)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not finite.any():
        return out
    lo, hi = float(depth[finite].min()), float(depth[finite].max())
    if hi - lo < 1e-8:
        hi = lo + 1e-6
    normed = np.where(finite, (depth - lo) / (hi - lo), 0)
    normed = np.clip(normed * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(normed, cv2.COLORMAP_TURBO)
    color[~finite] = 0
    return color


def compose_side_by_side(real_rgb, real_depth_vis, render_rgb, render_depth_vis, text: str) -> np.ndarray:
    panels = [real_rgb, real_depth_vis, render_rgb, render_depth_vis]
    panels = [cv2.cvtColor(p, cv2.COLOR_RGB2BGR) if p.ndim == 3 and p.shape[2] == 3 else p for p in panels]
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    panels = [cv2.resize(p, (w, h)) for p in panels]
    labels = ["real RGB", "real depth", "rendered RGB", "rendered depth"]
    labelled = []
    for p, lab in zip(panels, labels):
        strip = np.full((32, w, 3), 30, dtype=np.uint8)
        cv2.putText(strip, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
        labelled.append(np.concatenate([strip, p], axis=0))
    row = np.concatenate(labelled, axis=1)
    top = np.full((36, row.shape[1], 3), 15, dtype=np.uint8)
    cv2.putText(top, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
    return np.concatenate([top, row], axis=0)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "real_crop").mkdir(exist_ok=True)

    with open(args.pose_json) as f:
        pose_json = json.load(f)
    match = (pose_json.get("matches") or [{}])[0]
    bbox = np.asarray(match["bbox_xyxy"], dtype=np.float32)
    intr = match["camera_intrinsics"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    width = intr["resolution"]["width"]
    height = intr["resolution"]["height"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    rerank_ids: list[int] = []
    rerank = match.get("rerank") or {}
    for c in rerank.get("candidates") or []:
        try:
            rerank_ids.append(int(c["hypothesis_id"]))
        except Exception:
            pass
    only_ids = set(args.only_ids if args.only_ids is not None else rerank_ids)
    if args.dump_all:
        only_ids = None  # means "all"
    print(f"pose bbox={bbox.tolist()} rerank_top5={rerank_ids}")

    rgb = load_rgb(args.raw_rgb)
    if args.raw_depth is not None:
        depth = load_depth(args.raw_depth, (height, width))
    elif args.scene_ply is not None and args.scene_ply.is_file():
        depth = depth_from_ply(args.scene_ply, K, (height, width))
    else:
        depth = np.zeros((height, width), dtype=np.float32)
    assert rgb.shape[:2] == (height, width), f"rgb {rgb.shape} vs K res {(height, width)}"

    # Apply the same masking runtime does: RGB blacked outside the selection
    # mask and depth zeroed outside the depth_selection mask. We reconstruct
    # the selection mask from pose.json's segmentation_contour_uv.
    contour = match.get("segmentation_contour_uv") or match.get("segmentation_contours_uv")
    if isinstance(contour, list) and contour and isinstance(contour[0], list) and len(contour[0]) > 0 and isinstance(contour[0][0], list):
        contour = contour[0]  # unwrap list-of-contours
    if contour:
        selection_mask = build_mask_from_contour(contour, (height, width))
        rgb = rgb.copy()
        rgb[~selection_mask] = 0
        depth = depth.copy()
        depth[~selection_mask] = 0.0
        print(f"applied segmentation mask ({int(selection_mask.sum())} pixels)")
    else:
        print("WARNING: no segmentation_contour_uv in pose.json - feeding unmasked frame (NOT what runtime does)")

    rgb_t = torch.as_tensor(rgb).float().permute(2, 0, 1) / 255.0
    depth_t = torch.as_tensor(depth).float().unsqueeze(0)
    images = torch.cat([rgb_t, depth_t], dim=0).unsqueeze(0)  # [1, 4, H, W]
    K_t = torch.as_tensor(K).float().unsqueeze(0)             # [1, 3, 3]

    rotations = transform_utils.load_SO3_grid(args.grid)
    renderer, mesh_db, device = build_renderer(args.mesh, args.label, args.mesh_units)
    mesh = mesh_db.select([args.label])
    model_points_3d = mesh.points
    if model_points_3d.dim() == 2:
        model_points_3d = model_points_3d.unsqueeze(0)

    # TCO for every rotation, using full 2D bbox/K/mesh-points/R (see pose_estimator.forward_coarse_model)
    N = rotations.shape[0]
    boxes = torch.as_tensor(bbox, dtype=torch.float32).repeat(N, 1)
    K_rep = torch.as_tensor(K, dtype=torch.float32).repeat(N, 1, 1)
    pts_rep = model_points_3d.expand(N, -1, -1).cpu()
    TCO = TCO_init_from_boxes_autodepth_with_R(boxes, pts_rep, K_rep, rotations)  # [N, 4, 4] cpu

    TCO = TCO.to(device)
    K_dev = K_t.to(device)
    images_dev = images.to(device)

    # For each hyp we need the deepim crop of the observation at this TCO+K.
    # We do it in mini-batches but images has B=1 everywhere: expand on demand.
    mesh_points_sample_full = mesh.sample_points(2000, deterministic=True).to(device)  # [1, 2000, 3]

    grid_tiles: list[np.ndarray] = []
    saved_once_real_rgb = False

    def target_for_hyp(hid: int) -> bool:
        if only_ids is None:
            return True
        return hid in only_ids

    total = N
    for start in range(0, total, args.batch):
        stop = min(total, start + args.batch)
        b = stop - start
        TCO_b = TCO[start:stop]                                         # [b, 4, 4]
        tCR_b = TCO_b[:, :3, -1]                                         # [b, 3]
        images_b = images_dev.expand(b, -1, -1, -1)                      # [b, 4, H, W]
        K_b = K_dev.expand(b, -1, -1)                                    # [b, 3, 3]
        pts_b = mesh_points_sample_full.expand(b, -1, -1)                # [b, 2000, 3]

        uv = project_points_robust(pts_b, K_b, TCO_b)
        boxes_rend = boxes_from_uv(uv)
        boxes_crop, images_cropped = deepim_crops_robust(
            images=images_b,
            obs_boxes=boxes_rend,
            K=K_b,
            TCO_pred=TCO_b,
            tCR_in=tCR_b,
            O_vertices=pts_b,
            output_size=RENDER_SIZE,
            lamb=LAMB,
        )
        K_crop = get_K_crop_resize(
            K=K_b.clone(), boxes=boxes_crop, orig_size=images_b.shape[-2:], crop_resize=RENDER_SIZE
        ).detach()

        # Render the CAD at each TCO with the corresponding K_crop.
        light_datas = [[Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))]] * b
        render_out = renderer.render(
            labels=[args.label] * b,
            TCO=TCO_b,
            K=K_crop,
            light_datas=light_datas,
            resolution=RENDER_SIZE,
            render_depth=True,
            render_mask=False,
            render_normals=False,
        )
        rendered_rgbs = render_out.rgbs.detach().cpu().numpy()      # [b, 3, H, W] in [0,1]
        rendered_depths = render_out.depths.detach().cpu().numpy()  # [b, 1, H, W] in metres

        # Normalize depths by z = tCR[:, 2]  (matches normalize_depth)
        z_norm = tCR_b[:, 2].detach().cpu().numpy()  # [b]
        images_cropped_np = images_cropped.detach().cpu().numpy()   # [b, 4, H, W]
        for local_i in range(b):
            hid = start + local_i
            real_rgb_crop = np.clip(np.transpose(images_cropped_np[local_i, :3], (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            real_depth_crop = images_cropped_np[local_i, 3]
            real_depth_norm = real_depth_crop / max(1e-6, float(z_norm[local_i]))

            rendered_rgb = np.clip(np.transpose(rendered_rgbs[local_i], (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            rendered_depth = rendered_depths[local_i]
            if rendered_depth.ndim == 3 and rendered_depth.shape[0] == 1:
                rendered_depth = rendered_depth[0]
            rendered_depth_norm = rendered_depth / max(1e-6, float(z_norm[local_i]))

            real_depth_vis = visualize_depth(real_depth_norm)
            rendered_depth_vis = visualize_depth(rendered_depth_norm)

            if not saved_once_real_rgb:
                # "one copy" for the user — the real RGB crop is hypothesis-
                # dependent (K_crop changes), but their content only differs
                # by a projective offset; save hyp=0 as a reference.
                cv2.imwrite(str(args.output / "real_crop" / "rgb_hyp0.png"), cv2.cvtColor(real_rgb_crop, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(args.output / "real_crop" / "depth_hyp0.png"), real_depth_vis)
                saved_once_real_rgb = True

            if target_for_hyp(hid):
                hdir = args.output / f"hyp_{hid:04d}"
                hdir.mkdir(exist_ok=True)
                cv2.imwrite(str(hdir / "real_rgb.png"), cv2.cvtColor(real_rgb_crop, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(hdir / "real_depth.png"), real_depth_vis)
                cv2.imwrite(str(hdir / "render_rgb.png"), cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(hdir / "render_depth.png"), rendered_depth_vis)
                side = compose_side_by_side(real_rgb_crop, real_depth_vis, rendered_rgb, rendered_depth_vis, f"hyp {hid}")
                cv2.imwrite(str(hdir / "side_by_side.png"), side)
                thumb_w = args.tile_width
                thumb = cv2.resize(side, (thumb_w, int(round(side.shape[0] * thumb_w / side.shape[1]))))
                grid_tiles.append(thumb)

        print(f"processed {stop}/{total}")

    if grid_tiles:
        h_max = max(t.shape[0] for t in grid_tiles)
        padded = [np.pad(t, ((0, h_max - t.shape[0]), (0, 0), (0, 0))) for t in grid_tiles]
        n = len(padded)
        ncols = min(args.ncols, n)
        nrows = math.ceil(n / ncols)
        cell_h, cell_w = padded[0].shape[:2]
        sheet = np.zeros((nrows * cell_h, ncols * cell_w, 3), dtype=np.uint8)
        for i, t in enumerate(padded):
            r, c = divmod(i, ncols)
            sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = t
        out_sheet = args.output / "grid_network_inputs.png"
        cv2.imwrite(str(out_sheet), sheet)
        print(f"wrote {out_sheet}")

    print(f"\nDone. See {args.output}/")


if __name__ == "__main__":
    main()
