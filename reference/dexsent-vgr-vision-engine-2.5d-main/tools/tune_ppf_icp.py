"""Offline PPF + ICP tuning harness.

Loads a previously-saved PPF model cloud and scene cloud (from a debug
folder produced by a real pipeline run) and runs PPF + ICP repeatedly
with whatever params you sweep. Saves the resulting overlays as PLY files
so you can compare configurations side-by-side in MeshLab.

Usage
-----

    python tools/tune_ppf_icp.py \\
        --debug-dir /path/to/run/.../debug \\
        --out tune_out \\
        --sweep ppf_relative_sampling_step=0.02,0.025,0.03 \\
        --sweep icp_max_residual_m=0.008,0.012,0.02

Each sweep combination produces a sub-folder under ``--out`` containing:

    - model_pre_icp.ply      CAD at PPF's raw hypothesis pose
    - model_post_coarse.ply  CAD after coarse ICP
    - model_post_fine.ply    CAD after fine ICP
    - overlay_fine.ply       Model+scene combined for instant overlay
    - result.json            All hypothesis stats, residuals, accept flag

Tips
----

- Start from the debug PLYs already in your run folder so the tuning runs
  on the same data the pipeline saw.
- Don't sweep more than 2 params at a time -- the cross product explodes
  fast.  4x4 = 16 combos is already a lot of MeshLab tabs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vision_engine.modules.ppf_icp_bin_picking.runtime import (  # noqa: E402
    _lookup_ppf_api,
    _make_icp,
    _make_ppf_detector,
    _orthonormalize_pose,
    _parse_icp_result,
    _rotation_diagnostics,
    _transform_pc,
    _iter_ppf_poses,
)


def _read_ply_with_normals(path: Path) -> np.ndarray:
    """Read an ASCII PLY with x/y/z[/nx/ny/nz]. If the file lacks normals,
    raises -- tuning needs them."""
    text = path.read_text()
    lines = text.splitlines()
    end = lines.index("end_header")
    n = next(int(l.split()[2]) for l in lines if l.startswith("element vertex"))
    props = [l.split()[2] for l in lines[:end] if l.startswith("property ")]
    if not all(p in props for p in ("x", "y", "z")):
        raise ValueError(f"{path} missing x/y/z")
    has_normals = all(p in props for p in ("nx", "ny", "nz"))
    if not has_normals:
        raise ValueError(
            f"{path} missing normals -- regenerate with the new debug logger "
            f"that writes ppf_*_with_normals.ply files"
        )
    cols = []
    for raw in lines[end + 1 : end + 1 + n]:
        cols.append([float(v) for v in raw.split()])
    arr = np.asarray(cols, dtype=np.float32)
    ix = props.index("x")
    iy = props.index("y")
    iz = props.index("z")
    inx = props.index("nx")
    iny = props.index("ny")
    inz = props.index("nz")
    return np.stack(
        [
            arr[:, ix], arr[:, iy], arr[:, iz],
            arr[:, inx], arr[:, iny], arr[:, inz],
        ],
        axis=1,
    ).astype(np.float32)


def _save_xyz_ply(path: Path, points: np.ndarray) -> None:
    pts = np.asarray(points[:, :3], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def _run_one(
    *,
    model_pc: np.ndarray,
    scene_pc: np.ndarray,
    params: dict,
    out_dir: Path,
) -> dict:
    api = _lookup_ppf_api()
    if api is None:
        raise RuntimeError("OpenCV PPF not available in this Python env")

    t0 = time.perf_counter()
    detector = _make_ppf_detector(api, params)
    detector.trainModel(np.ascontiguousarray(model_pc, dtype=np.float32))
    train_dt = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw_poses = detector.match(
        np.ascontiguousarray(scene_pc, dtype=np.float32),
        float(params.get("ppf_scene_sample_step", 0.06)),
        float(params.get("ppf_scene_distance", 0.015)),
    )
    match_dt = time.perf_counter() - t1
    hypotheses = _iter_ppf_poses(
        raw_poses,
        limit=int(params.get("ppf_max_hypotheses", 30)),
    )
    if not hypotheses:
        return {"error": "no_hypotheses", "train_dt_s": train_dt, "match_dt_s": match_dt}

    hyp = hypotheses[0]
    raw_diag = _rotation_diagnostics(hyp.pose_matrix)
    pose = hyp.pose_matrix
    if not raw_diag.get("is_orthonormal", True):
        pose, _ = _orthonormalize_pose(pose)

    _save_xyz_ply(out_dir / "model_pre_icp.ply", _transform_pc(model_pc, pose))

    coarse_overrides = dict(params)
    coarse_overrides["icp_iterations"] = int(params.get("icp_coarse_iterations", 50))
    coarse_overrides["icp_rejection_scale"] = float(params.get("icp_coarse_rejection_scale", 3.0))
    coarse_overrides["icp_num_levels"] = int(params.get("icp_coarse_num_levels", 4))
    coarse_overrides["icp_tolerance"] = float(params.get("icp_coarse_tolerance", 0.02))

    icp_coarse = _make_icp(api, coarse_overrides)
    transformed = _transform_pc(model_pc, pose)
    res_coarse = icp_coarse.registerModelToScene(
        np.ascontiguousarray(transformed, dtype=np.float32),
        np.ascontiguousarray(scene_pc, dtype=np.float32),
    )
    coarse_residual, delta_coarse = _parse_icp_result(res_coarse)
    if delta_coarse is not None:
        pose = (delta_coarse @ pose).astype(np.float32)
    _save_xyz_ply(out_dir / "model_post_coarse.ply", _transform_pc(model_pc, pose))

    icp_fine = _make_icp(api, params)
    transformed = _transform_pc(model_pc, pose)
    res_fine = icp_fine.registerModelToScene(
        np.ascontiguousarray(transformed, dtype=np.float32),
        np.ascontiguousarray(scene_pc, dtype=np.float32),
    )
    fine_residual, delta_fine = _parse_icp_result(res_fine)
    if delta_fine is not None:
        pose = (delta_fine @ pose).astype(np.float32)
    _save_xyz_ply(out_dir / "model_post_fine.ply", _transform_pc(model_pc, pose))

    # Combined model+scene for instant overlay inspection.
    combined = np.vstack([
        _transform_pc(model_pc, pose)[:, :3],
        np.asarray(scene_pc[:, :3], dtype=np.float32),
    ])
    _save_xyz_ply(out_dir / "overlay_fine.ply", combined)

    accepted = (
        fine_residual is not None
        and fine_residual <= float(params.get("icp_max_residual_m", 0.012))
    )
    return {
        "params": {k: v for k, v in params.items() if not k.startswith("_")},
        "train_dt_s": float(train_dt),
        "match_dt_s": float(match_dt),
        "hypothesis_count": len(hypotheses),
        "votes": float(hyp.votes),
        "raw_implied_scale": raw_diag.get("implied_scale"),
        "raw_orthonormal_err": raw_diag.get("orthonormal_err"),
        "coarse_residual_m": coarse_residual,
        "fine_residual_m": fine_residual,
        "icp_accepted": bool(accepted),
        "final_pose": pose.tolist(),
    }


def _parse_sweep(spec: list[str]) -> dict:
    out = {}
    for s in spec or []:
        if "=" not in s:
            raise ValueError(f"sweep needs key=v1,v2,...; got {s!r}")
        k, v = s.split("=", 1)
        vals = []
        for raw in v.split(","):
            raw = raw.strip()
            if raw == "":
                continue
            try:
                vals.append(int(raw))
            except ValueError:
                try:
                    vals.append(float(raw))
                except ValueError:
                    if raw.lower() in ("true", "false"):
                        vals.append(raw.lower() == "true")
                    else:
                        vals.append(raw)
        out[k] = vals
    return out


DEFAULT_PARAMS = {
    "mesh_units": "m",
    "mesh_scale": 1.0,
    "ppf_model_sample_points": 1500,
    "ppf_relative_sampling_step": 0.025,
    "ppf_relative_distance_step": 0.025,
    "ppf_num_angles": 60,
    "ppf_search_position_threshold": 0.015,
    "ppf_search_rotation_threshold": 0.087,
    "ppf_weighted_clustering": True,
    "ppf_max_hypotheses": 30,
    "ppf_scene_sample_step": 0.05,
    "ppf_scene_distance": 0.015,
    "icp_iterations": 100,
    "icp_tolerance": 0.005,
    "icp_rejection_scale": 1.5,
    "icp_num_levels": 8,
    "icp_sample_type": "uniform",
    "icp_num_max_corr": 4,
    "icp_max_residual_m": 0.012,
    "icp_coarse_iterations": 50,
    "icp_coarse_rejection_scale": 3.0,
    "icp_coarse_num_levels": 4,
    "icp_coarse_tolerance": 0.02,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-dir", required=True, type=Path,
                    help="run debug folder (contains ppf_model_input_*_with_normals.ply)")
    ap.add_argument("--stage", default="primary",
                    help="which stage label to read (primary, all_poses_rank<N>, ...)")
    ap.add_argument("--out", default="tune_out", type=Path)
    ap.add_argument("--sweep", action="append", default=[],
                    help="param sweep, e.g. --sweep icp_max_residual_m=0.008,0.012")
    ap.add_argument("--set", action="append", default=[],
                    help="fixed param override, e.g. --set ppf_num_angles=48")
    args = ap.parse_args()

    model_path = args.debug_dir / f"ppf_model_input_{args.stage}_with_normals.ply"
    scene_path = args.debug_dir / f"ppf_scene_input_{args.stage}_with_normals.ply"
    if not model_path.exists() or not scene_path.exists():
        print(f"missing inputs: {model_path} or {scene_path}", file=sys.stderr)
        return 2

    model_pc = _read_ply_with_normals(model_path)
    scene_pc = _read_ply_with_normals(scene_path)
    print(f"model: {len(model_pc)} pts; scene: {len(scene_pc)} pts")

    base = dict(DEFAULT_PARAMS)
    for s in args.set:
        k, v = s.split("=", 1)
        try:
            base[k] = int(v)
        except ValueError:
            try:
                base[k] = float(v)
            except ValueError:
                base[k] = v.lower() == "true" if v.lower() in ("true", "false") else v

    sweep = _parse_sweep(args.sweep)
    if not sweep:
        sweep = {"_run": [0]}  # single run

    keys = list(sweep.keys())
    combos = list(itertools.product(*(sweep[k] for k in keys)))
    print(f"{len(combos)} combinations")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    summary = []
    for combo in combos:
        params = dict(base)
        label_parts = []
        for k, v in zip(keys, combo):
            if k != "_run":
                params[k] = v
                label_parts.append(f"{k}={v}")
        label = "__".join(label_parts) or "default"
        out_dir = args.out / label
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = _run_one(
                model_pc=model_pc,
                scene_pc=scene_pc,
                params=params,
                out_dir=out_dir,
            )
        except Exception as exc:
            result = {"error": f"{exc.__class__.__name__}: {exc}"}
        result["label"] = label
        result["out_dir"] = str(out_dir)
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        summary.append(result)
        fr = result.get("fine_residual_m")
        ok = result.get("icp_accepted")
        print(f"  {label}: fine_residual={fr} accepted={ok}")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {args.out}/summary.json")
    print(f"Open {args.out}/<combo>/overlay_fine.ply in MeshLab to inspect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
