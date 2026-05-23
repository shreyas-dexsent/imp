from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from vision_engine.modules.megapose_bin_picking.runtime import (
    WORKSPACE_ROOT,
    _BinPickingTimingLog,
    _build_output_dir as _build_megapose_output_dir,
    _clean_path_string,
    _coerce_float,
    _detection_passes_bin_roi,
    _json_safe,
    _normalize_axis,
    _project_point_uv,
    _runtime_log as _megapose_runtime_log,
    _safe_slug,
    _save_bgr_image,
    _save_json_file,
    _save_pose_annotated_overlay,
    _save_segmentation_candidates_overlay,
    build_camera_data,
    depth_to_point_cloud_points,
    extract_segmentation_contours,
    filter_depth_range,
    get_mesh_annotation_meta,
    mask_observation_to_selected_object,
    normalize_depth_map,
    resize_inputs_for_processing,
    resolve_object_assets,
    resolve_segmentation_backend,
    resolve_workspace_path,
    rotation_matrix_to_quaternion_xyzw,
    run_segmentation,
    save_multi_pose_scene_glb,
    save_pose_3d_assets,
    select_detection_candidate,
    suppress_nested_duplicate_detections,
    voxel_downsample_points,
)


DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "data" / "vision" / "ppf_icp"
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[Any, ...], "PpfModelBundle"] = {}


def _runtime_log(request_id: str | None, message: str) -> None:
    prefix = f"[ppf_icp_bin_picking:{request_id}]" if request_id else "[ppf_icp_bin_picking]"
    print(f"{prefix} {message}", flush=True)


@dataclass
class PpfApi:
    detector_ctor: Any
    icp_ctor: Any
    compute_normals: Any


@dataclass
class PpfModelBundle:
    detector: Any
    model_pc: np.ndarray
    mesh_path: Path
    mesh_units: str
    mesh_scale: float
    cache_key: tuple[Any, ...]
    lock: threading.Lock


@dataclass
class PpfPoseHypothesis:
    index: int
    pose_matrix: np.ndarray
    votes: float
    residual: float | None
    angle: float | None
    icp_applied: bool = False
    icp_residual: float | None = None
    icp_accepted: bool = True


class _PpfDebugLogger:
    """Persists PPF + ICP intermediate artifacts to <output_dir>/debug/.

    Helps diagnose pose-estimation issues like scale mismatches, where the
    rendered CAD looks too small or the pose is unstable across frames.
    """

    def __init__(self, output_dir: Path | None, request_id: str | None) -> None:
        self.request_id = request_id
        self.records: list[dict[str, Any]] = []
        if output_dir is None:
            self.dir = None
            self._jsonl_path = None
            return
        self.dir = output_dir / "debug"
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.dir = None
            self._jsonl_path = None
            return
        self._jsonl_path = self.dir / "pipeline.jsonl"

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def event(self, stage: str, **fields: Any) -> None:
        record = {"stage": str(stage), "ts_ns": time.time_ns(), **fields}
        self.records.append(record)
        if not self.enabled or self._jsonl_path is None:
            return
        try:
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_safe_json(record), separators=(",", ":")) + "\n")
        except Exception as exc:
            _runtime_log(self.request_id, f"debug_log_write_failed stage={stage} error={exc}")

    def save_point_cloud(self, name: str, pc: np.ndarray | None) -> str | None:
        if not self.enabled or pc is None or len(pc) == 0:
            return None
        try:
            import trimesh

            pts = np.asarray(pc[:, :3], dtype=np.float32)
            colors = np.tile(np.array([180, 180, 180, 255], dtype=np.uint8), (len(pts), 1))
            cloud = trimesh.points.PointCloud(vertices=pts, colors=colors)
            destination = self.dir / f"{name}.ply"
            cloud.export(str(destination))
            return str(destination)
        except Exception as exc:
            _runtime_log(self.request_id, f"debug_pc_save_failed name={name} error={exc}")
            return None

    def save_json(self, name: str, payload: Any) -> str | None:
        if not self.enabled:
            return None
        try:
            destination = self.dir / f"{name}.json"
            destination.write_text(
                json.dumps(_safe_json(payload), indent=2),
                encoding="utf-8",
            )
            return str(destination)
        except Exception as exc:
            _runtime_log(self.request_id, f"debug_json_save_failed name={name} error={exc}")
            return None

    def save_point_cloud_with_normals(
        self, name: str, pc: np.ndarray | None
    ) -> str | None:
        """Save an ASCII PLY with x/y/z + nx/ny/nz so MeshLab can render
        normal vectors as little arrows. Also handy for confirming that
        normals are unit-length (corrupt normals are the #1 cause of
        OpenCV PPF returning a non-orthonormal rotation)."""
        if not self.enabled or pc is None or len(pc) == 0:
            return None
        try:
            arr = np.asarray(pc, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 6:
                return self.save_point_cloud(name, pc)
            destination = self.dir / f"{name}_with_normals.ply"
            n = len(arr)
            with destination.open("w", encoding="ascii") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {n}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property float nx\nproperty float ny\nproperty float nz\n")
                f.write("end_header\n")
                for row in arr[:, :6]:
                    f.write(
                        f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} "
                        f"{row[3]:.6f} {row[4]:.6f} {row[5]:.6f}\n"
                    )
            return str(destination)
        except Exception as exc:
            _runtime_log(self.request_id, f"debug_pc_normals_save_failed name={name} error={exc}")
            return None

    def save_features_overlay(
        self,
        name: str,
        points: np.ndarray,
        *,
        feature_pair_indices: np.ndarray | None = None,
    ) -> str | None:
        """Save subsampled feature points as a coloured PLY plus, when
        feature_pair_indices is given, an additional PLY containing line
        segments connecting each pair (visible in MeshLab as wireframe
        edges). This lets the user inspect WHICH points PPF is hashing."""
        if not self.enabled or points is None or len(points) == 0:
            return None
        try:
            import trimesh

            pts = np.asarray(points[:, :3], dtype=np.float32)
            colors = np.tile(np.array([255, 80, 80, 255], dtype=np.uint8), (len(pts), 1))
            cloud = trimesh.points.PointCloud(vertices=pts, colors=colors)
            destination = self.dir / f"{name}_feature_points.ply"
            cloud.export(str(destination))
            if feature_pair_indices is not None and len(feature_pair_indices) > 0:
                pair_idx = np.asarray(feature_pair_indices, dtype=np.int64)
                pair_dest = self.dir / f"{name}_feature_pairs.ply"
                # Custom ASCII PLY with edges so MeshLab/CloudCompare draws
                # line segments connecting paired points.
                with pair_dest.open("w", encoding="ascii") as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {len(pts)}\n")
                    f.write("property float x\nproperty float y\nproperty float z\n")
                    f.write(f"element edge {len(pair_idx)}\n")
                    f.write("property int vertex1\nproperty int vertex2\n")
                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                    f.write("end_header\n")
                    for p in pts:
                        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                    rng = np.random.default_rng(42)
                    cmap = rng.integers(50, 255, size=(len(pair_idx), 3))
                    for (i, j), c in zip(pair_idx, cmap):
                        f.write(f"{int(i)} {int(j)} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            return str(destination)
        except Exception as exc:
            _runtime_log(self.request_id, f"debug_features_save_failed name={name} error={exc}")
            return None

    def save_image(self, name: str, image: np.ndarray | None) -> str | None:
        if not self.enabled or image is None:
            return None
        try:
            destination = self.dir / f"{name}.png"
            cv2.imwrite(str(destination), image)
            return str(destination)
        except Exception as exc:
            _runtime_log(self.request_id, f"debug_image_save_failed name={name} error={exc}")
            return None


def _safe_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _point_cloud_stats(points: np.ndarray | None) -> dict[str, Any]:
    if points is None or len(points) == 0:
        return {"count": 0}
    pts = np.asarray(points[:, :3], dtype=np.float64)
    mins = pts.min(axis=0).tolist()
    maxs = pts.max(axis=0).tolist()
    extents = (pts.max(axis=0) - pts.min(axis=0)).tolist()
    centroid = pts.mean(axis=0).tolist()
    diameter = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return {
        "count": int(len(pts)),
        "min_xyz": mins,
        "max_xyz": maxs,
        "extents": extents,
        "centroid": centroid,
        "diameter": diameter,
    }


def _rotation_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    R = np.asarray(matrix[:3, :3], dtype=np.float64)
    col_norms = np.linalg.norm(R, axis=0).tolist()
    row_norms = np.linalg.norm(R, axis=1).tolist()
    det = float(np.linalg.det(R))
    rtr = R.T @ R
    orthonormal_err = float(np.linalg.norm(rtr - np.eye(3)))
    avg_scale = float(np.mean(np.linalg.norm(R, axis=0)))
    return {
        "col_norms": col_norms,
        "row_norms": row_norms,
        "det": det,
        "orthonormal_err": orthonormal_err,
        "implied_scale": avg_scale,
        "is_orthonormal": orthonormal_err < 1e-3 and abs(det - 1.0) < 1e-3,
    }


def _orthonormalize_pose(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Project the rotation block to the closest proper rotation via SVD.

    Some OpenCV PPF builds return a pose whose rotation block is uniformly
    scaled (norms != 1). When that happens the rendered CAD shrinks by that
    factor and downstream scoring breaks. Fix by SVD-projecting onto SO(3)
    while preserving the translation column.
    """
    M = np.asarray(matrix, dtype=np.float64).copy()
    R = M[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_fixed = U @ Vt
    if np.linalg.det(R_fixed) < 0.0:
        U[:, -1] *= -1.0
        R_fixed = U @ Vt
    M[:3, :3] = R_fixed
    info = {
        "before": _rotation_diagnostics(matrix),
        "after": _rotation_diagnostics(M),
    }
    return M.astype(np.float32), info


def _finite_float(value: Any, default: float) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(num):
        return float(default)
    return float(num)


def _finite_int(value: Any, default: int) -> int:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return int(default)
    return int(num)


def _ppf_namespace() -> Any | None:
    return getattr(cv2, "ppf_match_3d", None)


def _lookup_ppf_api() -> PpfApi | None:
    ns = _ppf_namespace()
    detector_ctor = None
    icp_ctor = None
    compute_normals = None

    if ns is not None:
        detector_ctor = (
            getattr(ns, "PPF3DDetector", None)
            or getattr(ns, "PPF3DDetector_create", None)
        )
        icp_ctor = getattr(ns, "ICP", None) or getattr(ns, "ICP_create", None)
        compute_normals = getattr(ns, "computeNormalsPC3d", None)

    detector_ctor = (
        detector_ctor
        or getattr(cv2, "ppf_match_3d_PPF3DDetector", None)
        or getattr(cv2, "ppf_match_3d_PPF3DDetector_create", None)
    )
    icp_ctor = (
        icp_ctor
        or getattr(cv2, "ppf_match_3d_ICP", None)
        or getattr(cv2, "ppf_match_3d_ICP_create", None)
    )
    compute_normals = (
        compute_normals
        or getattr(cv2, "ppf_match_3d_computeNormalsPC3d", None)
    )
    if detector_ctor is None or icp_ctor is None or compute_normals is None:
        return None
    return PpfApi(
        detector_ctor=detector_ctor,
        icp_ctor=icp_ctor,
        compute_normals=compute_normals,
    )


def ppf_dependency_available() -> bool:
    return _lookup_ppf_api() is not None


def _make_ppf_detector(api: PpfApi, params: Dict[str, Any]) -> Any:
    rel_sampling = _finite_float(params.get("ppf_relative_sampling_step"), 0.025)
    rel_distance = _finite_float(params.get("ppf_relative_distance_step"), 0.025)
    num_angles = _finite_int(params.get("ppf_num_angles"), 36)
    ctor = api.detector_ctor
    attempts = (
        lambda: ctor(rel_sampling, rel_distance, num_angles),
        lambda: ctor(
            relativeSamplingStep=rel_sampling,
            relativeDistanceStep=rel_distance,
            numAngles=num_angles,
        ),
        lambda: ctor(),
    )
    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            detector = attempt()
            break
        except Exception as exc:
            last_exc = exc
    else:
        raise RuntimeError(f"ppf_icp_detector_create_failed:{last_exc}")

    set_search_params = getattr(detector, "setSearchParams", None)
    if callable(set_search_params):
        pos_threshold = _finite_float(params.get("ppf_search_position_threshold"), 0.02)
        rot_threshold = _finite_float(params.get("ppf_search_rotation_threshold"), 0.087)
        weighted = bool(params.get("ppf_weighted_clustering", False))
        try:
            set_search_params(pos_threshold, rot_threshold, weighted)
        except TypeError:
            try:
                set_search_params(
                    positionThreshold=pos_threshold,
                    rotationThreshold=rot_threshold,
                    useWeightedClustering=weighted,
                )
            except Exception:
                pass
        except Exception:
            pass
    return detector


def _mesh_unit_scale(mesh_units: str, mesh_scale: float) -> float:
    unit = str(mesh_units or "mm").strip().lower()
    base = 1.0 if unit == "m" else 0.001
    return base * float(mesh_scale)


def _load_mesh(mesh_path: Path) -> Any:
    import trimesh

    mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError("ppf_icp_mesh_missing_geometry")
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)!r}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError("ppf_icp_mesh_empty")
    return mesh


def build_cad_model_cloud(
    mesh_path: Path,
    *,
    mesh_units: str = "mm",
    mesh_scale: float = 1.0,
    sample_points: int = 1000,
) -> np.ndarray:
    import trimesh

    mesh = _load_mesh(mesh_path)
    sample_count = max(64, int(sample_points or 8000))
    points, face_indices = trimesh.sample.sample_surface(mesh, sample_count)
    points = np.asarray(points, dtype=np.float32)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float32)
    scale_to_m = _mesh_unit_scale(mesh_units, mesh_scale)
    points = points * np.float32(scale_to_m)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)
    pc = np.concatenate([points, normals], axis=1).astype(np.float32)
    finite = np.all(np.isfinite(pc), axis=1)
    pc = pc[finite]
    if pc.shape[0] == 0:
        raise ValueError("ppf_icp_model_cloud_empty")
    return pc


def _model_cache_key(mesh_path: Path, params: Dict[str, Any]) -> tuple[Any, ...]:
    stat = mesh_path.stat()
    mesh_units = str(params.get("mesh_units", "mm")).strip().lower() or "mm"
    mesh_scale = _finite_float(params.get("mesh_scale"), 1.0)
    return (
        str(mesh_path.resolve()),
        int(stat.st_mtime_ns),
        mesh_units,
        mesh_scale,
        _finite_int(params.get("ppf_model_sample_points"), 1500),
        _finite_float(params.get("ppf_relative_sampling_step"), 0.025),
        _finite_float(params.get("ppf_relative_distance_step"), 0.025),
        _finite_int(params.get("ppf_num_angles"), 36),
        _finite_float(params.get("ppf_search_position_threshold"), 0.02),
        _finite_float(params.get("ppf_search_rotation_threshold"), 0.087),
        bool(params.get("ppf_weighted_clustering", False)),
    )


def get_ppf_model_bundle(
    mesh_path: Path,
    params: Dict[str, Any],
    *,
    request_id: str | None = None,
) -> PpfModelBundle:
    api = _lookup_ppf_api()
    if api is None:
        raise RuntimeError("ppf_icp_dependency_missing")
    key = _model_cache_key(mesh_path, params)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    started_at = time.perf_counter()
    mesh_units = str(params.get("mesh_units", "mm")).strip().lower() or "mm"
    mesh_scale = _finite_float(params.get("mesh_scale"), 1.0)
    model_pc = build_cad_model_cloud(
        mesh_path,
        mesh_units=mesh_units,
        mesh_scale=mesh_scale,
        sample_points=_finite_int(params.get("ppf_model_sample_points"), 1500),
    )
    detector = _make_ppf_detector(api, params)
    detector.trainModel(np.ascontiguousarray(model_pc, dtype=np.float32))
    bundle = PpfModelBundle(
        detector=detector,
        model_pc=np.ascontiguousarray(model_pc, dtype=np.float32),
        mesh_path=mesh_path,
        mesh_units=mesh_units,
        mesh_scale=mesh_scale,
        cache_key=key,
        lock=threading.Lock(),
    )
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[key] = bundle
    _runtime_log(
        request_id,
        f"ppf_model_ready points={len(model_pc)} dt={time.perf_counter() - started_at:.3f}s",
    )
    return bundle


def _cap_points(points: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if max_points > 0 and len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), int(max_points), replace=False)
        pts = pts[np.sort(idx)]
    return np.ascontiguousarray(pts, dtype=np.float32)


def _extract_normals_output(result: Any) -> np.ndarray | None:
    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, (list, tuple)):
        candidates = [item for item in result if isinstance(item, np.ndarray)]
        for arr in candidates:
            if arr.ndim == 2 and arr.shape[1] >= 6:
                return arr
        return candidates[-1] if candidates else None
    return None


def compute_scene_cloud_with_normals(
    points: np.ndarray,
    *,
    normal_neighbors: int = 16,
) -> np.ndarray:
    api = _lookup_ppf_api()
    if api is None:
        raise RuntimeError("ppf_icp_dependency_missing")
    pts = np.ascontiguousarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    neighbors = max(4, int(normal_neighbors or 16))
    viewpoint = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    fn = api.compute_normals
    result = None
    errors: list[str] = []
    for args in (
        (pts, neighbors, True, viewpoint),
        (pts, neighbors, True),
        (pts, neighbors),
    ):
        try:
            result = fn(*args)
            break
        except Exception as exc:
            errors.append(str(exc))
    pc = _extract_normals_output(result)
    if pc is None:
        raise RuntimeError("ppf_icp_scene_normal_failed")
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 2:
        raise RuntimeError("ppf_icp_scene_normal_failed")
    if pc.shape[1] == 3 and len(pc) == len(pts):
        pc = np.concatenate([pts, pc], axis=1)
    if pc.shape[1] < 6:
        raise RuntimeError("ppf_icp_scene_normal_failed")
    pc = pc[:, :6]
    finite = np.all(np.isfinite(pc), axis=1)
    pc = pc[finite]
    norms = np.linalg.norm(pc[:, 3:6], axis=1)
    pc = pc[norms > 1e-8]
    # Force normals to unit length. OpenCV's PPF detector constructs the
    # pose's rotation block using cross products of point-pair normals;
    # if any normal is non-unit, the resulting "rotation" is scaled and
    # comes back non-orthonormal. We saw implied_scale ~0.12 in practice.
    norms = np.linalg.norm(pc[:, 3:6], axis=1, keepdims=True)
    pc[:, 3:6] = pc[:, 3:6] / np.maximum(norms, 1e-8)
    return np.ascontiguousarray(pc, dtype=np.float32)


def _filter_points_by_depth_band(
    points: np.ndarray,
    *,
    z_band_mad_k: float,
    z_band_min_half_width_m: float,
    z_band_max_half_width_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drop points whose Z is far from the median Z (object slab).

    Background points that bleed in via mask edges (bin floor, neighbours)
    typically sit several centimetres behind the object's median depth and
    appear as a tail in the Z histogram. We clamp to a robust band derived
    from the median absolute deviation.
    """
    info: dict[str, Any] = {"applied": False}
    if len(points) < 8 or z_band_mad_k <= 0.0:
        return points, info
    z = np.asarray(points[:, 2], dtype=np.float64)
    median_z = float(np.median(z))
    mad = float(np.median(np.abs(z - median_z)))
    half_width = max(
        float(z_band_min_half_width_m),
        min(float(z_band_max_half_width_m), float(z_band_mad_k) * mad * 1.4826),
    )
    keep = np.abs(z - median_z) <= half_width
    info.update(
        {
            "applied": True,
            "median_z_m": median_z,
            "mad_m": mad,
            "half_width_m": half_width,
            "input_count": int(len(points)),
            "kept_count": int(np.count_nonzero(keep)),
            "removed_count": int(len(points) - np.count_nonzero(keep)),
        }
    )
    return np.ascontiguousarray(points[keep], dtype=np.float32), info


def _statistical_outlier_removal(
    points: np.ndarray,
    *,
    k_neighbors: int,
    std_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove points whose mean kNN distance exceeds (mean + std_ratio * std).

    Catches isolated specks and thin bridges to background that survive
    the depth band filter.
    """
    info: dict[str, Any] = {"applied": False}
    n = len(points)
    if n < max(8, k_neighbors + 1) or k_neighbors <= 0 or std_ratio <= 0.0:
        return points, info
    pts = np.asarray(points[:, :3], dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=k_neighbors + 1)
        # Skip self-distance at column 0.
        mean_neighbor_dist = dists[:, 1:].mean(axis=1)
    except Exception:
        # Fallback: brute force for small clouds. Skip filter if cloud big.
        if n > 4000:
            return points, info
        diffs = pts[:, None, :] - pts[None, :, :]
        dist_sq = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        sorted_d = np.sort(np.sqrt(dist_sq), axis=1)[:, :k_neighbors]
        mean_neighbor_dist = sorted_d.mean(axis=1)
    global_mean = float(mean_neighbor_dist.mean())
    global_std = float(mean_neighbor_dist.std())
    threshold = global_mean + float(std_ratio) * global_std
    keep = mean_neighbor_dist <= threshold
    info.update(
        {
            "applied": True,
            "k_neighbors": int(k_neighbors),
            "std_ratio": float(std_ratio),
            "mean_neighbor_distance_m": global_mean,
            "std_neighbor_distance_m": global_std,
            "threshold_m": float(threshold),
            "input_count": int(n),
            "kept_count": int(np.count_nonzero(keep)),
            "removed_count": int(n - np.count_nonzero(keep)),
        }
    )
    return np.ascontiguousarray(pts[keep], dtype=np.float32), info


def _largest_3d_cluster(
    points: np.ndarray,
    *,
    cluster_distance_m: float,
    min_cluster_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the largest 3D-connected cluster.

    Removes background blobs that survived depth-band filtering but are not
    connected to the object in 3D (e.g. a neighbouring object at similar Z).
    """
    info: dict[str, Any] = {"applied": False}
    n = len(points)
    if n < 16 or cluster_distance_m <= 0.0:
        return points, info
    pts = np.asarray(points[:, :3], dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        neighbors = tree.query_ball_tree(tree, r=float(cluster_distance_m))
    except Exception:
        return points, info
    visited = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []
    for seed in range(n):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        members: list[int] = []
        while stack:
            current = stack.pop()
            members.append(current)
            for nb in neighbors[current]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        clusters.append(members)
    if not clusters:
        return points, info
    clusters.sort(key=len, reverse=True)
    largest = clusters[0]
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[largest] = True
    keep_total = int(keep_mask.sum())
    if keep_total < max(16, int(min_cluster_ratio * n)):
        # Largest cluster too small -> filter would be aggressive; skip.
        info.update({"applied": False, "skip_reason": "largest_cluster_too_small"})
        return points, info
    info.update(
        {
            "applied": True,
            "cluster_distance_m": float(cluster_distance_m),
            "cluster_count": int(len(clusters)),
            "input_count": int(n),
            "kept_count": keep_total,
            "removed_count": int(n - keep_total),
            "largest_cluster_ratio": float(keep_total) / float(n),
        }
    )
    return np.ascontiguousarray(pts[keep_mask], dtype=np.float32), info


def _adaptive_filter_thresholds(
    *,
    model_extents: tuple[float, float, float] | None,
    model_diameter: float | None,
    params: Dict[str, Any],
) -> dict[str, float]:
    """Derive scene-filter thresholds from the CAD's known dimensions.

    Reasoning:
    - The Z-band half-width must be at least the **longest CAD axis**
      because the object can be oriented arbitrarily; otherwise we
      truncate one end of a tilted/standing object.
    - The cluster3d distance must be larger than the worst expected
      depth gap *across the object surface*. For curved/cylindrical
      objects, neighbouring pixels along the silhouette can have a
      4-8 mm depth jump even when both belong to the object. Tie the
      cluster distance to a fraction of the object diameter (typically
      ~12 % works well for cylinders / cones).
    - SOR threshold is left scale-independent (it's already adaptive
      via mean+std) but we widen it for small objects where the voxel
      grid produces few neighbours.

    User overrides (params.scene_*) are still respected; we only fill
    defaults when the param is missing or zero.
    """
    if model_extents is None or model_diameter is None or model_diameter <= 0:
        return {}
    longest_axis = float(max(model_extents))
    diameter = float(model_diameter)

    # Z-band half-width: max of (0.6 * longest axis, half diameter, 15 mm).
    # This guarantees the band covers the object even when it sits on its
    # side. For a 42x30x42 mm barrel, longest=42 mm -> half-width 25 mm.
    z_half = max(0.6 * longest_axis, 0.5 * diameter, 0.015)

    # Cluster3d distance: 12 % of diameter, floored at 4 mm, capped at 25 mm.
    cluster_dist = max(0.004, min(0.025, 0.12 * diameter))

    # Cluster3d min ratio: small objects have fewer total points so be
    # permissive; default 5 % is fine but raise to 10 % for big objects
    # so we don't accept tiny stray subclusters as the "main" object.
    cluster_min_ratio = 0.05 if diameter < 0.10 else 0.10

    # MAD-multiplier k: when MAD is artificially small (e.g. flat top of
    # a cylinder seen end-on), 4*MAD severely truncates the curved sides.
    # Bumping k to 6 keeps the band reasonable; the half-width clamp is
    # what really protects us.
    z_band_k = 6.0

    # SOR std ratio: looser for small objects to avoid clipping the
    # curved silhouette. 1.5 is fine for big objects, 2.0 for small.
    sor_std = 2.0 if diameter < 0.10 else 1.5

    return {
        "z_band_min_half_width_m": z_half,
        "z_band_max_half_width_m": max(z_half, 0.10),
        "z_band_mad_k": z_band_k,
        "cluster3d_distance_m": cluster_dist,
        "cluster3d_min_ratio": cluster_min_ratio,
        "sor_std_ratio": sor_std,
    }


def _resolve_param(params: Dict[str, Any], key: str, fallback: float) -> float:
    """Return the user-supplied value if it's a positive finite number,
    otherwise the adaptive fallback. Treats 0 / None / missing as "use
    adaptive default" so config files that just omit the key get the
    smart value automatically."""
    raw = params.get(key)
    if raw is None:
        return float(fallback)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(v) or v <= 0.0:
        return float(fallback)
    return v


def build_scene_cloud(
    depth_m: np.ndarray | None,
    K: np.ndarray,
    mask: np.ndarray,
    params: Dict[str, Any],
    *,
    model_extents: tuple[float, float, float] | None = None,
    model_diameter: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_points = depth_to_point_cloud_points(
        depth_m,
        np.asarray(K, dtype=np.float32),
        mask=mask.astype(bool),
        max_depth_m=_coerce_float(params.get("depth_max_m"), None),
    )
    raw_count = int(len(raw_points))

    filter_info: dict[str, Any] = {}
    stage_clouds: dict[str, np.ndarray] = {"raw": np.asarray(raw_points, dtype=np.float32)}

    adaptive = _adaptive_filter_thresholds(
        model_extents=model_extents,
        model_diameter=model_diameter,
        params=params,
    )
    filter_info["adaptive_defaults"] = {
        "model_extents_m": list(model_extents) if model_extents else None,
        "model_diameter_m": float(model_diameter) if model_diameter else None,
        **adaptive,
    }

    points = raw_points
    # 1) Biggest 3D-connected cluster.
    if bool(params.get("scene_cluster3d_enabled", True)):
        points, filter_info["cluster3d"] = _largest_3d_cluster(
            points,
            cluster_distance_m=_resolve_param(
                params,
                "scene_cluster3d_distance_m",
                adaptive.get("cluster3d_distance_m", 0.008),
            ),
            min_cluster_ratio=_resolve_param(
                params,
                "scene_cluster3d_min_ratio",
                adaptive.get("cluster3d_min_ratio", 0.05),
            ),
        )
        stage_clouds["after_cluster3d"] = np.asarray(points, dtype=np.float32)

    # 2) Z-band gate.
    if bool(params.get("scene_z_band_filter_enabled", True)):
        points, filter_info["z_band"] = _filter_points_by_depth_band(
            points,
            z_band_mad_k=_resolve_param(
                params, "scene_z_band_mad_k", adaptive.get("z_band_mad_k", 4.0)
            ),
            z_band_min_half_width_m=_resolve_param(
                params,
                "scene_z_band_min_half_width_m",
                adaptive.get("z_band_min_half_width_m", 0.005),
            ),
            z_band_max_half_width_m=_resolve_param(
                params,
                "scene_z_band_max_half_width_m",
                adaptive.get("z_band_max_half_width_m", 0.06),
            ),
        )
        stage_clouds["after_z_band"] = np.asarray(points, dtype=np.float32)

    # 3) Statistical outlier removal.
    if bool(params.get("scene_sor_enabled", True)):
        points, filter_info["statistical_outlier"] = _statistical_outlier_removal(
            points,
            k_neighbors=_finite_int(params.get("scene_sor_k"), 16),
            std_ratio=_resolve_param(
                params, "scene_sor_std_ratio", adaptive.get("sor_std_ratio", 1.5)
            ),
        )
        stage_clouds["after_sor"] = np.asarray(points, dtype=np.float32)

    voxel_size = _finite_float(params.get("ppf_scene_voxel_size_m"), 0.002)
    downsampled = voxel_downsample_points(points, voxel_size)
    cap = max(0, _finite_int(params.get("ppf_scene_max_points"), 12000))
    capped = _cap_points(downsampled, cap)
    scene_pc = compute_scene_cloud_with_normals(
        capped,
        normal_neighbors=_finite_int(params.get("ppf_normal_neighbors"), 24),
    )
    stage_clouds["final_filtered"] = np.asarray(points, dtype=np.float32)
    stage_clouds["downsampled"] = np.asarray(downsampled, dtype=np.float32)
    return scene_pc, {
        "raw_point_count": raw_count,
        "filtered_point_count": int(len(points)),
        "downsampled_point_count": int(len(downsampled)),
        "point_count": int(len(scene_pc)),
        "max_points": cap,
        "voxel_size_m": voxel_size,
        "filters": filter_info,
        "stage_clouds": stage_clouds,
    }


def _pose_matrix_from_cv_pose(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        matrix = value
    else:
        matrix = None
        for attr_name in (
            "pose",
            "poseModelToScene",
            "pose_model_to_scene",
            "modelToScene",
            "model_to_scene",
        ):
            attr = getattr(value, attr_name, None)
            if attr is not None:
                matrix = attr() if callable(attr) else attr
                break
        if matrix is None:
            return None
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return matrix.astype(np.float32)


def _pose_attr_float(value: Any, attr_names: tuple[str, ...]) -> float | None:
    for name in attr_names:
        attr = getattr(value, name, None)
        if attr is None:
            continue
        try:
            attr_value = attr() if callable(attr) else attr
            num = float(attr_value)
        except Exception:
            continue
        if math.isfinite(num):
            return num
    return None


def _iter_ppf_poses(raw_poses: Any, *, limit: int) -> list[PpfPoseHypothesis]:
    if raw_poses is None:
        return []
    if isinstance(raw_poses, tuple) and raw_poses:
        arrays = [item for item in raw_poses if isinstance(item, (list, tuple))]
        if arrays:
            raw_poses = arrays[-1]
    try:
        pose_values = list(raw_poses)
    except TypeError:
        pose_values = [raw_poses]

    hypotheses: list[PpfPoseHypothesis] = []
    for idx, pose_value in enumerate(pose_values[: max(1, int(limit))]):
        matrix = _pose_matrix_from_cv_pose(pose_value)
        if matrix is None:
            continue
        votes = _pose_attr_float(
            pose_value,
            ("numVotes", "num_votes", "votes", "getNumVotes"),
        )
        residual = _pose_attr_float(pose_value, ("residual", "getResidual"))
        angle = _pose_attr_float(pose_value, ("angle", "getAngle"))
        hypotheses.append(
            PpfPoseHypothesis(
                index=idx,
                pose_matrix=matrix,
                votes=float(votes if votes is not None else 0.0),
                residual=residual,
                angle=angle,
            )
        )
    return hypotheses


def _transform_pc(pc: np.ndarray, pose_matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(pose_matrix, dtype=np.float32)
    points = np.asarray(pc[:, :3], dtype=np.float32)
    normals = np.asarray(pc[:, 3:6], dtype=np.float32)
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    out_points = (rot @ points.T).T + trans
    out_normals = (rot @ normals.T).T
    out = np.concatenate([out_points, out_normals], axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


def _make_icp(api: PpfApi, params: Dict[str, Any]) -> Any:
    ctor = api.icp_ctor
    sample_type_raw = params.get("icp_sample_type", "uniform")
    if isinstance(sample_type_raw, str):
        sample_type = 0 if sample_type_raw.strip().lower() != "gelfand" else 1
    else:
        sample_type = _finite_int(sample_type_raw, 0)
    args = (
        _finite_int(params.get("icp_iterations"), 80),
        _finite_float(params.get("icp_tolerance"), 0.005),
        _finite_float(params.get("icp_rejection_scale"), 4.0),
        _finite_int(params.get("icp_num_levels"), 6),
        int(sample_type),
        _finite_int(params.get("icp_num_max_corr"), 1),
    )
    attempts = (
        lambda: ctor(*args),
        lambda: ctor(
            iterations=args[0],
            tolerence=args[1],
            rejectionScale=args[2],
            numLevels=args[3],
            sampleType=args[4],
            numMaxCorr=args[5],
        ),
        lambda: ctor(),
    )
    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"ppf_icp_create_failed:{last_exc}")


def _parse_icp_result(result: Any) -> tuple[float | None, np.ndarray | None]:
    residual = None
    pose = None
    if isinstance(result, np.ndarray):
        pose = result
    elif isinstance(result, (list, tuple)):
        arrays = [item for item in result if isinstance(item, np.ndarray)]
        if arrays:
            pose = arrays[-1]
        numbers: list[float] = []
        for item in result:
            try:
                num = float(item)
            except Exception:
                continue
            if math.isfinite(num):
                numbers.append(num)
        if numbers:
            residual = numbers[-1]
    matrix = _pose_matrix_from_cv_pose(pose)
    return residual, matrix


def _refine_hypotheses_with_icp(
    hypotheses: list[PpfPoseHypothesis],
    *,
    model_pc: np.ndarray,
    scene_pc: np.ndarray,
    params: Dict[str, Any],
    debug: "_PpfDebugLogger | None" = None,
) -> list[PpfPoseHypothesis]:
    if not hypotheses or not bool(params.get("icp_enabled", True)):
        if debug is not None and debug.enabled:
            debug.event(
                "icp.skipped",
                reason="disabled" if not bool(params.get("icp_enabled", True)) else "no_hypotheses",
            )
        return hypotheses
    api = _lookup_ppf_api()
    if api is None:
        raise RuntimeError("ppf_icp_dependency_missing")
    max_residual = _finite_float(params.get("icp_max_residual_m"), 0.03)
    refined: list[PpfPoseHypothesis] = []
    for hyp in hypotheses:
        candidate = PpfPoseHypothesis(
            index=hyp.index,
            pose_matrix=np.asarray(hyp.pose_matrix, dtype=np.float32).copy(),
            votes=hyp.votes,
            residual=hyp.residual,
            angle=hyp.angle,
            icp_applied=False,
            icp_residual=None,
            icp_accepted=True,
        )
        pre_icp_pose = candidate.pose_matrix.copy()
        coarse_pose: np.ndarray | None = None
        coarse_residual: float | None = None
        fine_residual: float | None = None
        try:
            # Stage A: COARSE ICP — wider rejection scale, fewer iterations,
            # fewer multi-resolution levels. Pulls the PPF hypothesis into
            # the right basin even when it starts a few cm off.
            coarse_params = dict(params)
            coarse_params.setdefault(
                "_coarse_iterations",
                _finite_int(params.get("icp_coarse_iterations"), 50),
            )
            coarse_params.setdefault(
                "_coarse_rejection",
                _finite_float(params.get("icp_coarse_rejection_scale"), 3.0),
            )
            coarse_params.setdefault(
                "_coarse_levels",
                _finite_int(params.get("icp_coarse_num_levels"), 4),
            )
            coarse_params.setdefault(
                "_coarse_tolerance",
                _finite_float(params.get("icp_coarse_tolerance"), 0.02),
            )
            coarse_overrides = dict(params)
            coarse_overrides["icp_iterations"] = coarse_params["_coarse_iterations"]
            coarse_overrides["icp_rejection_scale"] = coarse_params["_coarse_rejection"]
            coarse_overrides["icp_num_levels"] = coarse_params["_coarse_levels"]
            coarse_overrides["icp_tolerance"] = coarse_params["_coarse_tolerance"]

            icp_coarse = _make_icp(api, coarse_overrides)
            transformed_model = _transform_pc(model_pc, candidate.pose_matrix)
            result_coarse = icp_coarse.registerModelToScene(
                np.ascontiguousarray(transformed_model, dtype=np.float32),
                np.ascontiguousarray(scene_pc, dtype=np.float32),
            )
            coarse_residual, delta_coarse = _parse_icp_result(result_coarse)
            delta_coarse_diag = None
            if delta_coarse is not None:
                delta_coarse_diag = _rotation_diagnostics(delta_coarse)
                candidate.pose_matrix = (
                    delta_coarse @ candidate.pose_matrix
                ).astype(np.float32)
            coarse_pose = candidate.pose_matrix.copy()

            # Stage B: FINE ICP — tight rejection scale, more iterations,
            # more levels. Polishes alignment to the millimetre.
            icp_fine = _make_icp(api, params)
            transformed_model = _transform_pc(model_pc, candidate.pose_matrix)
            result_fine = icp_fine.registerModelToScene(
                np.ascontiguousarray(transformed_model, dtype=np.float32),
                np.ascontiguousarray(scene_pc, dtype=np.float32),
            )
            fine_residual, delta_fine = _parse_icp_result(result_fine)
            delta_fine_diag = None
            if delta_fine is not None:
                delta_fine_diag = _rotation_diagnostics(delta_fine)
                candidate.pose_matrix = (
                    delta_fine @ candidate.pose_matrix
                ).astype(np.float32)

            residual = fine_residual if fine_residual is not None else coarse_residual
            candidate.icp_applied = True
            candidate.icp_residual = residual
            if residual is not None and max_residual > 0.0:
                candidate.icp_accepted = float(residual) <= max_residual

            if debug is not None and debug.enabled:
                model_at_ppf = _transform_pc(model_pc, pre_icp_pose)
                model_at_coarse = _transform_pc(model_pc, coarse_pose)
                model_at_fine = _transform_pc(model_pc, candidate.pose_matrix)
                # Save both the model at each pose AND a combined cloud
                # (model points first, scene points appended) so opening a
                # single PLY in MeshLab/CloudCompare immediately shows the
                # overlay quality without manually loading two files.
                debug.save_point_cloud(
                    f"icp_hyp{int(hyp.index)}_a_model_pre_icp", model_at_ppf
                )
                debug.save_point_cloud(
                    f"icp_hyp{int(hyp.index)}_b_model_after_coarse", model_at_coarse
                )
                debug.save_point_cloud(
                    f"icp_hyp{int(hyp.index)}_c_model_after_fine", model_at_fine
                )
                try:
                    scene_xyz = np.asarray(scene_pc[:, :3], dtype=np.float32)
                    debug.save_point_cloud(
                        f"icp_hyp{int(hyp.index)}_overlay_pre",
                        np.vstack([model_at_ppf[:, :3], scene_xyz]),
                    )
                    debug.save_point_cloud(
                        f"icp_hyp{int(hyp.index)}_overlay_coarse",
                        np.vstack([model_at_coarse[:, :3], scene_xyz]),
                    )
                    debug.save_point_cloud(
                        f"icp_hyp{int(hyp.index)}_overlay_fine",
                        np.vstack([model_at_fine[:, :3], scene_xyz]),
                    )
                except Exception:
                    pass
                debug.save_json(
                    f"icp_hyp{int(hyp.index)}_pose_delta",
                    {
                        "pre_icp_pose": pre_icp_pose.tolist(),
                        "coarse_pose": coarse_pose.tolist(),
                        "fine_pose": candidate.pose_matrix.tolist(),
                        "coarse_translation_delta_m": (
                            np.asarray(coarse_pose[:3, 3])
                            - np.asarray(pre_icp_pose[:3, 3])
                        ).tolist(),
                        "fine_translation_delta_m": (
                            np.asarray(candidate.pose_matrix[:3, 3])
                            - np.asarray(coarse_pose[:3, 3])
                        ).tolist(),
                        "total_translation_delta_m": (
                            np.asarray(candidate.pose_matrix[:3, 3])
                            - np.asarray(pre_icp_pose[:3, 3])
                        ).tolist(),
                        "coarse_residual_m": coarse_residual,
                        "fine_residual_m": fine_residual,
                        "icp_accepted": bool(candidate.icp_accepted),
                        "max_residual_threshold_m": float(max_residual),
                        "delta_coarse_rotation": delta_coarse_diag,
                        "delta_fine_rotation": delta_fine_diag,
                    },
                )
                debug.event(
                    "icp.refine",
                    hypothesis_index=int(hyp.index),
                    coarse_residual=coarse_residual,
                    fine_residual=fine_residual,
                    icp_accepted=bool(candidate.icp_accepted),
                    refined_pose=candidate.pose_matrix.tolist(),
                    refined_rotation=_rotation_diagnostics(candidate.pose_matrix),
                )
        except Exception as exc:
            candidate.icp_applied = False
            if debug is not None and debug.enabled:
                debug.event(
                    "icp.refine_failed",
                    hypothesis_index=int(hyp.index),
                    error=str(exc),
                )
        refined.append(candidate)
    return refined


def _run_ppf_icp(
    *,
    bundle: PpfModelBundle,
    scene_pc: np.ndarray,
    params: Dict[str, Any],
    debug: "_PpfDebugLogger | None" = None,
    stage_label: str = "primary",
) -> PpfPoseHypothesis | None:
    sample_step = _finite_float(params.get("ppf_scene_sample_step"), 0.05)
    scene_distance = _finite_float(params.get("ppf_scene_distance"), 0.015)
    max_hypotheses = max(1, _finite_int(params.get("ppf_max_hypotheses"), 20))

    def _normal_stats(pc: np.ndarray) -> dict[str, Any]:
        if pc is None or len(pc) == 0 or pc.shape[1] < 6:
            return {"present": False}
        norms = np.linalg.norm(np.asarray(pc[:, 3:6], dtype=np.float64), axis=1)
        return {
            "present": True,
            "count": int(len(pc)),
            "min": float(norms.min()),
            "max": float(norms.max()),
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "non_unit_fraction": float(np.mean(np.abs(norms - 1.0) > 1e-3)),
            "zero_or_nan_count": int(np.sum(~np.isfinite(norms)) + np.sum(norms < 1e-6)),
        }

    if debug is not None and debug.enabled:
        debug.event(
            f"ppf.match.start[{stage_label}]",
            sample_step=sample_step,
            scene_distance=scene_distance,
            max_hypotheses=max_hypotheses,
            scene_point_count=int(len(scene_pc)),
            scene_stats=_point_cloud_stats(scene_pc),
            model_point_count=int(len(bundle.model_pc)),
            model_stats=_point_cloud_stats(bundle.model_pc),
            model_normal_stats=_normal_stats(bundle.model_pc),
            scene_normal_stats=_normal_stats(scene_pc),
            ppf_relative_sampling_step=_finite_float(
                params.get("ppf_relative_sampling_step"), 0.025
            ),
            ppf_relative_distance_step=_finite_float(
                params.get("ppf_relative_distance_step"), 0.025
            ),
            ppf_num_angles=_finite_int(params.get("ppf_num_angles"), 36),
        )
        # Save full clouds with normals (ASCII PLY) so the user can confirm
        # in MeshLab whether normals are unit-length and consistently
        # oriented. Non-unit normals are the #1 cause of OpenCV PPF
        # returning a non-orthonormal rotation block.
        debug.save_point_cloud_with_normals(
            f"ppf_model_input_{stage_label}", bundle.model_pc
        )
        debug.save_point_cloud_with_normals(
            f"ppf_scene_input_{stage_label}", scene_pc
        )
        # Replicate PPF's voxel subsampling outside OpenCV so we can show
        # WHICH points PPF actually feeds into its hash table.
        try:
            model_diam = float(_point_cloud_stats(bundle.model_pc).get("diameter", 0.0))
            sampling_step_m = float(
                _finite_float(params.get("ppf_relative_sampling_step"), 0.025)
            ) * max(model_diam, 1e-6)
            scene_sampling_step_m = float(sample_step) * max(model_diam, 1e-6)
            model_features = voxel_downsample_points(
                np.asarray(bundle.model_pc[:, :3], dtype=np.float32),
                sampling_step_m,
            )
            scene_features = voxel_downsample_points(
                np.asarray(scene_pc[:, :3], dtype=np.float32),
                scene_sampling_step_m,
            )
            # Sample a handful of representative point pairs (at fixed
            # distances) so the user can see the PPF feature pairs the
            # detector hashes. We don't have access to the actual matched
            # pairs through OpenCV, but showing the candidate pair set
            # makes the algorithm visible.
            def _sample_pairs(pts, k=120):
                if len(pts) < 4:
                    return np.zeros((0, 2), dtype=np.int64)
                rng = np.random.default_rng(0)
                a = rng.integers(0, len(pts), size=k)
                b = rng.integers(0, len(pts), size=k)
                keep = a != b
                return np.stack([a[keep], b[keep]], axis=1)

            model_pairs = _sample_pairs(model_features, k=120)
            scene_pairs = _sample_pairs(scene_features, k=120)
            debug.save_features_overlay(
                f"ppf_model_{stage_label}",
                model_features,
                feature_pair_indices=model_pairs,
            )
            debug.save_features_overlay(
                f"ppf_scene_{stage_label}",
                scene_features,
                feature_pair_indices=scene_pairs,
            )
            debug.event(
                f"ppf.feature_subsampling[{stage_label}]",
                model_diameter_m=model_diam,
                model_sampling_step_m=sampling_step_m,
                scene_sampling_step_m=scene_sampling_step_m,
                model_feature_count=int(len(model_features)),
                scene_feature_count=int(len(scene_features)),
            )
        except Exception as exc:
            _runtime_log(debug.request_id, f"ppf_feature_save_failed err={exc}")
    with bundle.lock:
        raw_poses = bundle.detector.match(
            np.ascontiguousarray(scene_pc, dtype=np.float32),
            sample_step,
            scene_distance,
        )
    hypotheses = _iter_ppf_poses(raw_poses, limit=max_hypotheses)
    if debug is not None and debug.enabled:
        debug.event(
            f"ppf.match.done[{stage_label}]",
            hypothesis_count=len(hypotheses),
            hypotheses=[
                {
                    "index": int(h.index),
                    "votes": float(h.votes),
                    "residual": h.residual,
                    "angle": h.angle,
                    "pose_matrix": h.pose_matrix.tolist(),
                    "rotation_diag": _rotation_diagnostics(h.pose_matrix),
                }
                for h in hypotheses
            ],
        )
        # Save the scene PPF saw and the model placed at each raw hypothesis
        # (before any orthonormalisation / ICP) so failures can be inspected
        # cloud-by-cloud in MeshLab.
        debug.save_point_cloud(f"ppf_scene_{stage_label}", scene_pc)
        for h in hypotheses:
            try:
                debug.save_point_cloud(
                    f"ppf_hyp{int(h.index)}_model_at_pose_raw",
                    _transform_pc(bundle.model_pc, h.pose_matrix),
                )
            except Exception as exc:
                _runtime_log(
                    debug.request_id,
                    f"ppf_hypothesis_save_failed index={h.index} err={exc}",
                )
    if not hypotheses:
        return None
    best_votes = max(float(h.votes) for h in hypotheses)
    min_vote_ratio = max(0.0, _finite_float(params.get("ppf_min_vote_ratio"), 0.2))
    min_votes_floor = max(0.0, _finite_float(params.get("ppf_min_votes"), 800.0))
    min_votes = max(min_votes_floor, best_votes * min_vote_ratio)
    vote_filtered = [h for h in hypotheses if float(h.votes) >= min_votes]
    if not vote_filtered:
        vote_filtered = hypotheses[:1]
    if len(vote_filtered) != len(hypotheses):
        if debug is not None and debug.enabled:
            debug.event(
                f"ppf.vote_filter[{stage_label}]",
                before_count=len(hypotheses),
                after_count=len(vote_filtered),
                best_votes=best_votes,
                min_votes=min_votes,
                min_vote_ratio=min_vote_ratio,
                min_votes_floor=min_votes_floor,
                dropped=[
                    {"index": int(h.index), "votes": float(h.votes)}
                    for h in hypotheses
                    if float(h.votes) < min_votes
                ],
            )
        hypotheses = vote_filtered
    fix_scaled = bool(params.get("ppf_orthonormalize_pose", True))
    if fix_scaled:
        cleaned: list[PpfPoseHypothesis] = []
        for hyp in hypotheses:
            diag = _rotation_diagnostics(hyp.pose_matrix)
            if not diag.get("is_orthonormal", True):
                fixed_pose, fix_info = _orthonormalize_pose(hyp.pose_matrix)
                _runtime_log(
                    debug.request_id if debug is not None else None,
                    "ppf_pose_orthonormalized "
                    f"index={hyp.index} implied_scale={diag['implied_scale']:.4f} "
                    f"orthonormal_err={diag['orthonormal_err']:.4f} det={diag['det']:.4f}",
                )
                if debug is not None and debug.enabled:
                    debug.event(
                        f"ppf.orthonormalize[{stage_label}]",
                        hypothesis_index=int(hyp.index),
                        info=fix_info,
                    )
                hyp = PpfPoseHypothesis(
                    index=hyp.index,
                    pose_matrix=fixed_pose,
                    votes=hyp.votes,
                    residual=hyp.residual,
                    angle=hyp.angle,
                )
            cleaned.append(hyp)
        hypotheses = cleaned
    hypotheses = _refine_hypotheses_with_icp(
        hypotheses,
        model_pc=bundle.model_pc,
        scene_pc=scene_pc,
        params=params,
        debug=debug,
    )

    def _sort_key(hyp: PpfPoseHypothesis) -> tuple[bool, float, float, int]:
        residual = hyp.icp_residual
        if residual is None:
            residual = hyp.residual
        residual_value = float(residual) if residual is not None else float("inf")
        return (
            bool(not hyp.icp_accepted),
            residual_value,
            -float(hyp.votes),
            int(hyp.index),
        )

    return sorted(hypotheses, key=_sort_key)[0]


def _build_ppf_output_dir(params: Dict[str, Any], request_id: str | None, label: str) -> Path | None:
    if not bool(params.get("save_outputs", False)):
        return None
    root_value = params.get("output_root") or DEFAULT_OUTPUT_ROOT
    root = resolve_workspace_path(root_value, workspace_root=WORKSPACE_ROOT)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = _safe_slug(request_id or label)
    output_dir = root / timestamp / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _build_safety_point_clouds(
    *,
    depth_m: np.ndarray | None,
    K: np.ndarray,
    selection_mask: np.ndarray | None,
    params: Dict[str, Any],
    request_id: str | None,
) -> tuple[list[list[float]], list[list[float]], dict[str, Any]]:
    target_points_camera: list[list[float]] = []
    neighbor_points_camera: list[list[float]] = []
    meta: dict[str, Any] = {
        "frame": "camera",
        "voxel_size_m": float(_coerce_float(params.get("safety_pcd_voxel_size_m"), 0.003) or 0.003),
        "max_depth_m": _coerce_float(params.get("depth_max_m"), None),
        "max_points_per_cloud": int(_coerce_float(params.get("safety_pcd_max_points"), 20000) or 20000),
    }
    try:
        if depth_m is None or selection_mask is None:
            return target_points_camera, neighbor_points_camera, meta
        target_mask = selection_mask.astype(bool)
        neighbor_mask = ~target_mask
        raw_target = depth_to_point_cloud_points(
            depth_m,
            K,
            mask=target_mask,
            max_depth_m=meta["max_depth_m"],
        )
        raw_neighbor = depth_to_point_cloud_points(
            depth_m,
            K,
            mask=neighbor_mask,
            max_depth_m=meta["max_depth_m"],
        )
        tgt_ds = voxel_downsample_points(raw_target, float(meta["voxel_size_m"]))
        nbr_ds = voxel_downsample_points(raw_neighbor, float(meta["voxel_size_m"]))
        cap = int(meta["max_points_per_cloud"])
        tgt_ds = _cap_points(tgt_ds, cap)
        nbr_ds = _cap_points(nbr_ds, cap)
        target_points_camera = tgt_ds.tolist()
        neighbor_points_camera = nbr_ds.tolist()
    except Exception as exc:
        _runtime_log(request_id, f"safety_pcd_build_failed error={exc}")
    meta["target_point_count"] = len(target_points_camera)
    meta["neighbor_point_count"] = len(neighbor_points_camera)
    return target_points_camera, neighbor_points_camera, meta


def _selection_area_filters(params: Dict[str, Any], image_shape: tuple[int, int]) -> tuple[float, float | None]:
    image_area_px = float(image_shape[0] * image_shape[1])
    raw_min_area_px = params.get("selection_min_area_px")
    raw_max_area_px = params.get("selection_max_area_px")
    min_area_ratio = _finite_float(params.get("selection_min_area_ratio"), 0.0)
    max_area_ratio = _finite_float(params.get("selection_max_area_ratio"), 0.0)
    selection_min_area_px = max(
        0.0,
        float(
            (
                raw_min_area_px
                if raw_min_area_px is not None and float(raw_min_area_px) > 0.0
                else min_area_ratio * image_area_px
            )
            or 0.0
        ),
    )
    selection_max_area_px = max(
        0.0,
        float(
            (
                raw_max_area_px
                if raw_max_area_px is not None and float(raw_max_area_px) > 0.0
                else max_area_ratio * image_area_px
            )
            or 0.0
        ),
    )
    return selection_min_area_px, selection_max_area_px if selection_max_area_px > 0.0 else None


def _bin_roi_polygon(params: Dict[str, Any]) -> list[list[float]] | None:
    raw = params.get("bin_roi") if isinstance(params, dict) else None
    if not isinstance(raw, dict):
        return None
    points = raw.get("obb_points_uv") or []
    if isinstance(points, list) and len(points) == 4:
        try:
            return [[float(p[0]), float(p[1])] for p in points]
        except Exception:
            return None
    return None


def _candidate_depth_masks(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    candidate: dict[str, Any],
    K: np.ndarray,
    params: Dict[str, Any],
) -> dict[str, Any]:
    bbox_xyxy = np.asarray(candidate["bbox_xyxy"], dtype=np.float32)
    masked_rgb, masked_depth, selection_mask, depth_selection_mask, cluster_filter_info = (
        mask_observation_to_selected_object(
            rgb,
            depth_m,
            candidate.get("mask"),
            bbox_xyxy,
            K=np.asarray(K, dtype=np.float32),
            # Disabled by default for PPF: the 3D cluster filter inside
            # build_scene_cloud is more reliable than pixel-space depth
            # cluster filtering, especially on fused/TSDF clouds where
            # the segmentation mask edges may straddle small holes.
            filter_depth_clusters=bool(params.get("segmented_cluster_filter_enabled", False)),
            cluster_distance_m=_finite_float(params.get("segmented_cluster_distance_m"), 0.03),
            min_cluster_points=_finite_int(params.get("segmented_cluster_min_points"), 48),
            min_cluster_ratio=_finite_float(params.get("segmented_cluster_min_ratio"), 0.01),
            rgb_mask_dilation_base_px=0.0,
        )
    )
    return {
        "bbox_xyxy": bbox_xyxy,
        "bbox_xywh": candidate.get("bbox_xywh"),
        "masked_rgb": masked_rgb,
        "masked_depth": masked_depth,
        "selection_mask": selection_mask,
        "depth_selection_mask": depth_selection_mask,
        "segmentation_contours_uv": extract_segmentation_contours(selection_mask),
        "cluster_filter_info": cluster_filter_info,
    }


def _make_match(
    *,
    params: Dict[str, Any],
    assets: Any,
    candidate: dict[str, Any],
    candidate_data: dict[str, Any],
    pose_hypothesis: PpfPoseHypothesis,
    camera_data: dict[str, Any],
    annotation_meta: dict[str, Any],
    duplicate_filter_info: dict[str, Any],
    scene_info: dict[str, Any],
    model_bundle: PpfModelBundle,
    depth_m: np.ndarray | None,
    request_id: str | None,
    debug_paths: dict[str, str] | None = None,
    visualization_3d_paths: dict[str, str | None] | None = None,
    all_poses_match: bool = False,
) -> dict[str, Any]:
    pose_matrix = np.asarray(pose_hypothesis.pose_matrix, dtype=np.float32)
    rotation = pose_matrix[:3, :3]
    translation = pose_matrix[:3, 3]
    object_center_object_m = np.asarray(
        annotation_meta.get("object_center_object_m") or [0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    object_center_camera = (rotation @ object_center_object_m) + translation
    axis_object = _normalize_axis(params.get("axis", [0.0, 0.0, 1.0]))
    axis_camera = rotation @ axis_object
    yaw_deg = None
    planar_norm = math.hypot(float(axis_camera[0]), float(axis_camera[1]))
    if planar_norm > 1e-6:
        yaw_deg = math.degrees(math.atan2(float(axis_camera[1]), float(axis_camera[0])))
    quat_xyzw = rotation_matrix_to_quaternion_xyzw(rotation)
    bbox_xywh = candidate_data.get("bbox_xywh") or candidate.get("bbox_xywh") or []
    center_uv = _project_point_uv(object_center_camera, camera_data["K"]) or [
        float(bbox_xywh[0] + (bbox_xywh[2] * 0.5)) if len(bbox_xywh) >= 3 else 0.0,
        float(bbox_xywh[1] + (bbox_xywh[3] * 0.5)) if len(bbox_xywh) >= 4 else 0.0,
    ]
    selection_mask = candidate_data["selection_mask"]
    target_points_camera, neighbor_points_camera, safety_pcd_meta = _build_safety_point_clouds(
        depth_m=depth_m,
        K=np.asarray(camera_data["K"], dtype=np.float32),
        selection_mask=selection_mask,
        params=params,
        request_id=request_id,
    )
    contours = candidate_data.get("segmentation_contours_uv") or []
    bbox_xyxy = np.asarray(candidate_data["bbox_xyxy"], dtype=np.float32)
    match = {
        "object_id": str(params.get("object_id") or assets.label),
        "label": assets.label,
        "method": "ppf_icp_bin_picking",
        "camera_calibration_source": str(
            params.get("camera_calibration_source") or "module_params"
        ),
        "score": float(candidate.get("score", 0.0)),
        "bbox_xywh": [float(v) for v in bbox_xywh],
        "bbox_xyxy": [float(v) for v in bbox_xyxy.tolist()],
        "area_px": float(candidate.get("area_px", 0.0)),
        "rank": int(candidate.get("rank", 0)),
        "center_uv": center_uv,
        "center_xyz_m": [
            float(object_center_camera[0]),
            float(object_center_camera[1]),
            float(object_center_camera[2]),
        ],
        "depth_m": float(object_center_camera[2]),
        "initial_pose_origin_xyz_m": [
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ],
        "pose_origin_xyz_m": [
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ],
        "annotation_origin_xyz_m": [
            float(object_center_camera[0]),
            float(object_center_camera[1]),
            float(object_center_camera[2]),
        ],
        "yaw_deg": None if yaw_deg is None else float(yaw_deg),
        "surface_normal_cam": [
            float(axis_camera[0]),
            float(axis_camera[1]),
            float(axis_camera[2]),
        ],
        "orientation_axis_camera": [
            float(axis_camera[0]),
            float(axis_camera[1]),
            float(axis_camera[2]),
        ],
        "quaternion_xyzw": [float(v) for v in quat_xyzw.tolist()],
        "pose_quat_xyzw": [float(v) for v in quat_xyzw.tolist()],
        "initial_pose_matrix": pose_matrix.tolist(),
        "pose_matrix": pose_matrix.tolist(),
        "object_center_object_m": [float(v) for v in object_center_object_m.tolist()],
        "selection_mask_pixels": int(np.count_nonzero(selection_mask)),
        "segmentation_contour_uv": contours[0] if contours else [],
        "segmentation_contours_uv": contours,
        "segmented_cluster_filter": candidate_data.get("cluster_filter_info"),
        "segmentation_duplicate_filter": duplicate_filter_info,
        "selection_pool": candidate.get("selection_pool", []),
        "model_name": "ppf_icp",
        "mesh_path": str(assets.mesh_path),
        "segmentation_model_path": str(assets.segmentation_model_path),
        "camera_intrinsics": camera_data.get("intrinsics"),
        "object_extents_m": annotation_meta.get("object_extents_m"),
        "annotation_axis_length_m": annotation_meta.get("annotation_axis_length_m"),
        "ppf_icp": {
            "ppf_hypothesis_index": int(pose_hypothesis.index),
            "ppf_votes": float(pose_hypothesis.votes),
            "ppf_residual": pose_hypothesis.residual,
            "ppf_angle": pose_hypothesis.angle,
            "icp_enabled": bool(params.get("icp_enabled", True)),
            "icp_applied": bool(pose_hypothesis.icp_applied),
            "icp_residual": pose_hypothesis.icp_residual,
            "icp_accepted": bool(pose_hypothesis.icp_accepted),
            "model_point_count": int(len(model_bundle.model_pc)),
            "scene_point_count": int(scene_info.get("point_count", 0)),
            "scene_cloud": scene_info,
        },
        "debug_paths": debug_paths or {},
        "visualization": {},
        "visualization_3d": visualization_3d_paths or {},
        "safety_pcd": {
            **safety_pcd_meta,
            "target_points_camera_m": target_points_camera,
            "neighbor_points_camera_m": neighbor_points_camera,
        },
    }
    if all_poses_match:
        match["all_poses_match"] = True
    return match


def prewarm_ppf_icp_bin_picking(
    params: Dict[str, Any],
    *,
    request_id: str | None = None,
    force_bundle_warm: bool = True,
) -> Dict[str, Any]:
    del force_bundle_warm
    if _lookup_ppf_api() is None:
        return {
            "status": "error",
            "error": "ppf_icp_dependency_missing",
            "module": "ppf_icp_bin_picking",
        }
    raw_object_folder = _clean_path_string((params or {}).get("object_folder"))
    if not raw_object_folder:
        return {
            "status": "error",
            "error": "ppf_icp_object_folder_missing",
            "module": "ppf_icp_bin_picking",
        }
    label_override = (
        str((params or {}).get("label_override") or (params or {}).get("object_id") or "").strip()
        or None
    )
    assets = resolve_object_assets(
        raw_object_folder,
        workspace_root=WORKSPACE_ROOT,
        extra_roots=[WORKSPACE_ROOT / "data"],
        label_override=label_override,
        segmentation_model_override=(params or {}).get("segmentation_model_path"),
        mesh_extension_priority=list((params or {}).get("mesh_extension_priority") or [".glb", ".obj", ".stl"]),
    )
    bundle = get_ppf_model_bundle(
        assets.mesh_path,
        dict(params or {}),
        request_id=request_id,
    )
    return {
        "status": "ok",
        "module": "ppf_icp_bin_picking",
        "label": assets.label,
        "mesh_path": str(assets.mesh_path),
        "segmentation_model_path": str(assets.segmentation_model_path),
        "model_points": int(len(bundle.model_pc)),
        "cache_key": [str(v) for v in bundle.cache_key],
    }


def _no_candidate_result(
    *,
    params: Dict[str, Any],
    detections: list[dict[str, Any]],
    duplicate_filter_info: dict[str, Any],
    selection_min_area_px: float,
    selection_max_area_px: float | None,
    timing_log: _BinPickingTimingLog,
) -> Dict[str, Any]:
    detection_summaries = [
        {
            "rank": int(det.get("rank", idx)),
            "score": float(det.get("score", 0.0)),
            "area_px": float(det.get("area_px", 0.0)),
            "bbox_xywh": [float(v) for v in det.get("bbox_xywh", [])],
        }
        for idx, det in enumerate(detections[:5])
    ]
    timing_log.summary(
        valid=False,
        error="ppf_icp_no_candidate",
        total_detections=len(detections),
        timing_log_path=str(timing_log.path) if timing_log.path else None,
    )
    return {
        "valid": False,
        "matches": [],
        "terminal": True,
        "error": "ppf_icp_no_candidate",
        "details": {
            "detections": len(detections),
            "selection_top_k": int(params.get("selection_top_k", 5)),
            "selection_min_confidence": float(params.get("selection_min_confidence", 0.2)),
            "selection_min_area_px": selection_min_area_px,
            "selection_max_area_px": selection_max_area_px,
            "top_detections": detection_summaries,
            "duplicate_filter": duplicate_filter_info,
        },
    }


def run_ppf_icp_bin_picking(
    *,
    bgr: np.ndarray,
    depth: np.ndarray | None,
    params: Dict[str, Any],
    request_id: str | None = None,
) -> Dict[str, Any]:
    run_started_at = time.perf_counter()
    timing_log = _BinPickingTimingLog.from_params(params, request_id)
    queue_wait_seconds = _coerce_float(params.get("_worker_queue_wait_seconds"), 0.0) or 0.0
    if queue_wait_seconds > 0.0:
        timing_log.span("vision_worker_queue_wait", run_started_at - queue_wait_seconds)
    if _lookup_ppf_api() is None:
        timing_log.summary(valid=False, error="ppf_icp_dependency_missing")
        return {
            "valid": False,
            "matches": [],
            "terminal": True,
            "error": "ppf_icp_dependency_missing",
        }
    if depth is None:
        timing_log.summary(valid=False, error="ppf_icp_requires_depth")
        return {
            "valid": False,
            "matches": [],
            "terminal": True,
            "error": "ppf_icp_requires_depth",
        }

    raw_object_folder = _clean_path_string(params.get("object_folder"))
    if not raw_object_folder:
        raise ValueError("ppf_icp_object_folder_missing")

    setup_started_at = time.perf_counter()
    label_override = (
        str(params.get("label_override") or params.get("object_id") or "").strip() or None
    )
    assets = resolve_object_assets(
        raw_object_folder,
        workspace_root=WORKSPACE_ROOT,
        extra_roots=[WORKSPACE_ROOT / "data"],
        label_override=label_override,
        segmentation_model_override=params.get("segmentation_model_path"),
        mesh_extension_priority=list(
            params.get("mesh_extension_priority") or [".glb", ".obj", ".stl"]
        ),
    )
    timing_log.span("resolve_object_assets", setup_started_at, object_folder=str(assets.object_folder))

    output_dir = _build_ppf_output_dir(params, request_id, assets.label)
    debug_logger = _PpfDebugLogger(output_dir, request_id)
    if debug_logger.enabled:
        debug_logger.event(
            "run.start",
            request_id=str(request_id or ""),
            object_folder=str(assets.object_folder),
            mesh_path=str(assets.mesh_path),
            mesh_units=str(params.get("mesh_units", "mm")),
            mesh_scale=float(params.get("mesh_scale", 1.0)),
            depth_scale=float(params.get("depth_scale", 0.001)),
            params={k: v for k, v in params.items() if not k.startswith("_") and k not in {"camera_data", "intrinsics", "K", "bin_roi"}},
        )

    preprocess_started_at = time.perf_counter()
    color_order = str(params.get("input_color_order", "bgr")).lower()
    if color_order == "rgb":
        rgb = bgr.copy()
        bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr_frame = bgr.copy()
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    depth_m = normalize_depth_map(depth, float(params.get("depth_scale", 0.001)))
    depth_m = filter_depth_range(
        depth_m,
        min_depth_m=_coerce_float(params.get("depth_min_m"), None),
        max_depth_m=_coerce_float(params.get("depth_max_m"), None),
    )
    processing_max_width = int(params.get("processing_max_width", 1280) or 1280)
    processing_max_height = int(params.get("processing_max_height", 720) or 720)
    rgb, depth_m = resize_inputs_for_processing(
        rgb,
        depth_m,
        max_width=processing_max_width,
        max_height=processing_max_height,
    )
    camera_data = build_camera_data(params, rgb.shape[:2])
    timing_log.span(
        "preprocess_inputs",
        preprocess_started_at,
        rgb_shape=tuple(int(v) for v in rgb.shape),
        depth_present=depth_m is not None,
        model_name="ppf_icp",
    )
    if debug_logger.enabled:
        depth_summary: dict[str, Any] = {"present": depth_m is not None}
        if depth_m is not None:
            valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
            depth_summary.update(
                {
                    "shape": list(depth_m.shape),
                    "valid_pixels": int(valid.size),
                    "min_m": float(valid.min()) if valid.size else None,
                    "max_m": float(valid.max()) if valid.size else None,
                    "median_m": float(np.median(valid)) if valid.size else None,
                }
            )
        debug_logger.event(
            "inputs.preprocessed",
            rgb_shape=list(rgb.shape),
            depth=depth_summary,
            camera_K=np.asarray(camera_data["K"], dtype=np.float64).tolist(),
            camera_calibration_source=str(params.get("camera_calibration_source") or "module_params"),
        )
        debug_logger.save_image("rgb_preprocessed", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    bundle_started_at = time.perf_counter()
    model_bundle = get_ppf_model_bundle(assets.mesh_path, params, request_id=request_id)
    annotation_meta = get_mesh_annotation_meta(
        assets.mesh_path,
        str(params.get("mesh_units", "mm")),
        float(params.get("mesh_scale", 1.0)),
    )
    timing_log.span(
        "ppf_model_bundle_ready",
        bundle_started_at,
        model_points=int(len(model_bundle.model_pc)),
    )
    if debug_logger.enabled:
        debug_logger.event(
            "model_bundle.ready",
            model_point_count=int(len(model_bundle.model_pc)),
            model_stats=_point_cloud_stats(model_bundle.model_pc),
            mesh_path=str(model_bundle.mesh_path),
            mesh_units=str(model_bundle.mesh_units),
            mesh_scale=float(model_bundle.mesh_scale),
            annotation_meta=_safe_json(annotation_meta),
        )
        debug_logger.save_point_cloud("model_pc", model_bundle.model_pc)

    segmentation_started_at = time.perf_counter()
    segmentation_backend = resolve_segmentation_backend(
        str(params.get("segmentation_backend", "auto")),
        assets.segmentation_model_path,
    )
    segmentation_device = params.get("segmentation_device", params.get("yolo_device"))
    detections = run_segmentation(
        rgb=rgb,
        model_path=assets.segmentation_model_path,
        backend=segmentation_backend,
        conf=float(params.get("yolo_conf", 0.2)),
        device=str(segmentation_device) if segmentation_device not in (None, "") else None,
        imgsz=int(params.get("yolo_imgsz", 1024) or 1024),
        retina_masks=bool(params.get("retina_masks", True)),
    )
    timing_log.span(
        "segmentation",
        segmentation_started_at,
        backend=segmentation_backend,
        device=str(segmentation_device) if segmentation_device not in (None, "") else "",
        detections=len(detections),
        confidence=float(params.get("yolo_conf", 0.2)),
    )

    duplicate_filter_started_at = time.perf_counter()
    detections, duplicate_filter_info = suppress_nested_duplicate_detections(
        detections,
        image_shape=tuple(rgb.shape[:2]),
        depth_m=depth_m,
    )
    timing_log.span(
        "segmentation_duplicate_filter",
        duplicate_filter_started_at,
        kept=int(duplicate_filter_info.get("kept_count", len(detections))),
        suppressed=int(duplicate_filter_info.get("suppressed_count", 0)),
    )

    selection_min_area_px, selection_max_area_px = _selection_area_filters(params, rgb.shape[:2])
    bin_roi_polygon_uv = _bin_roi_polygon(params)
    candidate_select_started_at = time.perf_counter()
    candidate = select_detection_candidate(
        detections,
        image_shape=rgb.shape[:2],
        depth_m=depth_m,
        K=camera_data["K"],
        top_k=int(params.get("selection_top_k", 5)),
        min_confidence=float(params.get("selection_min_confidence", 0.2)),
        min_area_px=selection_min_area_px,
        max_area_px=selection_max_area_px,
        selection_mode=str(params.get("selection_mode", "high_confidence_highest_z")),
        high_confidence_gate=float(params.get("selection_high_confidence_gate", 0.8)),
        bin_roi_polygon_uv=bin_roi_polygon_uv,
    )
    timing_log.span(
        "candidate_selection",
        candidate_select_started_at,
        candidate_found=candidate is not None,
        selection_mode=str(params.get("selection_mode", "high_confidence_highest_z")),
    )
    if candidate is None:
        if debug_logger.enabled:
            debug_logger.event(
                "run.terminal",
                error="ppf_icp_no_candidate",
                detection_count=len(detections),
            )
        return _no_candidate_result(
            params=params,
            detections=detections,
            duplicate_filter_info=duplicate_filter_info,
            selection_min_area_px=selection_min_area_px,
            selection_max_area_px=selection_max_area_px,
            timing_log=timing_log,
        )

    min_scene_points = max(16, _finite_int(params.get("ppf_scene_min_points"), 128))

    def _run_candidate(
        selected_candidate: dict[str, Any],
        *,
        stage_label: str,
    ) -> tuple[dict[str, Any], dict[str, Any], PpfPoseHypothesis | None, dict[str, Any]]:
        candidate_stage_started_at = time.perf_counter()
        candidate_data = _candidate_depth_masks(
            rgb=rgb,
            depth_m=depth_m,
            candidate=selected_candidate,
            K=np.asarray(camera_data["K"], dtype=np.float32),
            params=params,
        )
        timing_log.span(
            f"{stage_label}.mask_preparation",
            candidate_stage_started_at,
            mask_pixels=int(np.count_nonzero(candidate_data["selection_mask"])),
            contours=len(candidate_data.get("segmentation_contours_uv") or []),
        )
        cloud_started_at = time.perf_counter()
        # Feed the RAW segmentation mask (not the depth-cluster-filtered one).
        # 3D filtering inside build_scene_cloud (cluster3d -> z_band -> SOR)
        # does the actual cleanup; doing pixel-space cluster filtering on
        # the depth selection mask just throws away points that the 3D
        # filter could decide on more reliably.
        # Pass the CAD's measured extents/diameter so build_scene_cloud
        # can size its z-band / cluster thresholds to the actual object
        # rather than using one-size-fits-all defaults.
        model_xyz = np.asarray(model_bundle.model_pc[:, :3], dtype=np.float64)
        if len(model_xyz) > 0:
            extents_arr = model_xyz.max(axis=0) - model_xyz.min(axis=0)
            model_extents_tuple = (
                float(extents_arr[0]),
                float(extents_arr[1]),
                float(extents_arr[2]),
            )
            model_diameter_val = float(np.linalg.norm(extents_arr))
        else:
            model_extents_tuple = None
            model_diameter_val = None
        scene_pc, scene_info = build_scene_cloud(
            depth_m,
            np.asarray(camera_data["K"], dtype=np.float32),
            candidate_data["selection_mask"],
            params,
            model_extents=model_extents_tuple,
            model_diameter=model_diameter_val,
        )
        candidate_data["filtered_scene_pc"] = scene_pc
        timing_log.span(
            f"{stage_label}.scene_cloud",
            cloud_started_at,
            scene_points=int(len(scene_pc)),
            raw_points=int(scene_info.get("raw_point_count", 0)),
        )
        # Pop heavy stage clouds out before JSON logging (raw arrays bloat
        # pose.json/pipeline.jsonl). Save each as a separate PLY for visual
        # inspection of how the filter chain progresses.
        stage_clouds = scene_info.pop("stage_clouds", {}) or {}
        if debug_logger.enabled:
            rank_value = int(selected_candidate.get("rank", 0))
            debug_logger.event(
                f"scene_cloud[{stage_label}]",
                rank=rank_value,
                score=float(selected_candidate.get("score", 0.0)),
                bbox_xywh=[float(v) for v in (selected_candidate.get("bbox_xywh") or [])],
                mask_pixels=int(np.count_nonzero(candidate_data["selection_mask"])),
                depth_mask_pixels=int(np.count_nonzero(candidate_data["depth_selection_mask"])),
                scene_info=_safe_json(scene_info),
                scene_stats=_point_cloud_stats(scene_pc),
                stage_point_counts={k: int(len(v)) for k, v in stage_clouds.items()},
            )
            debug_logger.save_point_cloud(
                f"scene_pc_{stage_label}_rank{rank_value}", scene_pc
            )
            for stage_name, stage_pts in stage_clouds.items():
                debug_logger.save_point_cloud(
                    f"scene_stage_{stage_name}_{stage_label}_rank{rank_value}",
                    stage_pts,
                )
            debug_logger.save_image(
                f"selection_mask_{stage_label}_rank{rank_value}",
                (candidate_data["selection_mask"].astype(np.uint8) * 255),
            )
            debug_logger.save_image(
                f"depth_selection_mask_{stage_label}_rank{rank_value}",
                (candidate_data["depth_selection_mask"].astype(np.uint8) * 255),
            )
        if len(scene_pc) < min_scene_points:
            scene_info["skip_reason"] = "insufficient_scene_points"
            sel_mask = candidate_data["depth_selection_mask"].astype(bool)
            mask_pixels = int(np.count_nonzero(sel_mask))
            depth_diag: dict[str, Any] = {
                "depth_present": depth_m is not None,
                "mask_pixels": mask_pixels,
                "min_scene_points": min_scene_points,
            }
            if depth_m is not None:
                finite_depth = np.isfinite(depth_m) & (depth_m > 0.0)
                depth_diag["depth_shape"] = list(depth_m.shape)
                depth_diag["depth_dtype"] = str(depth_m.dtype)
                depth_diag["depth_total_valid_px"] = int(np.count_nonzero(finite_depth))
                if depth_m.size:
                    nonzero_vals = depth_m[finite_depth]
                    if nonzero_vals.size:
                        depth_diag["depth_min_m"] = float(nonzero_vals.min())
                        depth_diag["depth_max_m"] = float(nonzero_vals.max())
                        depth_diag["depth_median_m"] = float(np.median(nonzero_vals))
                if sel_mask.shape == depth_m.shape[:2]:
                    inside_mask = finite_depth & sel_mask
                    depth_diag["mask_pixels_with_valid_depth"] = int(
                        np.count_nonzero(inside_mask)
                    )
                    if np.any(inside_mask):
                        masked_vals = depth_m[inside_mask]
                        depth_diag["mask_depth_min_m"] = float(masked_vals.min())
                        depth_diag["mask_depth_max_m"] = float(masked_vals.max())
                        depth_diag["mask_depth_median_m"] = float(np.median(masked_vals))
                else:
                    depth_diag["mask_shape_mismatch"] = {
                        "mask_shape": list(sel_mask.shape),
                        "depth_shape": list(depth_m.shape),
                    }
            depth_diag["depth_max_m_param"] = _coerce_float(params.get("depth_max_m"), None)
            depth_diag["depth_min_m_param"] = _coerce_float(params.get("depth_min_m"), None)
            depth_diag["depth_scale_param"] = float(params.get("depth_scale", 0.001))
            scene_info["depth_diagnostics"] = depth_diag
            if debug_logger.enabled:
                debug_logger.event(
                    f"scene_cloud.insufficient[{stage_label}]",
                    rank=int(selected_candidate.get("rank", 0)),
                    score=float(selected_candidate.get("score", 0.0)),
                    scene_info=_safe_json(scene_info),
                    depth_diagnostics=_safe_json(depth_diag),
                )
            _runtime_log(
                request_id,
                "insufficient_scene_points "
                f"stage={stage_label} rank={int(selected_candidate.get('rank', 0))} "
                f"mask_px={mask_pixels} "
                f"valid_depth_in_mask={depth_diag.get('mask_pixels_with_valid_depth', 'NA')} "
                f"depth_total_valid_px={depth_diag.get('depth_total_valid_px', 'NA')} "
                f"depth_max_m_param={depth_diag.get('depth_max_m_param')}",
            )
            timing_log.span(
                f"{stage_label}.candidate_total",
                candidate_stage_started_at,
                rank=int(selected_candidate.get("rank", 0)),
                score=float(selected_candidate.get("score", 0.0)),
                pose_found=False,
            )
            return selected_candidate, candidate_data, None, scene_info
        inference_started_at = time.perf_counter()
        pose_hypothesis = _run_ppf_icp(
            bundle=model_bundle,
            scene_pc=scene_pc,
            params=params,
            debug=debug_logger,
            stage_label=stage_label,
        )
        if debug_logger.enabled and pose_hypothesis is not None:
            try:
                model_at_final = _transform_pc(
                    model_bundle.model_pc, pose_hypothesis.pose_matrix
                )
                debug_logger.save_point_cloud(
                    f"final_model_at_pose_{stage_label}", model_at_final
                )
            except Exception as exc:
                _runtime_log(request_id, f"final_model_save_failed err={exc}")
            debug_logger.event(
                f"pose.selected[{stage_label}]",
                hypothesis_index=int(pose_hypothesis.index),
                pose_matrix=pose_hypothesis.pose_matrix.tolist(),
                rotation_diag=_rotation_diagnostics(pose_hypothesis.pose_matrix),
                votes=float(pose_hypothesis.votes),
                ppf_residual=pose_hypothesis.residual,
                icp_applied=bool(pose_hypothesis.icp_applied),
                icp_residual=pose_hypothesis.icp_residual,
                icp_accepted=bool(pose_hypothesis.icp_accepted),
            )
        pose_estimation_seconds = time.perf_counter() - inference_started_at
        scene_info["pose_estimation_seconds"] = float(pose_estimation_seconds)
        timing_log.span(
            f"{stage_label}.pose_estimation",
            inference_started_at,
            pose_found=pose_hypothesis is not None,
            scene_points=int(len(scene_pc)),
        )
        timing_log.span(
            f"{stage_label}.candidate_total",
            candidate_stage_started_at,
            rank=int(selected_candidate.get("rank", 0)),
            score=float(selected_candidate.get("score", 0.0)),
            pose_found=pose_hypothesis is not None,
        )
        return selected_candidate, candidate_data, pose_hypothesis, scene_info

    candidate, candidate_data, pose_hypothesis, scene_info = _run_candidate(
        candidate,
        stage_label="primary",
    )
    if pose_hypothesis is None:
        # Distinguish between (a) the upstream depth source giving us no
        # usable depth at all (fusion failed or live cam not streaming
        # depth) and (b) a real PPF/scene problem. Surfacing a unique
        # error code for the depth-empty case stops "ppf_icp_no_pose"
        # being misattributed to the pose pipeline.
        depth_diag = (scene_info or {}).get("depth_diagnostics") or {}
        empty_depth = (
            int(depth_diag.get("depth_total_valid_px", 0) or 0) == 0
            and bool(depth_diag.get("depth_present"))
        )
        error_code = (
            "ppf_icp_depth_source_empty" if empty_depth else "ppf_icp_no_pose"
        )
        if debug_logger.enabled:
            debug_logger.event(
                "run.terminal",
                error=error_code,
                scene_info=_safe_json(scene_info),
            )
        timing_log.summary(
            valid=False,
            error=error_code,
            total_detections=len(detections),
            timing_log_path=str(timing_log.path) if timing_log.path else None,
        )
        return {
            "valid": False,
            "matches": [],
            "terminal": True,
            "error": error_code,
            "details": {
                "scene_cloud": scene_info,
                "selected_detection": {
                    "rank": int(candidate.get("rank", 0)),
                    "score": float(candidate.get("score", 0.0)),
                    "area_px": float(candidate.get("area_px", 0.0)),
                },
                "hint": (
                    "Fusion published a depth map with 0 valid pixels. "
                    "Check fusion_ready event in timeline.jsonl: if "
                    "depth_valid_px=0, the multi-view fusion produced "
                    "no usable depth (degenerate views, ROI excludes "
                    "object, hand-eye drift, or all captures had empty "
                    "depth). Inspect data/runs/<id>/fusion/cycle_*/ — "
                    "missing fused.ply / tsdf_cloud.ply confirms fusion "
                    "produced nothing."
                )
                if empty_depth
                else None,
            },
        }

    output_setup_started_at = time.perf_counter()
    debug_paths: dict[str, str] = {}
    if debug_logger.enabled and debug_logger.dir is not None:
        debug_paths["debug_dir"] = str(debug_logger.dir)
        debug_paths["debug_pipeline_log"] = str(debug_logger.dir / "pipeline.jsonl")
    visualization_3d_paths: dict[str, str | None] = {}
    payload_path = output_dir / "pose.json" if output_dir is not None else None
    object_center_object_m = np.asarray(
        annotation_meta.get("object_center_object_m") or [0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    if output_dir is not None:
        artifacts_started_at = time.perf_counter()
        debug_paths["raw_rgb"] = str(_save_bgr_image(bgr_frame, output_dir / "raw_rgb.png"))
        debug_paths["segmentation_candidates"] = str(
            _save_segmentation_candidates_overlay(
                bgr_frame,
                detections,
                depth_m=depth_m,
                destination=output_dir / "segmentation_candidates.png",
            )
        )
        try:
            pose_overlay_path = _save_pose_annotated_overlay(
                bgr_frame,
                candidate_data["selection_mask"],
                np.asarray(candidate_data["bbox_xyxy"], dtype=np.float32),
                label=assets.label,
                score=float(candidate.get("score", 0.0)),
                origin_xyz_m=np.asarray(pose_hypothesis.pose_matrix[:3, 3], dtype=np.float32),
                pose_matrix=np.asarray(pose_hypothesis.pose_matrix, dtype=np.float32),
                K=np.asarray(camera_data["K"], dtype=np.float32),
                axis_length_m=float(annotation_meta.get("annotation_axis_length_m") or 0.015),
                destination=output_dir / "final_pose.png",
            )
            debug_paths["final_pose"] = str(pose_overlay_path)
            debug_paths["pose_annotated"] = str(pose_overlay_path)
        except Exception as exc:
            _runtime_log(request_id, f"pose_overlay_save_failed error={exc}")
        if bool(params.get("save_pose_3d_assets", True)):
            try:
                filtered_scene_pc = candidate_data.get("filtered_scene_pc")
                segmented_override = None
                if filtered_scene_pc is not None and len(filtered_scene_pc) > 0:
                    segmented_override = np.ascontiguousarray(
                        np.asarray(filtered_scene_pc, dtype=np.float32)[:, :3]
                    )
                visualization_3d_paths = save_pose_3d_assets(
                    rgb=rgb,
                    depth_m=depth_m,
                    K=np.asarray(camera_data["K"], dtype=np.float32),
                    mask=candidate_data["selection_mask"],
                    mesh_path=assets.mesh_path,
                    pose_matrix=np.asarray(pose_hypothesis.pose_matrix, dtype=np.float32),
                    mesh_units=str(params.get("mesh_units", "mm")),
                    mesh_scale=float(params.get("mesh_scale", 1.0)),
                    object_center_object_m=object_center_object_m,
                    scene_point_cloud_ply_path=output_dir / "scene_point_cloud.ply",
                    segmented_point_cloud_ply_path=output_dir / "segmented_point_cloud.ply",
                    pose_scene_glb_path=output_dir / "pose_scene.glb",
                    max_depth_m=_coerce_float(params.get("depth_max_m"), None),
                    segmented_points_override=segmented_override,
                )
                for key, value in visualization_3d_paths.items():
                    if value:
                        debug_paths[key] = str(value)
            except Exception as exc:
                _runtime_log(request_id, f"pose_3d_asset_save_failed error={exc}")
        timing_log.span(
            "debug_artifacts",
            artifacts_started_at,
            output_dir=str(output_dir),
        )
    timing_log.span(
        "output_setup",
        output_setup_started_at,
        save_outputs=output_dir is not None,
    )

    match = _make_match(
        params=params,
        assets=assets,
        candidate=candidate,
        candidate_data=candidate_data,
        pose_hypothesis=pose_hypothesis,
        camera_data=camera_data,
        annotation_meta=annotation_meta,
        duplicate_filter_info=duplicate_filter_info,
        scene_info=scene_info,
        model_bundle=model_bundle,
        depth_m=depth_m,
        request_id=request_id,
        debug_paths=debug_paths,
        visualization_3d_paths=visualization_3d_paths,
    )
    matches_out = [match]

    if bool(params.get("all_poses_mode", False)):
        all_poses_started_at = time.perf_counter()
        try:
            min_conf = float(params.get("selection_min_confidence", 0.2))
            primary_rank = int(candidate.get("rank", -1))
            eligible_extra = [
                d
                for d in detections
                if int(d.get("rank", -1)) != primary_rank
                and float(d.get("score", 0.0)) >= min_conf
                and float(d.get("area_px", 0.0)) >= selection_min_area_px
                and (
                    selection_max_area_px is None
                    or float(d.get("area_px", 0.0)) <= selection_max_area_px
                )
                and _detection_passes_bin_roi(d, bin_roi_polygon_uv)
            ]
            requested_total = max(1, int(params.get("all_poses_max", 20) or 20))
            cap = max(0, requested_total - 1)
            if debug_logger.enabled:
                debug_logger.event(
                    "all_poses.candidates",
                    detection_count=len(detections),
                    eligible_count=len(eligible_extra),
                    requested_total=requested_total,
                    extra_cap=cap,
                    primary_rank=primary_rank,
                    eligible_ranks=[
                        int(item.get("rank", -1)) for item in eligible_extra
                    ],
                )
            eligible_extra = eligible_extra[:cap]
            for extra_candidate in eligible_extra:
                try:
                    extra_cand, extra_data, extra_pose, extra_scene = _run_candidate(
                        extra_candidate,
                        stage_label=f"all_poses_rank{int(extra_candidate.get('rank', -1))}",
                    )
                except Exception as exc:
                    _runtime_log(
                        request_id,
                        f"all_poses_candidate_failed rank={int(extra_candidate.get('rank', -1))} err={exc}",
                    )
                    continue
                if extra_pose is None:
                    continue
                matches_out.append(
                    _make_match(
                        params=params,
                        assets=assets,
                        candidate=extra_cand,
                        candidate_data=extra_data,
                        pose_hypothesis=extra_pose,
                        camera_data=camera_data,
                        annotation_meta=annotation_meta,
                        duplicate_filter_info=duplicate_filter_info,
                        scene_info=extra_scene,
                        model_bundle=model_bundle,
                        depth_m=depth_m,
                        request_id=request_id,
                        debug_paths=debug_paths,
                        visualization_3d_paths=visualization_3d_paths,
                        all_poses_match=True,
                    )
                )
            if (
                output_dir is not None
                and bool(params.get("save_pose_3d_assets", True))
                and bool(params.get("render_all_poses_3d", True))
                and len(matches_out) > 1
            ):
                pose_matrices = [
                    np.asarray(item.get("pose_matrix"), dtype=np.float64)
                    for item in matches_out
                    if isinstance(item, dict) and item.get("pose_matrix") is not None
                ]
                if len(pose_matrices) > 1:
                    all_pose_scene_path = output_dir / "pose_scene_all_poses.glb"
                    save_multi_pose_scene_glb(
                        mesh_path=assets.mesh_path,
                        pose_matrices=pose_matrices,
                        mesh_units=str(params.get("mesh_units", "mm")),
                        mesh_scale=float(params.get("mesh_scale", 1.0)),
                        destination=all_pose_scene_path,
                        object_center_object_m=object_center_object_m,
                    )
                    visualization_3d_paths["pose_scene_glb_path"] = str(all_pose_scene_path)
                    visualization_3d_paths["pose_scene_all_poses_glb_path"] = str(all_pose_scene_path)
                    debug_paths["pose_scene_glb_path"] = str(all_pose_scene_path)
                    debug_paths["pose_scene_all_poses_glb_path"] = str(all_pose_scene_path)
        except Exception as exc:
            _runtime_log(request_id, f"all_poses_mode_failed err={exc}")
        timing_log.span("all_poses_mode", all_poses_started_at, total_matches=len(matches_out))

    result = {
        "valid": True,
        "matches": matches_out,
        "match_count": int(len(matches_out)),
        "all_poses_mode": bool(params.get("all_poses_mode", False)),
        "model_name": "ppf_icp",
        "object_folder": str(assets.object_folder),
        "camera_calibration_source": match["camera_calibration_source"],
        "selected_detection": {
            "rank": int(candidate.get("rank", 0)),
            "score": float(candidate.get("score", 0.0)),
            "area_px": float(candidate.get("area_px", 0.0)),
            "bbox_xywh": [float(v) for v in candidate_data.get("bbox_xywh", [])],
        },
        "segmentation_duplicate_filter": duplicate_filter_info,
        "timing": {
            "inference_seconds": float(scene_info.get("pose_estimation_seconds", 0.0) or 0.0),
            "total_detections": len(detections),
            "timing_log_path": str(timing_log.path) if timing_log.enabled and timing_log.path else None,
        },
        "debug_paths": debug_paths,
        "visualization": {},
        "visualization_3d": visualization_3d_paths,
        "request_id": request_id,
    }
    if payload_path is not None:
        payload_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
        result["pose_json_path"] = str(payload_path)
    if debug_logger.enabled:
        debug_logger.event(
            "run.done",
            match_count=int(len(matches_out)),
            primary_pose_matrix=match.get("pose_matrix"),
            primary_rotation_diag=_rotation_diagnostics(np.asarray(match.get("pose_matrix"), dtype=np.float64)),
            total_dt_s=float(time.perf_counter() - run_started_at),
        )
    timing_log.summary(
        valid=True,
        match_count=int(len(matches_out)),
        total_detections=len(detections),
        timing_log_path=str(timing_log.path) if timing_log.path else None,
    )
    _runtime_log(
        request_id,
        "done "
        f"score={float(candidate.get('score', 0.0)):.4f} "
        f"center_xyz_m={json.dumps(match['center_xyz_m'])} "
        f"matches={len(matches_out)} "
        f"total_dt={time.perf_counter() - run_started_at:.3f}s",
    )
    return result
