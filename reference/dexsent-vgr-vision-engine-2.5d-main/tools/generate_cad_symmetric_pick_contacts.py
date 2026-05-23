from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_engine.modules.megapose_bin_picking.runtime import (  # noqa: E402
    resolve_object_assets,
    resolve_workspace_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate symmetric pick contacts on a CAD mesh by selecting one or "
            "more seed points. The tool auto-detects the dominant rotational "
            "symmetry axis and revolves each seed into a circular contact ring."
        )
    )
    parser.add_argument(
        "--object-folder",
        type=str,
        default=None,
        help="MegaPose object folder containing the object mesh.",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Mesh path (.stl/.obj/.glb) when not using --object-folder.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Object label stored in the output payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <object-folder>/pick_contacts.json.",
    )
    parser.add_argument(
        "--mesh-units",
        choices=["mm", "m"],
        default="mm",
        help="Units of the source CAD before scaling.",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Extra scale applied on top of mesh-units conversion.",
    )
    parser.add_argument(
        "--surface-sample-count",
        type=int,
        default=120000,
        help="Surface samples used for picking and projection.",
    )
    parser.add_argument(
        "--contact-count",
        type=int,
        default=24,
        help="Number of contacts generated per selected seed point.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Open3D point size for the sampled mesh cloud.",
    )
    parser.add_argument(
        "--tool-axis",
        type=str,
        default="-z",
        help="Tool axis that should align with the surface normal in VGR.",
    )
    parser.add_argument(
        "--force-yaw-deg",
        type=float,
        default=0.0,
        help="Yaw VGR should keep while roll/pitch comes from the chosen normal.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="upward_then_camera",
        help="Selection strategy for VGR runtime.",
    )
    parser.add_argument(
        "--axis",
        choices=["auto", "x", "y", "z"],
        default="auto",
        help="Use a fixed symmetry axis, or auto-detect it from the mesh.",
    )
    parser.add_argument(
        "--sweep-deg",
        type=float,
        default=360.0,
        help="Angular sweep for each generated ring. Default is a full ring.",
    )
    return parser.parse_args()


def resolve_inputs(
    args: argparse.Namespace,
) -> tuple[Optional[Path], Path, Path, str]:
    if args.object_folder:
        object_folder = resolve_workspace_path(args.object_folder)
        assets = resolve_object_assets(
            object_folder,
            label_override=args.label,
        )
        label = args.label or assets.label or object_folder.name
        output_path = args.output or (object_folder / "pick_contacts.json")
        return object_folder, assets.mesh_path, output_path, str(label)

    if not args.mesh:
        raise ValueError("Provide either --object-folder or --mesh.")

    mesh_path = resolve_workspace_path(args.mesh)
    label = args.label or mesh_path.stem
    output_path = args.output or (mesh_path.parent / "pick_contacts.json")
    return None, mesh_path, output_path, str(label)


def load_centered_mesh(
    mesh_path: Path,
    mesh_units: str,
    mesh_scale: float,
) -> tuple[object, np.ndarray, float]:
    import trimesh

    mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"No geometry found in {mesh_path}")
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)!r}")
    mesh = mesh.copy()
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    center_mesh_units = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
    mesh.apply_translation(-center_mesh_units)
    scale_to_m = (1.0 if mesh_units == "m" else 0.001) * float(mesh_scale)
    mesh.apply_scale(scale_to_m)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh, np.asarray(center_mesh_units, dtype=np.float64), float(scale_to_m)


def sample_surface_points(
    mesh: object,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    points_m, face_indices = trimesh.sample.sample_surface(mesh, int(sample_count))
    face_indices = np.asarray(face_indices, dtype=np.int32)
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float64)
    norm = np.linalg.norm(normals, axis=1)
    valid = norm > 1e-9
    if not np.any(valid):
        raise ValueError("No valid face normals found on the sampled surface.")
    points_m = np.asarray(points_m, dtype=np.float64)[valid]
    face_indices = face_indices[valid]
    normals = normals[valid] / norm[valid][:, None]
    return points_m, normals, face_indices


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        raise ValueError("Degenerate vector encountered.")
    return arr / norm


def _canonicalize_axis(axis: np.ndarray) -> np.ndarray:
    axis = _normalize(axis)
    dominant_idx = int(np.argmax(np.abs(axis)))
    if float(axis[dominant_idx]) < 0.0:
        axis = -axis
    return axis


def run_seed_picker(
    points_m: np.ndarray,
    normals: np.ndarray,
    point_size: float,
) -> list[int]:
    import open3d as o3d

    colors = np.clip(0.5 * (normals + 1.0), 0.0, 1.0)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_m)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    extent = float(np.linalg.norm(points_m.max(axis=0) - points_m.min(axis=0)))
    axis_size = max(0.03, extent * 0.18)
    center_radius = max(0.004, extent * 0.02)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    origin = o3d.geometry.TriangleMesh.create_sphere(radius=center_radius)
    origin.paint_uniform_color([0.85, 0.2, 0.2])

    print("Open3D symmetry picker controls:")
    print("  Shift + left click : pick one or more seed points")
    print("  Shift + right click: undo the last pick")
    print("  Q or Escape        : finish selection")
    print("Pick seed points on the cylindrical/symmetric band(s) you want to replicate.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="CAD Symmetric Pick Contacts",
        width=1600,
        height=1000,
        visible=True,
    )
    vis.add_geometry(point_cloud)
    vis.add_geometry(frame)
    vis.add_geometry(origin)
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.96, 0.97, 0.98], dtype=np.float64)
    render_option.point_size = float(point_size)
    vis.run()
    picked = [int(idx) for idx in vis.get_picked_points()]
    vis.destroy_window()
    picked = list(dict.fromkeys(picked))
    if not picked:
        raise ValueError("Pick at least one seed point.")
    return picked


def _score_symmetry_axis(
    surface_points_m: np.ndarray,
    axis_origin_m: np.ndarray,
    axis_dir: np.ndarray,
) -> dict[str, float]:
    centered = np.asarray(surface_points_m, dtype=np.float64) - axis_origin_m[None, :]
    axial = centered @ axis_dir
    radial_vec = centered - np.outer(axial, axis_dir)
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    radial_median = float(np.median(radial_dist))
    radial_mad = float(np.median(np.abs(radial_dist - radial_median)))
    axial_lo, axial_hi = np.quantile(axial, [0.025, 0.975])
    axial_span = float(axial_hi - axial_lo)
    radial_lo, radial_hi = np.quantile(radial_dist, [0.10, 0.90])
    radial_span = float(radial_hi - radial_lo)
    score = axial_span / max(radial_mad + 0.20 * radial_span + 0.05 * radial_median, 1e-6)
    return {
        "score": float(score),
        "axial_span_m": axial_span,
        "radial_median_m": radial_median,
        "radial_mad_m": radial_mad,
        "radial_span_m": radial_span,
    }


def _score_ring_fit_error(
    surface_points_m: np.ndarray,
    seed_points_m: np.ndarray,
    axis_origin_m: np.ndarray,
    axis_dir: np.ndarray,
    sweep_deg: float,
    eval_contact_count: int = 24,
) -> dict[str, float]:
    min_distances = []
    for seed_point_m in np.asarray(seed_points_m, dtype=np.float64):
        try:
            ring_points_m = generate_ring_candidates(
                seed_point_m=seed_point_m,
                axis_origin_m=axis_origin_m,
                axis_dir=axis_dir,
                contact_count=max(int(eval_contact_count), 8),
                sweep_deg=sweep_deg,
            )
        except ValueError:
            return {
                "ring_fit_mean_distance_m": float("inf"),
                "ring_fit_median_distance_m": float("inf"),
                "ring_fit_max_distance_m": float("inf"),
            }
        for point in ring_points_m:
            diff = surface_points_m - point[None, :]
            dist2 = np.einsum("ij,ij->i", diff, diff)
            min_distances.append(math.sqrt(float(np.min(dist2))))

    if not min_distances:
        return {
            "ring_fit_mean_distance_m": float("inf"),
            "ring_fit_median_distance_m": float("inf"),
            "ring_fit_max_distance_m": float("inf"),
        }

    distances = np.asarray(min_distances, dtype=np.float64)
    return {
        "ring_fit_mean_distance_m": float(np.mean(distances)),
        "ring_fit_median_distance_m": float(np.median(distances)),
        "ring_fit_max_distance_m": float(np.max(distances)),
    }


def detect_symmetry_axis(
    surface_points_m: np.ndarray,
    seed_points_m: np.ndarray,
    axis_mode: str,
    sweep_deg: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    axis_origin_m = np.asarray(surface_points_m, dtype=np.float64).mean(axis=0)
    axis_mode = str(axis_mode or "auto").strip().lower()

    if axis_mode in {"x", "y", "z"}:
        lookup = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        axis_dir = lookup[axis_mode]
        stats = _score_symmetry_axis(surface_points_m, axis_origin_m, axis_dir)
        stats.update(
            _score_ring_fit_error(
                surface_points_m=surface_points_m,
                seed_points_m=seed_points_m,
                axis_origin_m=axis_origin_m,
                axis_dir=axis_dir,
                sweep_deg=sweep_deg,
            )
        )
        stats["selected_candidate"] = axis_mode
        stats["candidates"] = [
            {
                "name": axis_mode,
                "direction_unit": [float(v) for v in axis_dir.tolist()],
                **stats,
            }
        ]
        return axis_origin_m, axis_dir, stats

    centered = np.asarray(surface_points_m, dtype=np.float64) - axis_origin_m[None, :]
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    raw_candidates: list[tuple[str, np.ndarray]] = []
    for rank, eig_idx in enumerate(order, start=1):
        raw_candidates.append(
            (f"mesh_pca_{rank}", _canonicalize_axis(eigvecs[:, eig_idx]))
        )
    raw_candidates.extend(
        [
            ("axis_x", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
            ("axis_y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
            ("axis_z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
        ]
    )
    if len(seed_points_m) >= 2:
        seed_center = np.mean(seed_points_m, axis=0, keepdims=True)
        seed_centered = np.asarray(seed_points_m, dtype=np.float64) - seed_center
        seed_cov = np.cov(seed_centered, rowvar=False)
        seed_eigvals, seed_eigvecs = np.linalg.eigh(seed_cov)
        seed_order = np.argsort(seed_eigvals)[::-1]
        raw_candidates.append(
            ("seed_pca_1", _canonicalize_axis(seed_eigvecs[:, seed_order[0]]))
        )

    deduped_candidates: list[tuple[str, np.ndarray]] = []
    for name, axis_dir in raw_candidates:
        axis_dir = _canonicalize_axis(axis_dir)
        if any(abs(float(np.dot(axis_dir, prev_axis))) >= 0.98 for _, prev_axis in deduped_candidates):
            continue
        deduped_candidates.append((name, axis_dir))

    candidates = []
    for name, axis_dir in deduped_candidates:
        axis_stats = _score_symmetry_axis(surface_points_m, axis_origin_m, axis_dir)
        axis_stats.update(
            _score_ring_fit_error(
                surface_points_m=surface_points_m,
                seed_points_m=seed_points_m,
                axis_origin_m=axis_origin_m,
                axis_dir=axis_dir,
                sweep_deg=sweep_deg,
            )
        )
        axis_stats["name"] = name
        axis_stats["direction_unit"] = [float(v) for v in axis_dir.tolist()]
        axis_stats["_axis_dir"] = axis_dir
        candidates.append(axis_stats)

    if not candidates:
        raise ValueError("Failed to evaluate symmetry-axis candidates.")

    best_ring_fit = min(float(item["ring_fit_mean_distance_m"]) for item in candidates)
    tie_tolerance = max(2.5e-4, best_ring_fit * 0.25)
    shortlisted = [
        item
        for item in candidates
        if float(item["ring_fit_mean_distance_m"]) <= best_ring_fit + tie_tolerance
    ]
    best_stats = max(
        shortlisted,
        key=lambda item: (
            float(item["score"]),
            -float(item["ring_fit_mean_distance_m"]),
            -float(item["ring_fit_max_distance_m"]),
        ),
    )
    best_axis = np.asarray(best_stats["_axis_dir"], dtype=np.float64)

    if best_axis is None or best_stats is None:
        raise ValueError("Failed to detect a symmetry axis from the sampled mesh.")

    stats_out = {k: v for k, v in best_stats.items() if k != "_axis_dir"}
    stats_out["selected_candidate"] = str(best_stats["name"])
    stats_out["candidates"] = [
        {k: v for k, v in item.items() if k != "_axis_dir"}
        for item in candidates
    ]
    return axis_origin_m, best_axis, stats_out


def rotate_about_axis(vec: np.ndarray, axis_dir: np.ndarray, angle_rad: float) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    axis_dir = _normalize(axis_dir)
    cos_a = math.cos(float(angle_rad))
    sin_a = math.sin(float(angle_rad))
    return (
        vec * cos_a
        + np.cross(axis_dir, vec) * sin_a
        + axis_dir * float(np.dot(axis_dir, vec)) * (1.0 - cos_a)
    )


def generate_ring_candidates(
    seed_point_m: np.ndarray,
    axis_origin_m: np.ndarray,
    axis_dir: np.ndarray,
    contact_count: int,
    sweep_deg: float,
) -> np.ndarray:
    if contact_count < 3:
        raise ValueError("Symmetry ring generation requires at least 3 contact points per ring.")
    seed_point_m = np.asarray(seed_point_m, dtype=np.float64)
    axis_origin_m = np.asarray(axis_origin_m, dtype=np.float64)
    axis_dir = _normalize(axis_dir)

    axial_coord = float(np.dot(seed_point_m - axis_origin_m, axis_dir))
    axis_point_m = axis_origin_m + axis_dir * axial_coord
    radial_vec = seed_point_m - axis_point_m
    radius_m = float(np.linalg.norm(radial_vec))
    if radius_m <= 1e-6:
        raise ValueError("Selected seed lies too close to the detected symmetry axis.")

    sweep_rad = math.radians(float(sweep_deg))
    full_circle = abs(abs(float(sweep_deg)) - 360.0) <= 1e-6
    angles = np.linspace(
        0.0,
        sweep_rad,
        int(contact_count),
        endpoint=not full_circle,
        dtype=np.float64,
    )
    candidates = []
    for angle in angles:
        point = axis_point_m + rotate_about_axis(radial_vec, axis_dir, float(angle))
        candidates.append(point)
    return np.asarray(candidates, dtype=np.float64)


def project_candidates_to_surface(
    candidate_points_m: np.ndarray,
    surface_points_m: np.ndarray,
    surface_normals: np.ndarray,
    face_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    projected_points = []
    projected_normals = []
    projected_faces = []
    projected_sample_indices = []
    seen_samples: set[int] = set()

    for point in np.asarray(candidate_points_m, dtype=np.float64):
        diff = surface_points_m - point[None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        sample_idx = int(np.argmin(dist2))
        if sample_idx in seen_samples:
            continue
        seen_samples.add(sample_idx)
        projected_points.append(surface_points_m[sample_idx])
        projected_normals.append(surface_normals[sample_idx])
        projected_faces.append(int(face_indices[sample_idx]))
        projected_sample_indices.append(sample_idx)

    if not projected_points:
        raise ValueError("No contacts could be projected onto the sampled surface.")

    return (
        np.asarray(projected_points, dtype=np.float64),
        np.asarray(projected_normals, dtype=np.float64),
        np.asarray(projected_faces, dtype=np.int32),
        np.asarray(projected_sample_indices, dtype=np.int32),
    )


def build_payload(
    *,
    object_folder: Optional[Path],
    mesh_path: Path,
    label: str,
    mesh_units: str,
    mesh_scale: float,
    scale_to_m: float,
    center_mesh_units: np.ndarray,
    points_m: np.ndarray,
    normals: np.ndarray,
    face_indices: np.ndarray,
    sample_indices: np.ndarray,
    tool_axis: str,
    force_yaw_deg: float,
    selection_mode: str,
    seed_indices: list[int],
    seed_contact_groups: list[list[int]],
    symmetry_axis_origin_m: np.ndarray,
    symmetry_axis_direction_unit: np.ndarray,
    symmetry_stats: dict[str, Any],
    contact_count: int,
    sweep_deg: float,
) -> dict:
    contacts = []
    for order, (point_local_m, normal_local, face_index, sample_index) in enumerate(
        zip(points_m, normals, face_indices, sample_indices),
        start=1,
    ):
        point_local_mesh_units = point_local_m / float(scale_to_m)
        contacts.append(
            {
                "id": f"contact_{order:03d}",
                "label": f"contact_{order:03d}",
                "point_local_m": [float(v) for v in point_local_m.tolist()],
                "point_local_mesh_units": [
                    float(v) for v in point_local_mesh_units.tolist()
                ],
                "normal_local": [float(v) for v in normal_local.tolist()],
                "face_index": int(face_index),
                "sample_index": int(sample_index),
                "enabled": True,
            }
        )

    generator = {
        "type": "symmetry_revolve",
        "seed_sample_indices": [int(v) for v in seed_indices],
        "seed_projected_contact_sample_index_groups": [
            [int(v) for v in group]
            for group in seed_contact_groups
        ],
        "ring_count": int(len(seed_indices)),
        "contacts_per_ring_requested": int(contact_count),
        "sweep_deg": float(sweep_deg),
        "symmetry_axis_origin_m": [float(v) for v in symmetry_axis_origin_m.tolist()],
        "symmetry_axis_direction_unit": [
            float(v) for v in symmetry_axis_direction_unit.tolist()
        ],
        "symmetry_detection": symmetry_stats,
    }

    return {
        "format": "vgr_pick_contacts/v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "object_label": label,
        "object_folder": str(object_folder) if object_folder else None,
        "source_mesh_path": str(mesh_path),
        "origin_mode": "mesh_aabb_center",
        "origin_adjustment_mesh_units": [float(v) for v in center_mesh_units.tolist()],
        "mesh_units": mesh_units,
        "mesh_scale": float(mesh_scale),
        "selection_mode": selection_mode,
        "tool_axis": str(tool_axis).strip().lower() or "-z",
        "force_yaw_deg": float(force_yaw_deg),
        "generator": generator,
        "contacts": contacts,
    }


def preview_generated_contacts(
    surface_points_m: np.ndarray,
    surface_normals: np.ndarray,
    seed_points_m: np.ndarray,
    contact_points_m: np.ndarray,
    contact_normals: np.ndarray,
    axis_origin_m: np.ndarray,
    axis_dir: np.ndarray,
    point_size: float,
) -> None:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(surface_points_m)
    cloud.normals = o3d.utility.Vector3dVector(surface_normals)
    cloud.paint_uniform_color([0.72, 0.78, 0.84])

    contact_cloud = o3d.geometry.PointCloud()
    contact_cloud.points = o3d.utility.Vector3dVector(contact_points_m)
    contact_cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray([[0.85, 0.15, 0.15]], dtype=np.float64), (len(contact_points_m), 1))
    )

    seed_cloud = o3d.geometry.PointCloud()
    seed_cloud.points = o3d.utility.Vector3dVector(seed_points_m)
    seed_cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray([[0.15, 0.55, 0.15]], dtype=np.float64), (len(seed_points_m), 1))
    )

    line_points = []
    line_indices = []
    line_colors = []
    normal_length = max(
        0.01,
        0.06 * float(np.linalg.norm(surface_points_m.max(axis=0) - surface_points_m.min(axis=0))),
    )
    for point, normal in zip(contact_points_m, contact_normals):
        start_idx = len(line_points)
        line_points.append(point.tolist())
        line_points.append((point + normal * normal_length).tolist())
        line_indices.append([start_idx, start_idx + 1])
        line_colors.append([0.95, 0.25, 0.25])
    normal_lines = o3d.geometry.LineSet()
    normal_lines.points = o3d.utility.Vector3dVector(np.asarray(line_points, dtype=np.float64))
    normal_lines.lines = o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32))
    normal_lines.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))

    axial = (surface_points_m - axis_origin_m[None, :]) @ axis_dir
    axis_pad = 0.05 * max(float(axial.max() - axial.min()), 0.04)
    axis_start = axis_origin_m + axis_dir * (float(axial.min()) - axis_pad)
    axis_end = axis_origin_m + axis_dir * (float(axial.max()) + axis_pad)
    axis_line = o3d.geometry.LineSet()
    axis_line.points = o3d.utility.Vector3dVector(
        np.asarray([axis_start, axis_end], dtype=np.float64)
    )
    axis_line.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    axis_line.colors = o3d.utility.Vector3dVector(
        np.asarray([[0.20, 0.40, 0.95]], dtype=np.float64)
    )

    extent = float(np.linalg.norm(surface_points_m.max(axis=0) - surface_points_m.min(axis=0)))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.03, extent * 0.18))

    print("Preview window:")
    print("  gray  : sampled mesh surface")
    print("  green : picked seed points")
    print("  blue  : detected symmetry axis")
    print("  red   : generated grasp contacts and normals")
    print("  Q     : close preview")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Generated Symmetric CAD Contacts", width=1600, height=1000)
    vis.add_geometry(cloud)
    vis.add_geometry(contact_cloud)
    vis.add_geometry(seed_cloud)
    vis.add_geometry(axis_line)
    vis.add_geometry(normal_lines)
    vis.add_geometry(frame)
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.98, 0.98, 0.99], dtype=np.float64)
    render_option.point_size = float(point_size)
    vis.run()
    vis.destroy_window()


def main() -> None:
    args = parse_args()
    object_folder, mesh_path, output_path, label = resolve_inputs(args)
    mesh, center_mesh_units, scale_to_m = load_centered_mesh(
        mesh_path=mesh_path,
        mesh_units=args.mesh_units,
        mesh_scale=args.mesh_scale,
    )
    surface_points_m, surface_normals, surface_faces = sample_surface_points(
        mesh,
        args.surface_sample_count,
    )
    seed_indices = run_seed_picker(
        surface_points_m,
        surface_normals,
        args.point_size,
    )
    seed_points_m = surface_points_m[np.asarray(seed_indices, dtype=np.int32)]

    symmetry_axis_origin_m, symmetry_axis_dir, symmetry_stats = detect_symmetry_axis(
        surface_points_m,
        seed_points_m,
        args.axis,
        args.sweep_deg,
    )
    print("Detected symmetry axis:")
    print(
        "  origin_m = "
        f"{[round(float(v), 6) for v in symmetry_axis_origin_m.tolist()]}"
    )
    print(
        "  direction = "
        f"{[round(float(v), 6) for v in symmetry_axis_dir.tolist()]}"
    )
    print(
        "  score = "
        f"{float(symmetry_stats.get('score', 0.0)):.3f} "
        f"({symmetry_stats.get('selected_candidate', 'auto')})"
    )
    print(
        "  ring_fit_mean_distance_m = "
        f"{float(symmetry_stats.get('ring_fit_mean_distance_m', float('nan'))):.6f}"
    )

    all_contact_points: list[np.ndarray] = []
    all_contact_normals: list[np.ndarray] = []
    all_contact_faces: list[np.ndarray] = []
    all_contact_sample_indices: list[np.ndarray] = []
    seed_contact_groups: list[list[int]] = []
    seen_samples: set[int] = set()

    for seed_idx, seed_point_m in zip(seed_indices, seed_points_m):
        candidate_points_m = generate_ring_candidates(
            seed_point_m=seed_point_m,
            axis_origin_m=symmetry_axis_origin_m,
            axis_dir=symmetry_axis_dir,
            contact_count=args.contact_count,
            sweep_deg=args.sweep_deg,
        )
        (
            ring_contact_points_m,
            ring_contact_normals,
            ring_contact_faces,
            ring_contact_sample_indices,
        ) = project_candidates_to_surface(
            candidate_points_m,
            surface_points_m,
            surface_normals,
            surface_faces,
        )
        keep_mask = np.asarray(
            [int(sample_idx) not in seen_samples for sample_idx in ring_contact_sample_indices],
            dtype=bool,
        )
        kept_sample_indices = ring_contact_sample_indices[keep_mask]
        for sample_idx in kept_sample_indices.tolist():
            seen_samples.add(int(sample_idx))
        if np.any(keep_mask):
            all_contact_points.append(ring_contact_points_m[keep_mask])
            all_contact_normals.append(ring_contact_normals[keep_mask])
            all_contact_faces.append(ring_contact_faces[keep_mask])
            all_contact_sample_indices.append(kept_sample_indices)
        seed_contact_groups.append([int(v) for v in kept_sample_indices.tolist()])
        print(
            f"Seed sample #{int(seed_idx)} -> generated "
            f"{len(ring_contact_sample_indices)} contacts "
            f"({len(kept_sample_indices)} new after dedupe)."
        )

    if not all_contact_points:
        raise ValueError("All generated rings collapsed to duplicate surface samples.")

    contact_points_m = np.vstack(all_contact_points)
    contact_normals = np.vstack(all_contact_normals)
    contact_faces = np.concatenate(all_contact_faces)
    contact_sample_indices = np.concatenate(all_contact_sample_indices)

    preview_generated_contacts(
        surface_points_m=surface_points_m,
        surface_normals=surface_normals,
        seed_points_m=seed_points_m,
        contact_points_m=contact_points_m,
        contact_normals=contact_normals,
        axis_origin_m=symmetry_axis_origin_m,
        axis_dir=symmetry_axis_dir,
        point_size=args.point_size,
    )

    payload = build_payload(
        object_folder=object_folder,
        mesh_path=mesh_path,
        label=label,
        mesh_units=args.mesh_units,
        mesh_scale=args.mesh_scale,
        scale_to_m=scale_to_m,
        center_mesh_units=center_mesh_units,
        points_m=contact_points_m,
        normals=contact_normals,
        face_indices=contact_faces,
        sample_indices=contact_sample_indices,
        tool_axis=args.tool_axis,
        force_yaw_deg=args.force_yaw_deg,
        selection_mode=args.selection_mode,
        seed_indices=seed_indices,
        seed_contact_groups=seed_contact_groups,
        symmetry_axis_origin_m=symmetry_axis_origin_m,
        symmetry_axis_direction_unit=symmetry_axis_dir,
        symmetry_stats=symmetry_stats,
        contact_count=args.contact_count,
        sweep_deg=args.sweep_deg,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved {len(payload['contacts'])} grasp contacts to {output_path}")
    print(f"Mesh: {mesh_path}")
    print("This file is directly usable by VGR bin picking as pick_contacts.json.")


if __name__ == "__main__":
    main()
