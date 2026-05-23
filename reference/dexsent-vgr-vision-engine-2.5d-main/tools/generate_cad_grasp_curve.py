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
            "Generate grasp contacts on a CAD mesh by defining either a surface "
            "line or a circular band. The output is compatible with VGR "
            "pick_contacts.json."
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
        "--mode",
        choices=["line", "circle"],
        required=True,
        help="Curve type to generate on the mesh surface.",
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
        help="Number of grasp contacts to generate on the curve.",
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
        help="Selection strategy for VGR runtime. Default prefers upward normals, then camera proximity.",
    )
    parser.add_argument(
        "--sweep-deg",
        type=float,
        default=360.0,
        help="Angular sweep for circle mode. Default is a full ring.",
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
        raise ValueError("Degenerate vector encountered while building grasp curve.")
    return arr / norm


def run_anchor_picker(
    points_m: np.ndarray,
    normals: np.ndarray,
    point_size: float,
    mode: str,
) -> list[int]:
    import open3d as o3d

    required_picks = 2 if mode == "line" else 3
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

    print("Open3D curve picker controls:")
    print("  Shift + left click : pick anchor points on the mesh surface")
    print("  Shift + right click: undo the last pick")
    print("  Q or Escape        : finish selection")
    if mode == "line":
        print("Pick exactly 2 anchors along the desired grasp line.")
    else:
        print("Pick exactly 3 anchors on the desired circular band.")
        print("After each circle closes, type N in the terminal to draw the next circle.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name=f"CAD Grasp Curve ({mode})",
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
    if len(picked) != required_picks:
        raise ValueError(
            f"{mode} mode requires exactly {required_picks} picked anchors, got {len(picked)}."
        )
    return picked


def prompt_continue_circle_selection() -> bool:
    print("Circle completed.")
    print("Press N then Enter to draw another circle, or just Enter to finish.")
    response = input("> ").strip().lower()
    return response == "n"


def generate_line_candidates(anchor_points_m: np.ndarray, contact_count: int) -> np.ndarray:
    if anchor_points_m.shape != (2, 3):
        raise ValueError("Line mode requires exactly 2 anchor points.")
    if contact_count < 2:
        raise ValueError("Line mode requires at least 2 contact points.")
    weights = np.linspace(0.0, 1.0, int(contact_count), endpoint=True, dtype=np.float64)
    start = anchor_points_m[0]
    end = anchor_points_m[1]
    return start[None, :] * (1.0 - weights[:, None]) + end[None, :] * weights[:, None]


def _fit_circle(anchor_points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if anchor_points_m.shape != (3, 3):
        raise ValueError("Circle mode requires exactly 3 anchor points.")

    p0, p1, p2 = [np.asarray(p, dtype=np.float64) for p in anchor_points_m]
    axis_u = _normalize(p1 - p0)
    plane_normal = _normalize(np.cross(p1 - p0, p2 - p0))
    axis_v = _normalize(np.cross(plane_normal, axis_u))

    p0_2d = np.array([0.0, 0.0], dtype=np.float64)
    p1_2d = np.array([np.linalg.norm(p1 - p0), 0.0], dtype=np.float64)
    rel_2 = p2 - p0
    p2_2d = np.array(
        [float(np.dot(rel_2, axis_u)), float(np.dot(rel_2, axis_v))],
        dtype=np.float64,
    )

    matrix = 2.0 * np.array(
        [
            [p1_2d[0] - p0_2d[0], p1_2d[1] - p0_2d[1]],
            [p2_2d[0] - p0_2d[0], p2_2d[1] - p0_2d[1]],
        ],
        dtype=np.float64,
    )
    rhs = np.array(
        [
            p1_2d[0] ** 2 + p1_2d[1] ** 2 - (p0_2d[0] ** 2 + p0_2d[1] ** 2),
            p2_2d[0] ** 2 + p2_2d[1] ** 2 - (p0_2d[0] ** 2 + p0_2d[1] ** 2),
        ],
        dtype=np.float64,
    )
    if abs(np.linalg.det(matrix)) <= 1e-10:
        raise ValueError("Picked circle anchors are nearly collinear.")
    center_2d = np.linalg.solve(matrix, rhs)
    center_m = p0 + axis_u * center_2d[0] + axis_v * center_2d[1]
    radius_m = float(np.linalg.norm(p0 - center_m))
    if radius_m <= 1e-9:
        raise ValueError("Invalid circle radius from selected anchors.")
    return center_m, axis_u, axis_v, radius_m


def generate_circle_candidates(
    anchor_points_m: np.ndarray,
    contact_count: int,
    sweep_deg: float,
) -> np.ndarray:
    if contact_count < 3:
        raise ValueError("Circle mode requires at least 3 contact points.")
    center_m, _, _, radius_m = _fit_circle(anchor_points_m)

    start_dir = _normalize(anchor_points_m[0] - center_m)
    plane_normal = _normalize(np.cross(anchor_points_m[1] - anchor_points_m[0], anchor_points_m[2] - anchor_points_m[0]))
    tangent_dir = _normalize(np.cross(plane_normal, start_dir))
    second_dir = _normalize(anchor_points_m[1] - center_m)
    if float(np.dot(second_dir, tangent_dir)) < 0.0:
        tangent_dir = -tangent_dir

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
        point = center_m + radius_m * (
            math.cos(float(angle)) * start_dir + math.sin(float(angle)) * tangent_dir
        )
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
        raise ValueError("No grasp contacts could be projected onto the sampled surface.")

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
    mode: str,
    anchor_indices: list[int] | list[list[int]],
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

    generator: dict[str, Any] = {
        "type": "curve",
        "curve_mode": mode,
        "sweep_deg": float(sweep_deg),
    }
    if mode == "line":
        generator["anchor_sample_indices"] = [int(v) for v in anchor_indices]
    else:
        anchor_groups = [
            [int(v) for v in group]
            for group in anchor_indices
        ]
        generator["anchor_sample_index_groups"] = anchor_groups
        generator["circle_count"] = len(anchor_groups)

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
    anchor_points_m: np.ndarray,
    contact_points_m: np.ndarray,
    contact_normals: np.ndarray,
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

    anchor_cloud = o3d.geometry.PointCloud()
    anchor_cloud.points = o3d.utility.Vector3dVector(anchor_points_m)
    anchor_cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray([[0.15, 0.55, 0.15]], dtype=np.float64), (len(anchor_points_m), 1))
    )

    line_points = []
    line_indices = []
    line_colors = []
    normal_length = max(
        0.01,
        0.06 * float(np.linalg.norm(surface_points_m.max(axis=0) - surface_points_m.min(axis=0))),
    )
    for idx, (point, normal) in enumerate(zip(contact_points_m, contact_normals)):
        start_idx = len(line_points)
        line_points.append(point.tolist())
        line_points.append((point + normal * normal_length).tolist())
        line_indices.append([start_idx, start_idx + 1])
        line_colors.append([0.95, 0.25, 0.25])
    normal_lines = o3d.geometry.LineSet()
    normal_lines.points = o3d.utility.Vector3dVector(np.asarray(line_points, dtype=np.float64))
    normal_lines.lines = o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32))
    normal_lines.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))

    extent = float(np.linalg.norm(surface_points_m.max(axis=0) - surface_points_m.min(axis=0)))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.03, extent * 0.18))

    print("Preview window:")
    print("  gray  : sampled mesh surface")
    print("  green : chosen anchors")
    print("  red   : generated grasp contacts and normals")
    print("  Q     : close preview")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Generated CAD Grasp Contacts", width=1600, height=1000)
    vis.add_geometry(cloud)
    vis.add_geometry(contact_cloud)
    vis.add_geometry(anchor_cloud)
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
    if args.mode == "line":
        anchor_indices = run_anchor_picker(
            surface_points_m,
            surface_normals,
            args.point_size,
            args.mode,
        )
        anchor_points_m = surface_points_m[np.asarray(anchor_indices, dtype=np.int32)]
        candidate_points_m = generate_line_candidates(anchor_points_m, args.contact_count)
        contact_points_m, contact_normals, contact_faces, contact_sample_indices = (
            project_candidates_to_surface(
                candidate_points_m,
                surface_points_m,
                surface_normals,
                surface_faces,
            )
        )
    else:
        all_anchor_indices: list[list[int]] = []
        all_anchor_points: list[np.ndarray] = []
        all_contact_points: list[np.ndarray] = []
        all_contact_normals: list[np.ndarray] = []
        all_contact_faces: list[np.ndarray] = []
        all_contact_sample_indices: list[np.ndarray] = []

        while True:
            circle_anchor_indices = run_anchor_picker(
                surface_points_m,
                surface_normals,
                args.point_size,
                args.mode,
            )
            circle_anchor_points_m = surface_points_m[
                np.asarray(circle_anchor_indices, dtype=np.int32)
            ]
            candidate_points_m = generate_circle_candidates(
                circle_anchor_points_m,
                args.contact_count,
                args.sweep_deg,
            )
            (
                circle_contact_points_m,
                circle_contact_normals,
                circle_contact_faces,
                circle_contact_sample_indices,
            ) = project_candidates_to_surface(
                candidate_points_m,
                surface_points_m,
                surface_normals,
                surface_faces,
            )
            all_anchor_indices.append(circle_anchor_indices)
            all_anchor_points.append(circle_anchor_points_m)
            all_contact_points.append(circle_contact_points_m)
            all_contact_normals.append(circle_contact_normals)
            all_contact_faces.append(circle_contact_faces)
            all_contact_sample_indices.append(circle_contact_sample_indices)

            if not prompt_continue_circle_selection():
                break

        anchor_indices = all_anchor_indices
        anchor_points_m = np.vstack(all_anchor_points)
        contact_points_m = np.vstack(all_contact_points)
        contact_normals = np.vstack(all_contact_normals)
        contact_faces = np.concatenate(all_contact_faces)
        contact_sample_indices = np.concatenate(all_contact_sample_indices)

    preview_generated_contacts(
        surface_points_m,
        surface_normals,
        anchor_points_m,
        contact_points_m,
        contact_normals,
        args.point_size,
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
        mode=args.mode,
        anchor_indices=anchor_indices,
        sweep_deg=args.sweep_deg if args.mode == "circle" else 0.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved {len(payload['contacts'])} grasp contacts to {output_path}")
    print(f"Mesh: {mesh_path}")
    print("This file is directly usable by VGR bin picking as pick_contacts.json.")


if __name__ == "__main__":
    main()
