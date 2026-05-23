from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

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
            "Pick CAD suction contacts on a MegaPose object mesh and save them as "
            "object-local points/normals in pick_contacts.json."
        )
    )
    parser.add_argument(
        "--object-folder",
        type=str,
        default=None,
        help="MegaPose object folder containing object.pt and object mesh assets.",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Optional mesh path when not using --object-folder.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Object label used in the saved payload. Defaults to the folder name or mesh stem.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output json path. Defaults to <object-folder>/pick_contacts.json.",
    )
    parser.add_argument(
        "--mesh-units",
        choices=["mm", "m"],
        default="mm",
        help="Units of the source CAD vertices before scaling.",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Extra scale applied on top of mesh-units conversion.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=40000,
        help="Number of surface samples shown for picking.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Open3D point size for the sampled cloud.",
    )
    parser.add_argument(
        "--tool-axis",
        type=str,
        default="-z",
        help="Tool axis that should align with the saved surface normal in VGR.",
    )
    parser.add_argument(
        "--force-yaw-deg",
        type=float,
        default=0.0,
        help="Base-frame yaw VGR should keep while roll/pitch comes from the contact normal.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="upward_then_camera",
        help="Selection strategy for VGR. Default prefers upward normals, then camera proximity.",
    )
    return parser.parse_args()


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


def sample_surface_points(mesh: object, sample_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def run_picker(points_m: np.ndarray, normals: np.ndarray, point_size: float) -> list[int]:
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

    print("Open3D contact picker controls:")
    print("  Shift + left click : pick a contact sample")
    print("  Shift + right click: undo the last pick")
    print("  Q or Escape        : finish and save")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="CAD Pick Contacts",
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
    return list(dict.fromkeys(picked))


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
    picked_indices: list[int],
    tool_axis: str,
    force_yaw_deg: float,
    selection_mode: str,
) -> dict:
    contacts = []
    for order, sample_idx in enumerate(picked_indices, start=1):
        point_local_m = np.asarray(points_m[sample_idx], dtype=np.float64)
        point_local_mesh_units = point_local_m / float(scale_to_m)
        normal_local = np.asarray(normals[sample_idx], dtype=np.float64)
        contacts.append(
            {
                "id": f"contact_{order:03d}",
                "label": f"contact_{order:03d}",
                "point_local_m": [float(v) for v in point_local_m.tolist()],
                "point_local_mesh_units": [
                    float(v) for v in point_local_mesh_units.tolist()
                ],
                "normal_local": [float(v) for v in normal_local.tolist()],
                "face_index": int(face_indices[sample_idx]),
                "sample_index": int(sample_idx),
                "enabled": True,
            }
        )

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
        "contacts": contacts,
    }


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


def main() -> None:
    args = parse_args()
    object_folder, mesh_path, output_path, label = resolve_inputs(args)

    mesh, center_mesh_units, scale_to_m = load_centered_mesh(
        mesh_path=mesh_path,
        mesh_units=args.mesh_units,
        mesh_scale=args.mesh_scale,
    )
    points_m, normals, face_indices = sample_surface_points(mesh, args.sample_count)
    picked_indices = run_picker(points_m, normals, args.point_size)
    if not picked_indices:
        print("No contacts selected. Nothing was saved.")
        return

    payload = build_payload(
        object_folder=object_folder,
        mesh_path=mesh_path,
        label=label,
        mesh_units=args.mesh_units,
        mesh_scale=args.mesh_scale,
        scale_to_m=scale_to_m,
        center_mesh_units=center_mesh_units,
        points_m=points_m,
        normals=normals,
        face_indices=face_indices,
        picked_indices=picked_indices,
        tool_axis=args.tool_axis,
        force_yaw_deg=args.force_yaw_deg,
        selection_mode=args.selection_mode,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved {len(payload['contacts'])} pick contacts to {output_path}")
    print(f"Mesh: {mesh_path}")
    print("VGR will auto-load this file when the MegaPose object folder is used in bin picking.")


if __name__ == "__main__":
    main()
