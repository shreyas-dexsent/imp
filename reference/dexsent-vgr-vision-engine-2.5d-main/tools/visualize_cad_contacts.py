from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
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


def _normalize_vec(vec: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return None
    return arr / norm


def _rotate_vec_about_axis(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_n = _normalize_vec(axis)
    if axis_n is None:
        return np.asarray(vec, dtype=np.float64)
    vec_n = np.asarray(vec, dtype=np.float64)
    cos_t = math.cos(float(angle_rad))
    sin_t = math.sin(float(angle_rad))
    return (
        (vec_n * cos_t)
        + (np.cross(axis_n, vec_n) * sin_t)
        + (axis_n * float(np.dot(axis_n, vec_n)) * (1.0 - cos_t))
    )


def _repair_legacy_symmetric_ring_approach_axis(
    payload: dict,
    grasp: dict,
    grasp_index: int,
) -> Optional[np.ndarray]:
    if str(payload.get("approach_axis_convention") or "").strip().lower() == "pregrasp_to_center":
        return _normalize_vec(np.asarray(grasp.get("approach_axis_local") or [], dtype=np.float64))
    jaw_axis = _normalize_vec(np.asarray(grasp.get("jaw_axis_local") or [], dtype=np.float64))
    stored = _normalize_vec(np.asarray(grasp.get("approach_axis_local") or [], dtype=np.float64))
    if jaw_axis is None or stored is None:
        return stored
    generator = payload.get("generator") if isinstance(payload.get("generator"), dict) else {}
    if str(generator.get("mode") or "").strip().lower() != "symmetric_ring":
        return stored
    if str(generator.get("symmetric_approach_mode") or "").strip().lower() != "axis_roll":
        return stored
    group_index = int(grasp.get("generator_group_index") or 0)
    grasp_count_per_ring = int(generator.get("grasp_count_per_ring") or 0)
    if group_index <= 0 and grasp_count_per_ring > 0:
        group_index = int(grasp_index // grasp_count_per_ring) + 1
    axis_record = None
    for record in (generator.get("pair_axis_records") or []):
        if int(record.get("group_index") or 0) == group_index:
            axis_record = record
            break
    axis_direction = _normalize_vec(
        np.asarray((axis_record or {}).get("axis_direction") or [], dtype=np.float64)
    )
    if axis_direction is None:
        return stored
    if abs(float(np.dot(stored, axis_direction))) < 0.9:
        return stored
    finger_axis = axis_direction - (float(np.dot(axis_direction, jaw_axis)) * jaw_axis)
    finger_axis = _normalize_vec(finger_axis)
    if finger_axis is None:
        return stored
    roll_values = generator.get("approach_roll_deg_per_group") or []
    roll_deg = None
    if isinstance(roll_values, list) and len(roll_values) >= group_index > 0:
        roll_deg = float(roll_values[group_index - 1])
    elif generator.get("approach_roll_deg") is not None:
        roll_deg = float(generator.get("approach_roll_deg"))
    if roll_deg is not None and abs(float(roll_deg)) > 1e-9:
        finger_axis = _normalize_vec(
            _rotate_vec_about_axis(finger_axis, jaw_axis, math.radians(float(roll_deg)))
        )
    if finger_axis is None:
        return stored
    repaired = _normalize_vec(np.cross(finger_axis, jaw_axis))
    return repaired if repaired is not None else stored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize pick_contacts.json points and normals on a CAD mesh."
    )
    parser.add_argument(
        "--object-folder",
        type=str,
        default=None,
        help="Object folder containing the CAD mesh and pick_contacts.json.",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Mesh path when not using --object-folder.",
    )
    parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="pick_contacts.json path when not using --object-folder.",
    )
    parser.add_argument(
        "--mesh-units",
        choices=["mm", "m"],
        default=None,
        help="Override mesh units if they are not defined in pick_contacts.json.",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=None,
        help="Override mesh scale if it is not defined in pick_contacts.json.",
    )
    parser.add_argument(
        "--mesh-color",
        type=float,
        nargs=3,
        default=[0.72, 0.78, 0.84],
        metavar=("R", "G", "B"),
        help="Mesh RGB color in the 0..1 range.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=8.0,
        help="Open3D point size for contact markers.",
    )
    parser.add_argument(
        "--normal-length",
        type=float,
        default=0.01,
        help="Contact normal length in meters.",
    )
    parser.add_argument(
        "--group-index",
        type=int,
        nargs="*",
        default=None,
        help="Optional grasp family indices to review (for example: --group-index 1 3).",
    )
    parser.add_argument(
        "--review-groups",
        action="store_true",
        help="Show one grasp family at a time and label it with keyboard shortcuts.",
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=None,
        help="Optional JSON path where grasp family labels will be saved in review mode.",
    )
    parser.add_argument(
        "--show-group",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.object_folder:
        object_folder = resolve_workspace_path(args.object_folder)
        assets = resolve_object_assets(object_folder)
        mesh_path = assets.mesh_path
        contacts_path = object_folder / "pick_contacts.json"
        return mesh_path, contacts_path

    if not args.mesh or not args.contacts:
        raise ValueError(
            "Provide --object-folder, or provide both --mesh and --contacts."
        )

    return resolve_workspace_path(args.mesh), resolve_workspace_path(args.contacts)


def load_centered_mesh(
    mesh_path: Path,
    mesh_units: str,
    mesh_scale: float,
):
    import trimesh

    mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"No geometry found in {mesh_path}")
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)!r}")
    mesh = mesh.copy()
    center_mesh_units = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
    mesh.apply_translation(-center_mesh_units)
    scale_to_m = (1.0 if mesh_units == "m" else 0.001) * float(mesh_scale)
    mesh.apply_scale(scale_to_m)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def load_contacts(
    contacts_path: Path,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    payload = json.loads(contacts_path.read_text(encoding="utf-8"))
    contacts = payload.get("contacts", [])
    enabled_contacts = [entry for entry in contacts if entry.get("enabled", True)]
    grasp_pairs = [
        entry
        for entry in (payload.get("grasps") or payload.get("grasp_candidates") or [])
        if isinstance(entry, dict) and entry.get("enabled", True)
    ]

    if not enabled_contacts and not grasp_pairs:
        raise ValueError(f"No enabled contacts or grasps found in {contacts_path}")

    points_m = np.asarray(
        [entry["point_local_m"] for entry in enabled_contacts],
        dtype=np.float64,
    ) if enabled_contacts else np.zeros((0, 3), dtype=np.float64)
    normals = np.asarray(
        [entry["normal_local"] for entry in enabled_contacts],
        dtype=np.float64,
    ) if enabled_contacts else np.zeros((0, 3), dtype=np.float64)
    return payload, points_m, normals, grasp_pairs


def _group_roll_map(payload: dict) -> dict[int, float]:
    generator = payload.get("generator") if isinstance(payload.get("generator"), dict) else {}
    roll_values = generator.get("approach_roll_deg_per_group")
    if not isinstance(roll_values, list):
        return {}
    result: dict[int, float] = {}
    for idx, value in enumerate(roll_values, start=1):
        try:
            result[idx] = float(value)
        except Exception:
            continue
    return result


def _group_color(group_index: int) -> np.ndarray:
    palette = {
        1: np.array([0.17, 0.50, 0.96], dtype=np.float64),
        2: np.array([0.96, 0.55, 0.12], dtype=np.float64),
        3: np.array([0.12, 0.70, 0.32], dtype=np.float64),
    }
    return palette.get(int(group_index), np.array([0.84, 0.24, 0.75], dtype=np.float64))


def _classify_grasp_family(entries: list[dict]) -> str:
    approach_vectors = []
    for entry in entries:
        vec = _normalize_vec(np.asarray(entry.get("approach_axis_local") or [], dtype=np.float64))
        if vec is not None:
            approach_vectors.append(vec)
    if not approach_vectors:
        return "unknown"
    abs_avg = np.mean(np.abs(np.asarray(approach_vectors, dtype=np.float64)), axis=0)
    axis_labels = ["x", "y", "z"]
    dominant_axis = axis_labels[int(np.argmax(abs_avg))]
    if dominant_axis == "y":
        return "axial_end"
    return "radial_side"


def _summarize_grasp_groups(payload: dict, grasp_pairs: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for grasp in grasp_pairs:
        group_index = int(grasp.get("generator_group_index") or 0)
        grouped.setdefault(group_index, []).append(grasp)
    roll_map = _group_roll_map(payload)
    summaries: list[dict] = []
    for group_index in sorted(grouped):
        entries = grouped[group_index]
        widths = [float(entry.get("opening_width_m") or 0.0) for entry in entries]
        center_y = [
            float((entry.get("center_local_m") or [0.0, 0.0, 0.0])[1])
            for entry in entries
        ]
        role = _classify_grasp_family(entries)
        summaries.append(
            {
                "group_index": group_index,
                "count": len(entries),
                "mean_opening_width_m": float(np.mean(widths)) if widths else 0.0,
                "mean_center_y_m": float(np.mean(center_y)) if center_y else 0.0,
                "approach_roll_deg": roll_map.get(group_index),
                "role": role,
            }
        )
    return summaries


def _family_name(role: str) -> str:
    return {
        "axial_end": "axial/end-entry",
        "radial_side": "radial/side-pinch",
        "unknown": "unknown",
    }.get(str(role), "unknown")


def trimesh_to_open3d(mesh, mesh_color: list[float]):
    import open3d as o3d

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices, dtype=np.float64)
    )
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.faces, dtype=np.int32)
    )
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color(np.asarray(mesh_color, dtype=np.float64))
    return o3d_mesh


def build_normal_lines(
    points_m: np.ndarray,
    normals: np.ndarray,
    normal_length: float,
):
    import open3d as o3d

    line_points = []
    line_indices = []
    line_colors = []
    for point, normal in zip(points_m, normals):
        start_idx = len(line_points)
        line_points.append(point.tolist())
        line_points.append((point + normal * float(normal_length)).tolist())
        line_indices.append([start_idx, start_idx + 1])
        line_colors.append([0.92, 0.22, 0.22])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(line_points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))
    return line_set


def build_parallel_jaw_lines(
    payload: dict,
    grasp_pairs: list[dict],
    normal_length: float,
    *,
    family_color: np.ndarray,
):
    import open3d as o3d

    line_points = []
    line_indices = []
    line_colors = []
    pair_points = []

    for grasp_index, grasp in enumerate(grasp_pairs):
        contact_a = np.asarray(grasp.get("contact_a_local_m") or [], dtype=np.float64)
        contact_b = np.asarray(grasp.get("contact_b_local_m") or [], dtype=np.float64)
        center = np.asarray(grasp.get("center_local_m") or [], dtype=np.float64)
        approach_axis = _repair_legacy_symmetric_ring_approach_axis(
            payload,
            grasp,
            grasp_index,
        )
        if contact_a.shape != (3,) or contact_b.shape != (3,):
            continue
        pair_points.append(contact_a)
        pair_points.append(contact_b)
        start_idx = len(line_points)
        line_points.append(contact_a.tolist())
        line_points.append(contact_b.tolist())
        line_indices.append([start_idx, start_idx + 1])
        line_colors.append(family_color.tolist())
        if center.shape == (3,) and approach_axis is not None:
            arrow_idx = len(line_points)
            approach_start = center - (approach_axis * float(normal_length))
            line_points.append(approach_start.tolist())
            line_points.append(center.tolist())
            line_indices.append([arrow_idx, arrow_idx + 1])
            line_colors.append([0.98, 0.98, 0.98])

    line_points_arr = (
        np.asarray(line_points, dtype=np.float64).reshape(-1, 3)
        if line_points
        else np.zeros((0, 3), dtype=np.float64)
    )
    line_indices_arr = (
        np.asarray(line_indices, dtype=np.int32).reshape(-1, 2)
        if line_indices
        else np.zeros((0, 2), dtype=np.int32)
    )
    line_colors_arr = (
        np.asarray(line_colors, dtype=np.float64).reshape(-1, 3)
        if line_colors
        else np.zeros((0, 3), dtype=np.float64)
    )
    pair_points_arr = (
        np.asarray(pair_points, dtype=np.float64).reshape(-1, 3)
        if pair_points
        else np.zeros((0, 3), dtype=np.float64)
    )

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points_arr)
    line_set.lines = o3d.utility.Vector2iVector(line_indices_arr)
    line_set.colors = o3d.utility.Vector3dVector(line_colors_arr)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pair_points_arr)
    point_cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray([family_color], dtype=np.float64), (len(pair_points_arr), 1))
    )
    return point_cloud, line_set


def build_grasp_markers(
    grasp_pairs: list[dict],
    *,
    family_color: np.ndarray,
    marker_radius: float,
):
    import open3d as o3d

    markers = []
    accent_color = np.clip((family_color * 0.75) + 0.25, 0.0, 1.0)
    center_color = np.asarray([0.98, 0.96, 0.28], dtype=np.float64)
    for grasp in grasp_pairs:
        for key in ("contact_a_local_m", "contact_b_local_m"):
            point = np.asarray(grasp.get(key) or [], dtype=np.float64)
            if point.shape != (3,):
                continue
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(marker_radius))
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(accent_color)
            sphere.translate(point)
            markers.append(sphere)
        center = np.asarray(grasp.get("center_local_m") or [], dtype=np.float64)
        if center.shape == (3,):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(marker_radius) * 0.8)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(center_color)
            sphere.translate(center)
            markers.append(sphere)
    return markers


def _filter_grasp_pairs_by_group(
    grasp_pairs: list[dict],
    group_index: int,
) -> list[dict]:
    return [
        grasp
        for grasp in grasp_pairs
        if int(grasp.get("generator_group_index") or 0) == int(group_index)
    ]


def build_surface_preview_cloud(mesh, sample_count: int = 16000) -> np.ndarray:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    samples = mesh.sample(max(2048, int(sample_count)))
    return np.asarray(samples, dtype=np.float64).reshape(-1, 3)


def show_group(
    payload: dict,
    mesh,
    points_m: np.ndarray,
    normals: np.ndarray,
    grasp_pairs: list[dict],
    point_size: float,
    normal_length: float,
    mesh_color: list[float],
    *,
    group_index: int,
) -> None:
    import open3d as o3d

    group_entries = _filter_grasp_pairs_by_group(grasp_pairs, group_index)
    if not group_entries:
        raise ValueError(f"No grasp entries found for group {group_index}.")

    surface_points_m = build_surface_preview_cloud(mesh)
    surface_cloud = o3d.geometry.PointCloud()
    if len(surface_points_m):
        surface_cloud.points = o3d.utility.Vector3dVector(surface_points_m)
        surface_cloud.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray([[0.72, 0.78, 0.84]], dtype=np.float64), (len(surface_points_m), 1))
        )

    contact_cloud = o3d.geometry.PointCloud()
    if len(points_m):
        contact_cloud.points = o3d.utility.Vector3dVector(points_m)
        contact_cloud.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray([[0.95, 0.18, 0.18]], dtype=np.float64), (len(points_m), 1))
        )
    normal_lines = (
        build_normal_lines(points_m, normals, normal_length) if len(points_m) else None
    )
    jaw_points, jaw_lines = build_parallel_jaw_lines(
        payload,
        group_entries,
        normal_length,
        family_color=_group_color(group_index),
    )
    grasp_markers = build_grasp_markers(
        group_entries,
        family_color=_group_color(group_index),
        marker_radius=max(0.0008, float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) * 0.018),
    )
    extent = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.03, extent * 0.18))

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"CAD Contact Review | group {group_index} | close window when done",
        width=1600,
        height=1000,
    )
    if len(surface_points_m):
        vis.add_geometry(surface_cloud)
    if len(points_m):
        vis.add_geometry(contact_cloud)
    if normal_lines is not None:
        vis.add_geometry(normal_lines)
    vis.add_geometry(jaw_points)
    vis.add_geometry(jaw_lines)
    for marker in grasp_markers:
        vis.add_geometry(marker)
    vis.add_geometry(frame)
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.98, 0.98, 0.99], dtype=np.float64)
    render_option.point_size = max(float(point_size), 3.0)
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()


def review_groups(
    mesh_path: Path,
    contacts_path: Path,
    payload: dict,
    mesh,
    points_m: np.ndarray,
    normals: np.ndarray,
    grasp_pairs: list[dict],
    point_size: float,
    normal_length: float,
    mesh_color: list[float],
    *,
    group_indices: Optional[set[int]],
    labels_out: Path,
) -> None:
    group_summaries = _summarize_grasp_groups(payload, grasp_pairs)
    if group_indices:
        group_summaries = [
            summary for summary in group_summaries if int(summary["group_index"]) in group_indices
        ]
    if not group_summaries:
        raise ValueError("No grasp groups available for review.")

    label_map: dict[str, str] = {}
    review_notes: dict[str, dict] = {}
    ordered_groups = [int(summary["group_index"]) for summary in group_summaries]
    index = 0
    quit_requested = False

    labels_out.parent.mkdir(parents=True, exist_ok=True)

    def save_labels() -> None:
        payload_out = {
            "object_name": payload.get("object_name"),
            "contacts_path": str(labels_out.parent / "pick_contacts.json"),
            "labels": label_map,
            "group_summaries": review_notes,
        }
        labels_out.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")

    print("")
    print("Review mode flow:")
    print("  1. Inspect the current group in the Open3D window.")
    print("  2. Close the window with Q or Escape.")
    print("  3. Enter a label in the terminal: i/e/s/u/n/p/q.")

    while 0 <= index < len(ordered_groups):
        group_index = ordered_groups[index]
        group_summary = next(
            summary for summary in group_summaries if int(summary["group_index"]) == group_index
        )
        current_label = label_map.get(str(group_index), "unlabeled")
        review_notes[str(group_index)] = {
            "group_index": group_index,
            "suggested_role": str(group_summary["role"]),
            "suggested_role_label": _family_name(str(group_summary["role"])),
            "current_label": current_label,
            "count": int(group_summary["count"]),
            "mean_opening_width_m": float(group_summary["mean_opening_width_m"]),
            "mean_center_y_m": float(group_summary["mean_center_y_m"]),
            "approach_roll_deg": group_summary["approach_roll_deg"],
        }
        print("")
        print(
            f"Showing group {group_index} "
            f"({index + 1}/{len(ordered_groups)}) "
            f"type={_family_name(str(group_summary['role']))} "
            f"current_label={current_label}"
        )

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mesh",
                str(mesh_path),
                "--contacts",
                str(contacts_path),
                "--point-size",
                str(point_size),
                "--normal-length",
                str(normal_length),
                "--mesh-color",
                *(str(v) for v in mesh_color),
                "--show-group",
                str(group_index),
            ],
            check=True,
        )

        selected_action = input(
            f"Label group {group_index} [i=internal, e=external, s=skip, u=clear, n=next, p=prev, q=quit]: "
        ).strip().lower()[:1]
        if selected_action not in {"i", "e", "s", "u", "n", "p", "q"}:
            selected_action = "n"
        selected_action = {
            "i": "internal",
            "e": "external",
            "s": "skip",
            "u": "clear",
            "n": "next",
            "p": "prev",
            "q": "quit",
        }[selected_action]
        if selected_action == "internal":
            label_map[str(group_index)] = "internal"
            print(f"Marked group {group_index} as internal.")
            index += 1
        elif selected_action == "external":
            label_map[str(group_index)] = "external"
            print(f"Marked group {group_index} as external.")
            index += 1
        elif selected_action == "skip":
            print(f"Skipped group {group_index}.")
            index += 1
        elif selected_action == "clear":
            label_map.pop(str(group_index), None)
            print(f"Cleared label for group {group_index}.")
        elif selected_action == "prev":
            index = max(0, index - 1)
        elif selected_action == "quit":
            quit_requested = True
            break
        else:
            index += 1
        save_labels()

    save_labels()
    print("")
    print(f"Saved grasp family labels to {labels_out}")
    if quit_requested:
        print("Review ended early by user request.")


def main() -> None:
    args = parse_args()
    mesh_path, contacts_path = resolve_inputs(args)
    payload, points_m, normals, grasp_pairs = load_contacts(contacts_path)
    mesh_units = args.mesh_units or payload.get("mesh_units", "mm")
    mesh_scale = (
        float(args.mesh_scale)
        if args.mesh_scale is not None
        else float(payload.get("mesh_scale", 1.0))
    )
    mesh = load_centered_mesh(mesh_path, mesh_units, mesh_scale)

    print(f"Mesh: {mesh_path}")
    print(f"Contacts: {contacts_path}")
    print(f"Loaded {len(points_m)} enabled vacuum contacts.")
    print(f"Loaded {len(grasp_pairs)} enabled parallel-jaw grasps.")

    labels_out = (
        resolve_workspace_path(args.labels_out)
        if args.labels_out
        else contacts_path.with_name("grasp_family_labels.json")
    )

    if args.show_group is not None:
        show_group(
            payload=payload,
            mesh=mesh,
            points_m=points_m,
            normals=normals,
            grasp_pairs=grasp_pairs,
            point_size=args.point_size,
            normal_length=args.normal_length,
            mesh_color=args.mesh_color,
            group_index=int(args.show_group),
        )
        return

    review_groups(
        mesh_path=mesh_path,
        contacts_path=contacts_path,
        payload=payload,
        mesh=mesh,
        points_m=points_m,
        normals=normals,
        grasp_pairs=grasp_pairs,
        point_size=args.point_size,
        normal_length=args.normal_length,
        mesh_color=args.mesh_color,
        group_indices=None if not args.group_index else {int(v) for v in args.group_index},
        labels_out=labels_out,
    )


if __name__ == "__main__":
    main()
