# 
# python tools/generate_cad_parallel_jaw_grasps.py   --object-folder ~/imp/data/stations/station-1/assets/asset-1/objects/barel2   --mode symmetric_ring   --axis y   --symmetric-seed-mode single   --symmetric-approach-mode axis_roll   --grasp-count 24   --sweep-deg 360   --tool-axis z   --jaw-axis y   --selection-mode camera_normal

#

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

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
            "Generate parallel-jaw CAD grasp candidates. Manual mode creates one "
            "grasp from 3 picked anchors (jaw A, jaw B, approach reference). "
            "Symmetric mode creates antipodal jaw pairs around a detected "
            "rotational axis, useful for cylinders."
        )
    )
    parser.add_argument("--object-folder", type=str, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=["manual_pairs", "symmetric_ring"], required=True)
    parser.add_argument("--mesh-units", choices=["mm", "m"], default="mm")
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--surface-sample-count", type=int, default=120000)
    parser.add_argument(
        "--grasp-count",
        type=int,
        default=24,
        help="Number of jaw grasps to generate per symmetric ring.",
    )
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument(
        "--tool-axis",
        type=str,
        default="-z",
        help="Tool axis aligned with grasp approach direction in runtime.",
    )
    parser.add_argument(
        "--jaw-axis",
        type=str,
        default="x",
        help="Tool axis aligned with the jaw closing direction in runtime.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="closest_contact_to_camera",
        help="Runtime selection mode. Default prefers the grasp whose closest jaw contact is nearest the camera.",
    )
    parser.add_argument(
        "--axis",
        choices=["prompt", "auto", "x", "y", "z", "custom_points"],
        default="prompt",
        help="Symmetry axis for symmetric_ring mode. 'prompt' shows an axis guide first.",
    )
    parser.add_argument(
        "--symmetric-seed-mode",
        choices=["single", "pair"],
        default="single",
        help="For symmetric_ring: pick one seed point and infer the opposite contact, or pick the full pair manually.",
    )
    parser.add_argument(
        "--symmetric-approach-mode",
        choices=["axis_roll", "reference_pick"],
        default="axis_roll",
        help=(
            "How to define the approach direction in symmetric_ring mode. "
            "'axis_roll' derives it from the symmetry axis and a numeric roll around the jaw axis. "
            "'reference_pick' uses an extra point pick."
        ),
    )
    parser.add_argument(
        "--approach-roll-deg",
        type=float,
        default=0.0,
        help=(
            "With --symmetric-approach-mode axis_roll, rotate the derived approach "
            "direction around the jaw axis by this many degrees."
        ),
    )
    parser.add_argument("--sweep-deg", type=float, default=360.0)
    parser.add_argument(
        "--preview-single-candidate-only",
        action="store_true",
        help=(
            "Debug preview mode: show only the first candidate in the current preview "
            "instead of the full generated ring."
        ),
    )
    return parser.parse_args()


def resolve_inputs(
    args: argparse.Namespace,
) -> tuple[Optional[Path], Path, Path, str]:
    if args.object_folder:
        object_folder = resolve_workspace_path(args.object_folder)
        assets = resolve_object_assets(object_folder, label_override=args.label)
        label = args.label or assets.label or object_folder.name
        output_path = args.output or (object_folder / "pick_contacts.json")
        return object_folder, assets.mesh_path, output_path, str(label)
    if not args.mesh:
        raise ValueError("Provide either --object-folder or --mesh.")
    mesh_path = resolve_workspace_path(args.mesh)
    output_path = args.output or (mesh_path.parent / "pick_contacts.json")
    return None, mesh_path, output_path, str(args.label or mesh_path.stem)


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


def _normalize(vec: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        raise ValueError("Degenerate vector encountered.")
    return arr / norm


def infer_grasp_family_label(
    normal_a_local: np.ndarray | list[float],
    normal_b_local: np.ndarray | list[float],
    jaw_axis_local: np.ndarray | list[float],
) -> str:
    """Infer inner/outer family from contact normals and jaw axis.

    External/outer antipodal grasps have contact normals pointing away from the
    center, producing opposite signs along the jaw axis. Internal/inner grasps
    have same-sign normal components because the jaws expand into a cavity.
    """
    try:
        jaw_n = _normalize(jaw_axis_local)
        normal_a_n = _normalize(normal_a_local)
        normal_b_n = _normalize(normal_b_local)
    except ValueError:
        return "unknown"
    dot_a = float(np.dot(normal_a_n, jaw_n))
    dot_b = float(np.dot(normal_b_n, jaw_n))
    if abs(dot_a) < 1e-3 or abs(dot_b) < 1e-3:
        return "unknown"
    return "internal" if (dot_a * dot_b) > 0.0 else "external"


def _prompt_continue(next_action: str = "finish") -> bool:
    print("Pair completed.")
    print(
        "Press N then Enter to draw another pair, "
        f"or just Enter to {next_action}."
    )
    return input("> ").strip().lower() == "n"


def _set_closeup_friendly_projection(scene_widget: Any, bbox_extent: np.ndarray) -> None:
    import open3d as o3d

    extent_norm = float(np.linalg.norm(np.asarray(bbox_extent, dtype=np.float64)))
    near_plane = max(1e-4, extent_norm * 0.0025)
    far_plane = max(1.0, extent_norm * 12.0)
    frame = scene_widget.frame
    aspect_ratio = max(float(frame.width) / max(float(frame.height), 1.0), 1e-3)
    camera = scene_widget.scene.camera
    camera.set_projection(
        float(camera.get_field_of_view()),
        aspect_ratio,
        near_plane,
        far_plane,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )


def show_symmetry_axis_guide(
    points_m: np.ndarray,
    point_size: float,
) -> None:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_m)
    cloud.paint_uniform_color([0.72, 0.78, 0.84])

    extent = float(np.linalg.norm(points_m.max(axis=0) - points_m.min(axis=0)))
    axis_length = max(0.04, extent * 0.38)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.03, extent * 0.18))

    axis_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )
    axis_lines = o3d.geometry.LineSet()
    axis_lines.points = o3d.utility.Vector3dVector(axis_points)
    axis_lines.lines = o3d.utility.Vector2iVector(
        np.asarray([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    )
    axis_lines.colors = o3d.utility.Vector3dVector(
        np.asarray(
            [
                [1.0, 0.25, 0.25],  # X red
                [0.2, 0.82, 0.35],  # Y green
                [0.2, 0.52, 1.0],   # Z blue
            ],
            dtype=np.float64,
        )
    )

    print("Symmetry axis guide:")
    print("  X axis = red")
    print("  Y axis = green")
    print("  Z axis = blue")
    print("  Q or Escape = close the axis guide, then choose the axis in the terminal")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Symmetry Axis Guide", width=1600, height=1000)
    vis.add_geometry(cloud)
    vis.add_geometry(frame)
    vis.add_geometry(axis_lines)
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.96, 0.97, 0.98], dtype=np.float64)
    render_option.point_size = float(point_size)
    vis.run()
    vis.destroy_window()


def prompt_symmetry_axis(default_axis: str = "auto") -> str:
    default_value = str(default_axis or "auto").strip().lower()
    if default_value not in {"auto", "x", "y", "z", "custom_points"}:
        default_value = "auto"
    print("Choose symmetry axis from the guide colors:")
    print("  red   -> x")
    print("  green -> y")
    print("  blue  -> z")
    print("  custom_points -> pick 2 points to define a custom symmetry axis")
    print(f"Type x / y / z / auto / custom_points, then Enter. Default: {default_value}")
    response = input("> ").strip().lower()
    if response in {"x", "y", "z", "auto", "custom_points", "custom"}:
        if response == "custom":
            return "custom_points"
        return response
    return default_value


def _run_picker(
    mesh: object,
    points_m: np.ndarray,
    normals: np.ndarray,
    point_size: float,
    required_picks: int,
    window_name: str,
    description_lines: list[str],
) -> list[int]:
    import open3d as o3d
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for picking: {type(mesh)!r}")

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    pick_mesh = o3d.geometry.TriangleMesh()
    pick_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    pick_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    pick_mesh.paint_uniform_color([0.78, 0.82, 0.87])
    pick_mesh.compute_vertex_normals()
    extent = float(np.linalg.norm(points_m.max(axis=0) - points_m.min(axis=0)))
    frame_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.03, extent * 0.18))

    print("Open3D picker controls:")
    print("  Left click         : pick the hovered mesh surface point")
    print("  Left drag          : orbit the camera")
    print("  Mouse wheel        : zoom")
    print("  Backspace          : undo the last pick")
    print("  Enter / Escape     : finish selection")
    for line in description_lines:
        print(line)

    window = app.create_window(window_name, 1600, 1000)
    scene = o3d.visualization.gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    scene.set_view_controls(o3d.visualization.gui.SceneWidget.Controls.ROTATE_CAMERA)

    mesh_mat = o3d.visualization.rendering.MaterialRecord()
    mesh_mat.shader = "defaultLit"
    mesh_mat.base_color = [0.78, 0.82, 0.87, 1.0]
    scene.scene.add_geometry("pick_mesh", pick_mesh, mesh_mat)

    frame_mat = o3d.visualization.rendering.MaterialRecord()
    frame_mat.shader = "defaultUnlit"
    frame_mat.line_width = 2.0
    scene.scene.add_geometry("pick_frame", frame_geom, frame_mat)

    marker_mat = o3d.visualization.rendering.MaterialRecord()
    marker_mat.shader = "defaultUnlit"
    marker_mat.point_size = max(10.0, float(point_size) * 2.5)

    bbox = pick_mesh.get_axis_aligned_bounding_box()
    bbox_min = np.asarray(bbox.min_bound, dtype=np.float64)
    bbox_max = np.asarray(bbox.max_bound, dtype=np.float64)
    bbox_extent = bbox_max - bbox_min
    padding = np.maximum(bbox_extent * 0.35, 0.02)
    padded_bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min - padding, bbox_max + padding)
    scene.setup_camera(60.0, padded_bbox, padded_bbox.get_center())
    scene.scene.set_background([0.96, 0.97, 0.98, 1.0])
    _set_closeup_friendly_projection(scene, bbox_extent)

    window.add_child(scene)

    def on_layout(_layout_context: Any) -> None:
        scene.frame = window.content_rect

    window.set_on_layout(on_layout)

    state: dict[str, Any] = {
        "picked": [],
        "done": False,
        "last_down": None,
    }

    def refresh_markers() -> None:
        if scene.scene.has_geometry("picked_markers"):
            scene.scene.remove_geometry("picked_markers")
        if not state["picked"]:
            return
        marker_cloud = o3d.geometry.PointCloud()
        marker_cloud.points = o3d.utility.Vector3dVector(
            np.asarray([points_m[idx] for idx in state["picked"]], dtype=np.float64)
        )
        marker_cloud.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray([[0.95, 0.26, 0.18]], dtype=np.float64), (len(state["picked"]), 1))
        )
        scene.scene.add_geometry("picked_markers", marker_cloud, marker_mat)

    def finish() -> None:
        state["done"] = True
        window.close()

    def on_key(event: Any) -> Any:
        if event.type != o3d.visualization.gui.KeyEvent.Type.DOWN:
            return False
        if event.key in {o3d.visualization.gui.KeyName.ESCAPE, o3d.visualization.gui.KeyName.ENTER}:
            finish()
            return True
        if event.key == o3d.visualization.gui.KeyName.BACKSPACE and state["picked"]:
            state["picked"].pop()
            refresh_markers()
            print(f"Undo last pick. Remaining picks: {len(state['picked'])}")
            return True
        return False

    window.set_on_key(on_key)

    def try_pick(screen_x: int, screen_y: int) -> bool:
        frame = scene.frame
        x = float(screen_x - frame.x)
        y = float(screen_y - frame.y)
        if x < 0 or y < 0 or x >= frame.width or y >= frame.height:
            return False
        camera = scene.scene.camera
        view = np.asarray(camera.get_view_matrix(), dtype=np.float64)
        proj = np.asarray(camera.get_projection_matrix(), dtype=np.float64)
        pts_h = np.concatenate(
            [np.asarray(points_m, dtype=np.float64), np.ones((len(points_m), 1), dtype=np.float64)],
            axis=1,
        )
        clip = (proj @ (view @ pts_h.T)).T
        w = clip[:, 3]
        valid = np.abs(w) > 1e-9
        if not np.any(valid):
            return False
        ndc = np.zeros((len(points_m), 3), dtype=np.float64)
        ndc[valid] = clip[valid, :3] / w[valid, None]
        visible = (
            valid
            & (w > 0.0)
            & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
            & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
            & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
        )
        if not np.any(visible):
            return False
        screen = np.zeros((len(points_m), 2), dtype=np.float64)
        screen[visible, 0] = (ndc[visible, 0] * 0.5 + 0.5) * float(frame.width)
        screen[visible, 1] = ((1.0 - ndc[visible, 1]) * 0.5) * float(frame.height)
        diff = screen[visible] - np.asarray([[x, y]], dtype=np.float64)
        dist2 = np.einsum("ij,ij->i", diff, diff)
        visible_indices = np.flatnonzero(visible)
        nearby = dist2 <= (14.0 * 14.0)
        if np.any(nearby):
            candidate_indices = visible_indices[nearby]
            candidate_dist2 = dist2[nearby]
        else:
            order = np.argsort(dist2)
            top_k = order[: min(32, len(order))]
            candidate_indices = visible_indices[top_k]
            candidate_dist2 = dist2[top_k]
        candidate_depth = ndc[candidate_indices, 2]
        best_order = np.lexsort((candidate_dist2, candidate_depth))
        sample_idx = int(candidate_indices[int(best_order[0])])
        if sample_idx in state["picked"]:
            return True
        state["picked"].append(sample_idx)
        refresh_markers()
        print(
            f"Picked surface point {len(state['picked'])}/{required_picks}: "
            f"{[round(float(v), 4) for v in points_m[sample_idx].tolist()]}"
        )
        if len(state["picked"]) >= required_picks:
            finish()
        return True

    def on_mouse(event: Any) -> Any:
        if event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(
            o3d.visualization.gui.MouseButton.LEFT
        ):
            state["last_down"] = (int(event.x), int(event.y))
            return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED
        if event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP:
            last_down = state.get("last_down")
            state["last_down"] = None
            if last_down is None:
                return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED
            dx = int(event.x) - int(last_down[0])
            dy = int(event.y) - int(last_down[1])
            if (dx * dx + dy * dy) > 25:
                return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED
            if try_pick(int(event.x), int(event.y)):
                return o3d.visualization.gui.Widget.EventCallbackResult.HANDLED
        return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED

    scene.set_on_mouse(on_mouse)

    while not state["done"]:
        app.run_one_tick()

    picked = list(dict.fromkeys(int(v) for v in state["picked"]))
    if len(picked) != required_picks:
        raise ValueError(
            f"Expected exactly {required_picks} picked points, got {len(picked)}."
        )
    return picked


def collect_manual_pair_groups(
    mesh: object,
    points_m: np.ndarray,
    normals: np.ndarray,
    point_size: float,
) -> list[list[int]]:
    groups: list[list[int]] = []
    while True:
        groups.append(
            _run_picker(
                mesh,
                points_m,
                normals,
                point_size,
                required_picks=3,
                window_name="CAD Parallel Jaw Grasp Picker",
                description_lines=[
                    "Pick 3 anchors: jaw contact A, jaw contact B, then an approach reference point.",
                    "The approach reference defines the gripper approach direction around the pair.",
                ],
            )
        )
        if not _prompt_continue("finish"):
            return groups


def collect_symmetric_pair_groups(
    mesh: object,
    points_m: np.ndarray,
    normals: np.ndarray,
    point_size: float,
    axis_mode: str,
    seed_mode: str,
    approach_mode: str,
    grasp_count: int,
    sweep_deg: float,
    initial_approach_roll_deg: float,
    face_indices: np.ndarray,
    preview_single_candidate_only: bool = False,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    required_picks = 1 if str(seed_mode or "single").strip().lower() == "single" else 2
    use_reference_pick = str(approach_mode or "axis_roll").strip().lower() == "reference_pick"
    while True:
        pair_axis_mode = str(axis_mode or "auto").strip().lower()
        if pair_axis_mode == "prompt":
            pair_axis_mode = prompt_symmetry_axis("auto")
        print(f"Selected symmetry axis for this pair: {pair_axis_mode}")
        axis_override_indices = None
        if pair_axis_mode == "custom_points":
            axis_override_indices = _run_picker(
                mesh,
                points_m,
                normals,
                point_size,
                required_picks=2,
                window_name="CAD Parallel Jaw Custom Axis Picker",
                description_lines=[
                    "Pick 2 points to define the custom symmetry axis direction.",
                    "The axis will pass through the midpoint of those 2 points.",
                ],
            )
        picked = _run_picker(
            mesh,
            points_m,
            normals,
            point_size,
            required_picks=required_picks,
            window_name="CAD Parallel Jaw Symmetric Picker",
            description_lines=[
                (
                    "Pick 1 seed contact point on the symmetric band. "
                    "The opposite jaw point will be inferred from the chosen symmetry axis."
                    if required_picks == 1
                    else "Pick 2 opposite contact points on the symmetric band."
                ),
                f"Using symmetry axis: {pair_axis_mode}",
            ],
        )
        approach_pick = None
        if use_reference_pick:
            while True:
                approach_pick = _run_picker(
                    mesh,
                    points_m,
                    normals,
                    point_size,
                    required_picks=1,
                    window_name="CAD Parallel Jaw Approach Picker",
                    description_lines=[
                        "Pick 1 approach reference point for the first grasp on this band.",
                        "This point defines the initial gripper approach direction.",
                        "That approach direction will rotate with all generated grasps around the chosen symmetry axis.",
                        "Pick a point away from the jaw-contact line.",
                        f"Using symmetry axis: {pair_axis_mode}",
                    ],
                )[0]
                if int(approach_pick) not in {int(v) for v in picked}:
                    break
                print("Approach reference must be different from the jaw seed point(s). Pick another point.")
        group: dict[str, Any] = {
            "sample_indices": [int(v) for v in picked],
            "approach_reference_index": (
                None if approach_pick is None else int(approach_pick)
            ),
            "axis_mode": pair_axis_mode,
            "axis_override_sample_indices": (
                None if axis_override_indices is None else [int(v) for v in axis_override_indices]
            ),
            "seed_mode": "single" if required_picks == 1 else "pair",
            "approach_mode": "reference_pick" if use_reference_pick else "axis_roll",
            "approach_roll_deg": float(initial_approach_roll_deg),
            "approach_flip": False,
        }
        if not use_reference_pick:
            print("Opening per-pair interactive approach tuning...")

            def rebuild_single_pair_preview(
                roll_deg: float,
                approach_flip: bool,
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                local_group = dict(group)
                local_group["approach_roll_deg"] = float(roll_deg)
                local_group["approach_flip"] = bool(approach_flip)
                pair_candidates_local, axis_records_local = build_symmetric_pair_candidates(
                    groups=[local_group],
                    points_m=points_m,
                    surface_points_m=points_m,
                    grasp_count=grasp_count,
                    sweep_deg=sweep_deg,
                    approach_roll_deg=float(roll_deg),
                )
                projected_local = project_pair_candidates_to_surface(
                    pair_candidates_local,
                    points_m,
                    normals,
                    face_indices,
                )
                return projected_local, axis_records_local

            initial_grasps, initial_axis_records = rebuild_single_pair_preview(
                float(group["approach_roll_deg"]),
                bool(group.get("approach_flip", False)),
            )
            _print_axis_records(initial_axis_records)
            tuned_roll_deg, tuned_approach_flip, _, _ = interactive_axis_roll_preview(
                surface_points_m=points_m,
                anchor_points_m=np.asarray(
                    [points_m[idx] for idx in group.get("sample_indices") or []],
                    dtype=np.float64,
                ),
                point_size=point_size,
                axis_records=initial_axis_records,
                initial_roll_deg=float(group["approach_roll_deg"]),
                initial_approach_flip=bool(group.get("approach_flip", False)),
                build_preview=rebuild_single_pair_preview,
                single_candidate_only=bool(preview_single_candidate_only),
            )
            group["approach_roll_deg"] = float(tuned_roll_deg)
            group["approach_flip"] = bool(tuned_approach_flip)

        groups.append(group)
        if not _prompt_continue("finish"):
            return groups


def _canonicalize_axis(axis: np.ndarray) -> np.ndarray:
    axis_n = _normalize(axis)
    dominant_idx = int(np.argmax(np.abs(axis_n)))
    if float(axis_n[dominant_idx]) < 0.0:
        axis_n = -axis_n
    return axis_n


def _score_symmetry_axis(
    surface_points_m: np.ndarray,
    axis_origin_m: np.ndarray,
    axis_dir: np.ndarray,
) -> float:
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
    return axial_span / max(radial_mad + 0.20 * radial_span + 0.05 * radial_median, 1e-6)


def detect_symmetry_axis(
    surface_points_m: np.ndarray,
    seed_points_m: np.ndarray,
    axis_mode: str,
    custom_axis_points_m: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    axis_origin_m = np.asarray(surface_points_m, dtype=np.float64).mean(axis=0)
    axis_mode = str(axis_mode or "auto").strip().lower()
    if axis_mode == "custom_points":
        custom_points = np.asarray(custom_axis_points_m, dtype=np.float64)
        if custom_points.shape != (2, 3):
            raise ValueError("Custom symmetry axis requires exactly 2 picked points.")
        axis_origin_m = 0.5 * (custom_points[0] + custom_points[1])
        axis_dir = _canonicalize_axis(custom_points[1] - custom_points[0])
        return axis_origin_m, axis_dir, {
            "selected_candidate": "custom_points",
            "score": _score_symmetry_axis(surface_points_m, axis_origin_m, axis_dir),
            "axis_point_a_m": [float(v) for v in custom_points[0].tolist()],
            "axis_point_b_m": [float(v) for v in custom_points[1].tolist()],
        }
    if axis_mode in {"x", "y", "z"}:
        lookup = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        return axis_origin_m, lookup[axis_mode], {
            "selected_candidate": axis_mode,
            "score": _score_symmetry_axis(surface_points_m, axis_origin_m, lookup[axis_mode]),
        }

    centered = np.asarray(surface_points_m, dtype=np.float64) - axis_origin_m[None, :]
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    candidates: list[tuple[str, np.ndarray]] = []
    for rank, eig_idx in enumerate(order, start=1):
        candidates.append((f"mesh_pca_{rank}", _canonicalize_axis(eigvecs[:, eig_idx])))
    if len(seed_points_m) >= 2:
        seed_center = np.asarray(seed_points_m, dtype=np.float64).mean(axis=0, keepdims=True)
        seed_centered = np.asarray(seed_points_m, dtype=np.float64) - seed_center
        seed_cov = np.cov(seed_centered, rowvar=False)
        seed_eigvals, seed_eigvecs = np.linalg.eigh(seed_cov)
        seed_order = np.argsort(seed_eigvals)[::-1]
        candidates.append(("seed_pca_1", _canonicalize_axis(seed_eigvecs[:, seed_order[0]])))
    candidates.extend(
        [
            ("axis_x", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
            ("axis_y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
            ("axis_z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
        ]
    )

    deduped: list[tuple[str, np.ndarray]] = []
    for name, axis_dir in candidates:
        if any(abs(float(np.dot(axis_dir, prev))) >= 0.98 for _, prev in deduped):
            continue
        deduped.append((name, axis_dir))

    scored = [
        {
            "name": name,
            "direction_unit": [float(v) for v in axis_dir.tolist()],
            "score": _score_symmetry_axis(surface_points_m, axis_origin_m, axis_dir),
            "_axis_dir": axis_dir,
        }
        for name, axis_dir in deduped
    ]
    best = max(scored, key=lambda item: float(item["score"]))
    return axis_origin_m, np.asarray(best["_axis_dir"], dtype=np.float64), {
        "selected_candidate": str(best["name"]),
        "score": float(best["score"]),
        "candidates": [{k: v for k, v in item.items() if k != "_axis_dir"} for item in scored],
    }


def rotate_about_axis(vec: np.ndarray, axis_dir: np.ndarray, angle_rad: float) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    axis_n = _normalize(axis_dir)
    cos_a = math.cos(float(angle_rad))
    sin_a = math.sin(float(angle_rad))
    return (
        vec * cos_a
        + np.cross(axis_n, vec) * sin_a
        + axis_n * float(np.dot(axis_n, vec)) * (1.0 - cos_a)
    )


def project_pair_candidates_to_surface(
    pair_candidates: list[dict[str, Any]],
    surface_points_m: np.ndarray,
    surface_normals: np.ndarray,
    face_indices: np.ndarray,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for pair in pair_candidates:
        point_a = np.asarray(pair["contact_a_local_m"], dtype=np.float64)
        point_b = np.asarray(pair["contact_b_local_m"], dtype=np.float64)
        diff_a = surface_points_m - point_a[None, :]
        diff_b = surface_points_m - point_b[None, :]
        sample_a = int(np.argmin(np.einsum("ij,ij->i", diff_a, diff_a)))
        sample_b = int(np.argmin(np.einsum("ij,ij->i", diff_b, diff_b)))
        if sample_a == sample_b:
            continue
        # Keep contact order. In symmetric-ring generation from a single seed point,
        # theta and theta+pi can project to the same unordered surface pair with the
        # contacts swapped; sorting here incorrectly collapses the full 360 sweep to 180.
        pair_key = (sample_a, sample_b)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        proj_a = np.asarray(surface_points_m[sample_a], dtype=np.float64)
        proj_b = np.asarray(surface_points_m[sample_b], dtype=np.float64)
        jaw_axis = _normalize(proj_b - proj_a)
        approach_axis_raw = np.asarray(pair["approach_axis_local"], dtype=np.float64)
        approach_axis_proj = approach_axis_raw - np.dot(approach_axis_raw, jaw_axis) * jaw_axis
        try:
            approach_axis = _normalize(approach_axis_proj)
        except ValueError:
            approach_axis = _normalize(approach_axis_raw)
        normal_a = np.asarray(surface_normals[sample_a], dtype=np.float64)
        normal_b = np.asarray(surface_normals[sample_b], dtype=np.float64)
        grasp_family_label = infer_grasp_family_label(normal_a, normal_b, jaw_axis)
        center = 0.5 * (proj_a + proj_b)
        results.append(
            {
                "id": str(pair["id"]),
                "label": str(pair["label"]),
                "grasp_type": "parallel_jaw_pair",
                "contact_a_local_m": proj_a.tolist(),
                "contact_b_local_m": proj_b.tolist(),
                "normal_a_local": normal_a.tolist(),
                "normal_b_local": normal_b.tolist(),
                "center_local_m": center.tolist(),
                "jaw_axis_local": jaw_axis.tolist(),
                "approach_axis_local": approach_axis.tolist(),
                "opening_width_m": float(np.linalg.norm(proj_b - proj_a)),
                "face_indices": [int(face_indices[sample_a]), int(face_indices[sample_b])],
                "sample_indices": [sample_a, sample_b],
                "generator_group_index": int(pair.get("generator_group_index", 0) or 0),
                "grasp_family_label": grasp_family_label,
                "enabled": True,
            }
        )
    if not results:
        raise ValueError("No valid parallel-jaw grasp pairs could be projected to the surface.")
    return results


def build_manual_pair_candidates(
    groups: list[list[int]],
    points_m: np.ndarray,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for order, group in enumerate(groups, start=1):
        contact_a = np.asarray(points_m[group[0]], dtype=np.float64)
        contact_b = np.asarray(points_m[group[1]], dtype=np.float64)
        center = 0.5 * (contact_a + contact_b)
        jaw_axis = _normalize(contact_b - contact_a)
        approach_seed = np.asarray(points_m[group[2]], dtype=np.float64)
        approach_vec = approach_seed - center
        approach_vec = approach_vec - np.dot(approach_vec, jaw_axis) * jaw_axis
        approach_axis = _normalize(approach_vec)
        pairs.append(
            {
                "id": f"grasp_{order:03d}",
                "label": f"grasp_{order:03d}",
                "contact_a_local_m": contact_a.tolist(),
                "contact_b_local_m": contact_b.tolist(),
                "approach_axis_local": approach_axis.tolist(),
                "generator_group_index": int(order),
            }
        )
    return pairs


def build_symmetric_pair_candidates(
    groups: list[dict[str, Any]],
    points_m: np.ndarray,
    surface_points_m: np.ndarray,
    grasp_count: int,
    sweep_deg: float,
    approach_roll_deg: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    axis_records: list[dict[str, Any]] = []
    sweep_rad = math.radians(float(sweep_deg))
    full_circle = abs(abs(float(sweep_deg)) - 360.0) <= 1e-6
    grasp_count_int = max(1, int(grasp_count))
    if full_circle:
        angle_step = math.tau / grasp_count_int
        sign = -1.0 if float(sweep_deg) < 0.0 else 1.0
        angles = np.arange(grasp_count_int, dtype=np.float64) * (angle_step * sign)
    elif grasp_count_int == 1:
        angles = np.asarray([0.0], dtype=np.float64)
    else:
        angle_step = sweep_rad / float(grasp_count_int - 1)
        angles = np.arange(grasp_count_int, dtype=np.float64) * angle_step
    counter = 1
    for group_index, group in enumerate(groups, start=1):
        sample_indices = [int(v) for v in group.get("sample_indices") or []]
        if len(sample_indices) not in {1, 2}:
            continue
        seed_points = np.asarray([points_m[idx] for idx in sample_indices], dtype=np.float64)
        axis_mode = str(group.get("axis_mode") or "auto").strip().lower()
        custom_axis_points_m = None
        axis_override_indices = group.get("axis_override_sample_indices") or []
        if axis_mode == "custom_points" and len(axis_override_indices) >= 2:
            custom_axis_points_m = np.asarray(
                [points_m[int(axis_override_indices[0])], points_m[int(axis_override_indices[1])]],
                dtype=np.float64,
            )
        axis_origin_m, axis_dir, axis_stats = detect_symmetry_axis(
            surface_points_m=surface_points_m,
            seed_points_m=seed_points,
            axis_mode=axis_mode,
            custom_axis_points_m=custom_axis_points_m,
        )
        axis_origin = np.asarray(axis_origin_m, dtype=np.float64)
        axis_n = _normalize(axis_dir)
        axis_records.append(
            {
                "group_index": int(group_index),
                "requested_axis_mode": axis_mode,
                "sample_indices": sample_indices,
                "axis_override_sample_indices": [int(v) for v in axis_override_indices],
                "approach_reference_index": int(group.get("approach_reference_index"))
                if group.get("approach_reference_index") is not None
                else None,
                "seed_mode": str(group.get("seed_mode") or ("single" if len(sample_indices) == 1 else "pair")),
                "approach_mode": str(group.get("approach_mode") or "axis_roll"),
                "approach_roll_deg": float(approach_roll_deg),
                "axis_origin_m": [float(v) for v in axis_origin.tolist()],
                "axis_direction": [float(v) for v in axis_n.tolist()],
                "axis_stats": axis_stats,
            }
        )
        point_a = np.asarray(points_m[sample_indices[0]], dtype=np.float64)
        if len(sample_indices) == 1:
            axial_coord = float(np.dot(point_a - axis_origin, axis_n))
            axis_point_seed = axis_origin + axis_n * axial_coord
            radial_a_seed = point_a - axis_point_seed
            if float(np.linalg.norm(radial_a_seed)) <= 1e-6:
                continue
            point_b = axis_point_seed - radial_a_seed
        else:
            point_b = np.asarray(points_m[sample_indices[1]], dtype=np.float64)
        pair_mid = 0.5 * (point_a + point_b)
        axial_coord = float(np.dot(pair_mid - axis_origin, axis_n))
        axis_point = axis_origin + axis_n * axial_coord
        radial_a = point_a - axis_point
        radial_b = point_b - axis_point
        jaw_axis_seed = _normalize(point_b - point_a)
        approach_ref_idx = group.get("approach_reference_index")
        approach_mode = str(group.get("approach_mode") or "axis_roll").strip().lower()
        group_approach_roll_rad = math.radians(
            float(group.get("approach_roll_deg", approach_roll_deg))
        )
        approach_flip = bool(group.get("approach_flip", False))
        if approach_mode != "reference_pick" or approach_ref_idx is None:
            axis_projected = axis_n - np.dot(axis_n, jaw_axis_seed) * jaw_axis_seed
            if float(np.linalg.norm(axis_projected)) <= 1e-6:
                fallback_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                if abs(float(np.dot(fallback_axis, jaw_axis_seed))) >= 0.95:
                    fallback_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                axis_projected = fallback_axis - np.dot(fallback_axis, jaw_axis_seed) * jaw_axis_seed
            finger_axis_seed = _normalize(axis_projected)
            if abs(group_approach_roll_rad) > 1e-9:
                finger_axis_seed = _normalize(
                    rotate_about_axis(finger_axis_seed, jaw_axis_seed, group_approach_roll_rad)
                )
            approach_axis_seed = _normalize(np.cross(finger_axis_seed, jaw_axis_seed))
        else:
            approach_ref = np.asarray(points_m[int(approach_ref_idx)], dtype=np.float64)
            approach_seed = approach_ref - pair_mid
            approach_seed = approach_seed - np.dot(approach_seed, jaw_axis_seed) * jaw_axis_seed
            if float(np.linalg.norm(approach_seed)) <= 1e-6:
                raise ValueError(
                    f"Approach reference for symmetric pair {group_index} is collinear with the jaw axis. "
                    "Pick a point away from the jaw-contact line."
                )
            approach_axis_seed = _normalize(approach_seed)
        if approach_flip:
            approach_axis_seed = -approach_axis_seed
        for angle in angles:
            rot_a = axis_point + rotate_about_axis(radial_a, axis_n, float(angle))
            rot_b = axis_point + rotate_about_axis(radial_b, axis_n, float(angle))
            rot_approach = rotate_about_axis(approach_axis_seed, axis_n, float(angle))
            pairs.append(
                {
                    "id": f"grasp_{counter:03d}",
                    "label": f"grasp_{counter:03d}",
                    "contact_a_local_m": rot_a.tolist(),
                    "contact_b_local_m": rot_b.tolist(),
                    "approach_axis_local": rot_approach.tolist(),
                    "generator_group_index": int(group_index),
                    "seed_mode": str(group.get("seed_mode") or ("single" if len(sample_indices) == 1 else "pair")),
                }
            )
            counter += 1
    return pairs, axis_records


def build_payload(
    *,
    object_folder: Optional[Path],
    mesh_path: Path,
    label: str,
    mesh_units: str,
    mesh_scale: float,
    center_mesh_units: np.ndarray,
    tool_axis: str,
    jaw_axis: str,
    selection_mode: str,
    mode: str,
    grasps: list[dict[str, Any]],
    generator: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "vgr_grasp_candidates/v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "object_label": label,
        "object_folder": str(object_folder) if object_folder else None,
        "source_mesh_path": str(mesh_path),
        "origin_mode": "mesh_aabb_center",
        "origin_adjustment_mesh_units": [float(v) for v in center_mesh_units.tolist()],
        "mesh_units": mesh_units,
        "mesh_scale": float(mesh_scale),
        "default_gripper_type": "parallel_jaw",
        "selection_mode": str(selection_mode).strip().lower() or "closest_contact_to_camera",
        "tool_axis": str(tool_axis).strip().lower() or "-z",
        "jaw_axis": str(jaw_axis).strip().lower() or "x",
        "approach_axis_convention": "pregrasp_to_center",
        "grasp_family_label_source": "surface_normal_inference",
        "generator": {
            "type": "parallel_jaw",
            "mode": mode,
            **generator,
        },
        "grasps": grasps,
    }


def build_grasp_family_labels_payload(
    *,
    output_path: Path,
    label: str,
    grasps: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    label_source = str(
        payload.get("grasp_family_label_source") or "surface_normal_inference"
    )
    grouped: dict[int, list[str]] = {}
    for grasp in grasps:
        group_index = int(grasp.get("generator_group_index") or 0)
        family_label = str(grasp.get("grasp_family_label") or "").strip().lower()
        if group_index <= 0 or family_label not in {"internal", "external"}:
            continue
        grouped.setdefault(group_index, []).append(family_label)

    label_map: dict[str, str] = {}
    group_summaries: dict[str, dict[str, Any]] = {}
    for group_index in sorted(grouped):
        entries = grouped[group_index]
        internal_count = sum(1 for value in entries if value == "internal")
        external_count = sum(1 for value in entries if value == "external")
        family_label = "internal" if internal_count > external_count else "external"
        label_map[str(group_index)] = family_label
        group_grasps = [
            grasp
            for grasp in grasps
            if int(grasp.get("generator_group_index") or 0) == group_index
        ]
        widths = [float(grasp.get("opening_width_m") or 0.0) for grasp in group_grasps]
        centers_y = [
            float((grasp.get("center_local_m") or [0.0, 0.0, 0.0])[1])
            for grasp in group_grasps
        ]
        group_summaries[str(group_index)] = {
            "group_index": int(group_index),
            "current_label": family_label,
            "label_source": label_source,
            "count": len(group_grasps),
            "internal_count": internal_count,
            "external_count": external_count,
            "mean_opening_width_m": float(np.mean(widths)) if widths else 0.0,
            "mean_center_y_m": float(np.mean(centers_y)) if centers_y else 0.0,
        }

    return {
        "object_name": label,
        "contacts_path": str(output_path),
        "labels": label_map,
        "label_source": label_source,
        "group_summaries": group_summaries,
        "pick_contacts_format": payload.get("format"),
    }


def apply_symmetric_group_family_labels(
    grasps: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> None:
    """Use the interactive approach direction as the inner/outer label source.

    In this picker, `F` toggles the approach direction shown to the user as
    inward/outward. The intended grasp family follows that UI choice:
    inward -> internal, outward -> external.
    """
    label_by_group = {
        int(index): ("internal" if bool(group.get("approach_flip", False)) else "external")
        for index, group in enumerate(groups, start=1)
    }
    for grasp in grasps:
        group_index = int(grasp.get("generator_group_index") or 0)
        family_label = label_by_group.get(group_index)
        if family_label:
            grasp["grasp_family_label"] = family_label


def _build_grasp_preview_lines(
    grasps: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_points = []
    pair_lines = []
    pair_colors = []
    arrow_points = []
    arrow_lines = []
    arrow_colors = []
    marker_points = []
    marker_colors = []
    highlighted_groups: set[int] = set()
    for grasp_index, grasp in enumerate(grasps):
        contact_a = np.asarray(grasp["contact_a_local_m"], dtype=np.float64)
        contact_b = np.asarray(grasp["contact_b_local_m"], dtype=np.float64)
        center = np.asarray(grasp["center_local_m"], dtype=np.float64)
        approach_axis = _normalize(np.asarray(grasp["approach_axis_local"], dtype=np.float64))
        jaw_vector = contact_b - contact_a
        jaw_width = float(np.linalg.norm(jaw_vector))
        approach_length = max(0.025, jaw_width * 0.75)
        if approach_axis is None:
            continue
        group_index = int(grasp.get("generator_group_index") or 0)
        highlight_key = group_index if group_index > 0 else -(grasp_index + 1)
        is_seed_candidate = highlight_key not in highlighted_groups
        highlighted_groups.add(highlight_key)
        pair_color = [0.92, 0.12, 0.72] if is_seed_candidate else [0.16, 0.44, 0.86]
        approach_color = [1.0, 0.80, 0.10] if is_seed_candidate else [0.92, 0.94, 0.98]
        contact_color = [0.92, 0.12, 0.72] if is_seed_candidate else [0.95, 0.26, 0.18]
        center_color = [0.10, 0.92, 0.95] if is_seed_candidate else [1.0, 0.86, 0.18]
        approach_start = center - (approach_axis * approach_length)
        start_idx = len(pair_points)
        pair_points.append(contact_a.tolist())
        pair_points.append(contact_b.tolist())
        pair_lines.append([start_idx, start_idx + 1])
        pair_colors.append(pair_color)
        arrow_start = len(arrow_points)
        arrow_points.append(approach_start.tolist())
        arrow_points.append(center.tolist())
        arrow_lines.append([arrow_start, arrow_start + 1])
        arrow_colors.append(approach_color)
        marker_points.extend(
            [
                contact_a.tolist(),
                contact_b.tolist(),
                center.tolist(),
            ]
        )
        marker_colors.extend(
            [
                contact_color,
                contact_color,
                center_color,
            ]
        )
    return (
        np.asarray(pair_points, dtype=np.float64),
        np.asarray(pair_lines, dtype=np.int32),
        np.asarray(pair_colors, dtype=np.float64),
        np.asarray(arrow_points, dtype=np.float64),
        np.asarray(arrow_lines, dtype=np.int32),
        np.asarray(arrow_colors, dtype=np.float64),
        np.asarray(marker_points, dtype=np.float64),
        np.asarray(marker_colors, dtype=np.float64),
    )


def _preview_grasps_subset(
    grasps: list[dict[str, Any]],
    single_candidate_only: bool,
) -> list[dict[str, Any]]:
    if not single_candidate_only:
        return list(grasps)
    return list(grasps[:1])


def _set_line_set_data(
    line_set: Any,
    points: np.ndarray,
    lines: np.ndarray,
    colors: np.ndarray,
) -> None:
    import open3d as o3d

    if points.size == 0:
        points = np.zeros((0, 3), dtype=np.float64)
    if lines.size == 0:
        lines = np.zeros((0, 2), dtype=np.int32)
    if colors.size == 0:
        colors = np.zeros((0, 3), dtype=np.float64)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)


def _print_axis_records(axis_records: list[dict[str, Any]]) -> None:
    for record in axis_records:
        print(f"Pair {record['group_index']} symmetry axis:")
        print(f"  requested = {record['requested_axis_mode']}")
        print(f"  origin_m = {[round(float(v), 6) for v in record['axis_origin_m']]}")
        print(f"  direction = {[round(float(v), 6) for v in record['axis_direction']]}")
        print(
            f"  score = {round(float((record.get('axis_stats') or {}).get('score', 0.0)), 3)} "
            f"({(record.get('axis_stats') or {}).get('selected_candidate')})"
        )


def interactive_axis_roll_preview(
    *,
    surface_points_m: np.ndarray,
    anchor_points_m: np.ndarray,
    point_size: float,
    axis_records: list[dict[str, Any]],
    initial_roll_deg: float,
    initial_approach_flip: bool,
    build_preview: Callable[[float, bool], tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    single_candidate_only: bool = False,
) -> tuple[float, bool, list[dict[str, Any]], list[dict[str, Any]]]:
    import open3d as o3d

    initial_grasps, initial_axis_records = build_preview(float(initial_roll_deg), bool(initial_approach_flip))
    state: dict[str, Any] = {
        "roll_deg": float(initial_roll_deg),
        "approach_flip": bool(initial_approach_flip),
        "grasps": list(initial_grasps),
        "axis_records": list(initial_axis_records or axis_records),
    }

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(surface_points_m)
    cloud.paint_uniform_color([0.72, 0.78, 0.84])

    anchor_cloud = o3d.geometry.PointCloud()
    anchor_cloud.points = o3d.utility.Vector3dVector(anchor_points_m)
    anchor_cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray([[0.14, 0.62, 0.22]], dtype=np.float64), (len(anchor_points_m), 1))
    )

    extent = float(np.linalg.norm(surface_points_m.max(axis=0) - surface_points_m.min(axis=0)))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.03, extent * 0.18))

    axis_lines = None
    if state["axis_records"]:
        axis_points = []
        axis_indices = []
        axis_colors = []
        axis_length = max(0.05, extent * 0.35)
        for record in state["axis_records"]:
            axis_origin = np.asarray(record.get("axis_origin_m") or [], dtype=np.float64)
            axis_dir = np.asarray(record.get("axis_direction") or [], dtype=np.float64)
            if axis_origin.shape != (3,) or axis_dir.shape != (3,):
                continue
            axis_dir_n = _normalize(axis_dir)
            start_idx = len(axis_points)
            axis_points.append((axis_origin - axis_dir_n * (axis_length * 0.5)).tolist())
            axis_points.append((axis_origin + axis_dir_n * (axis_length * 0.5)).tolist())
            axis_indices.append([start_idx, start_idx + 1])
            axis_colors.append([0.18, 0.52, 0.96])
        if axis_points:
            axis_lines = o3d.geometry.LineSet()
            _set_line_set_data(
                axis_lines,
                np.asarray(axis_points, dtype=np.float64),
                np.asarray(axis_indices, dtype=np.int32),
                np.asarray(axis_colors, dtype=np.float64),
            )

    window = app.create_window("Parallel Jaw CAD Grasps", 1600, 1000)
    scene = o3d.visualization.gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    scene.set_view_controls(o3d.visualization.gui.SceneWidget.Controls.ROTATE_CAMERA)
    window.add_child(scene)

    def on_layout(_layout_context: Any) -> None:
        scene.frame = window.content_rect

    window.set_on_layout(on_layout)

    cloud_mat = o3d.visualization.rendering.MaterialRecord()
    cloud_mat.shader = "defaultUnlit"
    cloud_mat.point_size = float(point_size)

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 2.5

    marker_mat = o3d.visualization.rendering.MaterialRecord()
    marker_mat.shader = "defaultUnlit"
    marker_mat.point_size = max(10.0, float(point_size) * 2.5)

    mesh_mat = o3d.visualization.rendering.MaterialRecord()
    mesh_mat.shader = "defaultLit"

    state["done"] = False

    def refresh_geometry() -> None:
        grasps, latest_axis_records = build_preview(
            float(state["roll_deg"]),
            bool(state["approach_flip"]),
        )
        state["grasps"] = grasps
        state["axis_records"] = latest_axis_records
        preview_grasps = _preview_grasps_subset(grasps, single_candidate_only)
        (
            pair_points,
            pair_lines,
            pair_colors,
            arrow_points,
            arrow_lines,
            arrow_colors,
            marker_points,
            marker_colors,
        ) = _build_grasp_preview_lines(preview_grasps)
        pair_line_set = o3d.geometry.LineSet()
        arrow_line_set = o3d.geometry.LineSet()
        grasp_marker_cloud = o3d.geometry.PointCloud()
        _set_line_set_data(pair_line_set, pair_points, pair_lines, pair_colors)
        _set_line_set_data(arrow_line_set, arrow_points, arrow_lines, arrow_colors)
        grasp_marker_cloud.points = o3d.utility.Vector3dVector(marker_points)
        grasp_marker_cloud.colors = o3d.utility.Vector3dVector(marker_colors)
        for name in ("preview_cloud", "preview_anchor_cloud", "preview_pair_lines", "preview_arrow_lines", "preview_markers", "preview_frame", "preview_axis_lines"):
            if scene.scene.has_geometry(name):
                scene.scene.remove_geometry(name)
        scene.scene.add_geometry("preview_cloud", cloud, cloud_mat)
        scene.scene.add_geometry("preview_anchor_cloud", anchor_cloud, cloud_mat)
        scene.scene.add_geometry("preview_pair_lines", pair_line_set, line_mat)
        scene.scene.add_geometry("preview_arrow_lines", arrow_line_set, line_mat)
        scene.scene.add_geometry("preview_markers", grasp_marker_cloud, marker_mat)
        scene.scene.add_geometry("preview_frame", frame, mesh_mat)
        if axis_lines is not None:
            scene.scene.add_geometry("preview_axis_lines", axis_lines, line_mat)
        print(
            f"Approach roll: {float(state['roll_deg']):.1f} deg | "
            f"direction: {'inward' if bool(state['approach_flip']) else 'outward'}"
        )

    def make_roll_callback(delta_deg: float):
        def _callback() -> None:
            state["roll_deg"] = float(state["roll_deg"]) + float(delta_deg)
            refresh_geometry()

        return _callback

    def toggle_flip_callback() -> None:
        state["approach_flip"] = not bool(state["approach_flip"])
        refresh_geometry()

    def finish() -> None:
        state["done"] = True
        window.close()

    def on_key(event: Any) -> bool:
        if event.type != o3d.visualization.gui.KeyEvent.Type.DOWN:
            return False
        if event.key == o3d.visualization.gui.KeyName.A:
            make_roll_callback(-5.0)()
            return True
        if event.key == o3d.visualization.gui.KeyName.D:
            make_roll_callback(5.0)()
            return True
        if event.key == o3d.visualization.gui.KeyName.Z:
            make_roll_callback(-1.0)()
            return True
        if event.key == o3d.visualization.gui.KeyName.C:
            make_roll_callback(1.0)()
            return True
        if event.key == o3d.visualization.gui.KeyName.F:
            toggle_flip_callback()
            return True
        if event.key in {o3d.visualization.gui.KeyName.ESCAPE, o3d.visualization.gui.KeyName.ENTER, o3d.visualization.gui.KeyName.Q}:
            finish()
            return True
        return False

    print("Interactive approach-vector tuning:")
    print("  A : rotate approach roll -5 deg")
    print("  D : rotate approach roll +5 deg")
    print("  Z : rotate approach roll -1 deg")
    print("  C : rotate approach roll +1 deg")
    print("  F : flip approach direction inward/outward")
    print("  Q or Escape : close preview and keep the current roll")
    window.set_on_key(on_key)

    bbox_min = np.asarray(frame.get_axis_aligned_bounding_box().min_bound, dtype=np.float64)
    bbox_max = np.asarray(frame.get_axis_aligned_bounding_box().max_bound, dtype=np.float64)
    bbox_min = np.minimum(bbox_min, np.asarray(surface_points_m, dtype=np.float64).min(axis=0))
    bbox_max = np.maximum(bbox_max, np.asarray(surface_points_m, dtype=np.float64).max(axis=0))
    bbox_extent = bbox_max - bbox_min
    padding = np.maximum(bbox_extent * 0.35, 0.02)
    padded_bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min - padding, bbox_max + padding)
    scene.setup_camera(60.0, padded_bbox, padded_bbox.get_center())
    scene.scene.set_background([0.98, 0.98, 0.99, 1.0])
    _set_closeup_friendly_projection(scene, bbox_extent)
    refresh_geometry()
    while not state["done"]:
        app.run_one_tick()
    return (
        float(state["roll_deg"]),
        bool(state["approach_flip"]),
        list(state["grasps"]),
        list(state["axis_records"]),
    )


def preview_grasps(
    surface_points_m: np.ndarray,
    anchor_points_m: np.ndarray,
    grasps: list[dict[str, Any]],
    point_size: float,
    approach_reference_points_m: Optional[np.ndarray] = None,
    axis_records: Optional[list[dict[str, Any]]] = None,
    single_candidate_only: bool = False,
) -> None:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(surface_points_m)
    cloud.paint_uniform_color([0.72, 0.78, 0.84])

    anchor_cloud = o3d.geometry.PointCloud()
    anchor_cloud.points = o3d.utility.Vector3dVector(anchor_points_m)
    anchor_cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray([[0.14, 0.62, 0.22]], dtype=np.float64), (len(anchor_points_m), 1))
    )

    approach_reference_cloud = None
    if approach_reference_points_m is not None and len(approach_reference_points_m):
        approach_reference_cloud = o3d.geometry.PointCloud()
        approach_reference_cloud.points = o3d.utility.Vector3dVector(approach_reference_points_m)
        approach_reference_cloud.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray([[0.98, 0.78, 0.14]], dtype=np.float64), (len(approach_reference_points_m), 1))
        )

    grasps = _preview_grasps_subset(grasps, single_candidate_only)

    pair_line_set = o3d.geometry.LineSet()
    arrow_line_set = o3d.geometry.LineSet()
    grasp_marker_cloud = o3d.geometry.PointCloud()
    (
        pair_points,
        pair_lines,
        pair_colors,
        arrow_points,
        arrow_lines,
        arrow_colors,
        marker_points,
        marker_colors,
    ) = _build_grasp_preview_lines(grasps)
    _set_line_set_data(pair_line_set, pair_points, pair_lines, pair_colors)
    _set_line_set_data(arrow_line_set, arrow_points, arrow_lines, arrow_colors)
    grasp_marker_cloud.points = o3d.utility.Vector3dVector(marker_points)
    grasp_marker_cloud.colors = o3d.utility.Vector3dVector(marker_colors)

    extent = float(np.linalg.norm(surface_points_m.max(axis=0) - surface_points_m.min(axis=0)))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.03, extent * 0.18))

    geometries = [cloud, anchor_cloud, pair_line_set, arrow_line_set, grasp_marker_cloud, frame]
    if approach_reference_cloud is not None:
        geometries.insert(2, approach_reference_cloud)
    if axis_records:
        axis_points = []
        axis_indices = []
        axis_colors = []
        axis_length = max(0.05, extent * 0.35)
        for record in axis_records:
            axis_origin = np.asarray(record.get("axis_origin_m") or [], dtype=np.float64)
            axis_dir = np.asarray(record.get("axis_direction") or [], dtype=np.float64)
            if axis_origin.shape != (3,) or axis_dir.shape != (3,):
                continue
            axis_dir_n = _normalize(axis_dir)
            start_idx = len(axis_points)
            axis_points.append((axis_origin - axis_dir_n * (axis_length * 0.5)).tolist())
            axis_points.append((axis_origin + axis_dir_n * (axis_length * 0.5)).tolist())
            axis_indices.append([start_idx, start_idx + 1])
            axis_colors.append([0.18, 0.52, 0.96])
        if axis_points:
            axis_lines = o3d.geometry.LineSet()
            axis_lines.points = o3d.utility.Vector3dVector(np.asarray(axis_points, dtype=np.float64))
            axis_lines.lines = o3d.utility.Vector2iVector(np.asarray(axis_indices, dtype=np.int32))
            axis_lines.colors = o3d.utility.Vector3dVector(np.asarray(axis_colors, dtype=np.float64))
            geometries.append(axis_lines)

    print("Preview window:")
    print("  gray  : sampled mesh surface")
    print("  green : picked anchors")
    if approach_reference_cloud is not None:
        print("  yellow: picked approach reference points")
    print("  magenta/yellow: seed candidate for each picked group")
    print("  blue/red      : other candidate contact spans + contacts")
    print("  white/gray    : other candidate approach directions")
    print("  yellow: grasp center marker")
    if axis_records:
        print("  blue  : detected symmetry axis")
    print("  Q     : close preview")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Parallel Jaw CAD Grasps", width=1600, height=1000)
    for geom in geometries:
        vis.add_geometry(geom)
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.98, 0.98, 0.99], dtype=np.float64)
    render_option.point_size = float(point_size)
    render_option.line_width = 2.5
    vis.run()
    vis.destroy_window()


def main() -> None:
    args = parse_args()
    object_folder, mesh_path, output_path, label = resolve_inputs(args)
    mesh, center_mesh_units, _scale_to_m = load_centered_mesh(
        mesh_path=mesh_path,
        mesh_units=args.mesh_units,
        mesh_scale=args.mesh_scale,
    )
    surface_points_m, surface_normals, face_indices = sample_surface_points(
        mesh,
        sample_count=args.surface_sample_count,
    )

    if args.mode == "manual_pairs":
        groups = collect_manual_pair_groups(
            mesh, surface_points_m, surface_normals, args.point_size
        )
        pair_candidates = build_manual_pair_candidates(groups, surface_points_m)
        projected_grasps = project_pair_candidates_to_surface(
            pair_candidates, surface_points_m, surface_normals, face_indices
        )
        generator = {
            "anchor_sample_index_groups": [[int(v) for v in group] for group in groups],
            "pair_count": len(projected_grasps),
        }
        axis_origin_m = None
        axis_dir = None
    else:
        chosen_axis = str(args.axis or "prompt").strip().lower()
        print("Symmetric ring mode axis colors:")
        print("  X axis = red")
        print("  Y axis = green")
        print("  Z axis = blue")
        if chosen_axis == "prompt":
            show_symmetry_axis_guide(surface_points_m, args.point_size)
        elif chosen_axis not in {"auto", "x", "y", "z"}:
            chosen_axis = "auto"
        groups = collect_symmetric_pair_groups(
            mesh,
            surface_points_m,
            surface_normals,
            args.point_size,
            chosen_axis,
            args.symmetric_seed_mode,
            args.symmetric_approach_mode,
            args.grasp_count,
            args.sweep_deg,
            args.approach_roll_deg,
            face_indices,
            args.preview_single_candidate_only,
        )
        def rebuild_symmetric_preview() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            pair_candidates_local, axis_records_local = build_symmetric_pair_candidates(
                groups=groups,
                points_m=surface_points_m,
                surface_points_m=surface_points_m,
                grasp_count=args.grasp_count,
                sweep_deg=args.sweep_deg,
                approach_roll_deg=float(args.approach_roll_deg),
            )
            projected_local = project_pair_candidates_to_surface(
                pair_candidates_local, surface_points_m, surface_normals, face_indices
            )
            return projected_local, axis_records_local

        projected_grasps, axis_records = rebuild_symmetric_preview()
        apply_symmetric_group_family_labels(projected_grasps, groups)
        _print_axis_records(axis_records)
        preview_grasps(
            surface_points_m=surface_points_m,
            anchor_points_m=np.asarray(
                [
                    surface_points_m[idx]
                    for group in groups
                    for idx in (group.get("sample_indices") or [])
                ],
                dtype=np.float64,
            ),
            grasps=projected_grasps,
            point_size=args.point_size,
            approach_reference_points_m=(
                np.asarray(
                    [
                        surface_points_m[int(group.get("approach_reference_index"))]
                        for group in groups
                        if group.get("approach_reference_index") is not None
                    ],
                    dtype=np.float64,
                )
                if str(args.symmetric_approach_mode).strip().lower() == "reference_pick"
                else None
            ),
            axis_records=axis_records,
            single_candidate_only=False,
        )

        generator = {
            "anchor_sample_index_groups": [
                [int(v) for v in (group.get("sample_indices") or [])]
                for group in groups
            ],
            "axis_override_sample_index_groups": [
                [int(v) for v in (group.get("axis_override_sample_indices") or [])]
                for group in groups
                if group.get("axis_override_sample_indices")
            ],
            "approach_reference_sample_indices": [
                int(group.get("approach_reference_index"))
                for group in groups
                if group.get("approach_reference_index") is not None
            ],
            "symmetric_seed_mode": str(args.symmetric_seed_mode),
            "symmetric_approach_mode": str(args.symmetric_approach_mode),
            "approach_roll_deg": float(args.approach_roll_deg),
            "approach_roll_deg_per_group": [
                float(group.get("approach_roll_deg", args.approach_roll_deg))
                for group in groups
            ],
            "approach_flip_per_group": [
                bool(group.get("approach_flip", False))
                for group in groups
            ],
            "sweep_deg": float(args.sweep_deg),
            "grasp_count_per_ring": int(args.grasp_count),
            "requested_symmetry_axis_mode": chosen_axis,
            "pair_axis_records": axis_records,
        }

    payload = build_payload(
        object_folder=object_folder,
        mesh_path=mesh_path,
        label=label,
        mesh_units=args.mesh_units,
        mesh_scale=args.mesh_scale,
        center_mesh_units=center_mesh_units,
        tool_axis=args.tool_axis,
        jaw_axis=args.jaw_axis,
        selection_mode=args.selection_mode,
        mode=args.mode,
        grasps=projected_grasps,
        generator=generator,
    )
    if args.mode == "symmetric_ring":
        payload["grasp_family_label_source"] = "approach_direction_inward_internal"

    if args.mode == "manual_pairs":
        preview_grasps(
            surface_points_m=surface_points_m,
            anchor_points_m=np.asarray(
                [surface_points_m[idx] for group in groups for idx in group],
                dtype=np.float64,
            ),
            grasps=projected_grasps,
            point_size=args.point_size,
            approach_reference_points_m=None,
            axis_records=None,
            single_candidate_only=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    labels_path = output_path.with_name("grasp_family_labels.json")
    labels_payload = build_grasp_family_labels_payload(
        output_path=output_path,
        label=label,
        grasps=projected_grasps,
        payload=payload,
    )
    labels_path.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")
    print(f"Saved {len(projected_grasps)} parallel-jaw grasps to {output_path}")
    print(f"Saved grasp family labels to {labels_path}")
    print(f"Mesh: {mesh_path}")
    print("This file is directly usable by VGR bin picking as pick_contacts.json.")


if __name__ == "__main__":
    main()
