from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


AXIS_LEN = 0.08
SPHERE_RADIUS = 0.006


def create_line(p1: np.ndarray, p2: np.ndarray, color: list[float]) -> o3d.geometry.LineSet:
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([p1, p2])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([color])
    return line


def create_axes(
    origin: np.ndarray, rotation: np.ndarray | None = None, scale: float = AXIS_LEN
) -> list[o3d.geometry.LineSet]:
    rot = np.eye(3) if rotation is None else rotation
    axes = []
    axes.append(create_line(origin, origin + rot @ np.array([scale, 0.0, 0.0]), [1, 0, 0]))
    axes.append(create_line(origin, origin + rot @ np.array([0.0, scale, 0.0]), [0, 1, 0]))
    axes.append(create_line(origin, origin + rot @ np.array([0.0, 0.0, scale]), [0, 0, 1]))
    return axes


def create_sphere(origin: np.ndarray, color: list[float]) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
    sphere.paint_uniform_color(color)
    sphere.translate(origin)
    return sphere


def find_latest_handeye_file(folder: Path) -> Path:
    candidates = sorted(
        folder.glob("handeye*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No handeye*.json found in {folder}")
    return candidates[0]


def load_camera_in_gripper(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = str(payload.get("hand_eye_frame") or payload.get("frame") or "").strip().lower()

    if frame == "camera_in_gripper":
        translation = payload.get("translation_m")
        quat = payload.get("rotation_quat_xyzw")
    elif isinstance(payload.get("handeye_result"), dict) and isinstance(
        payload["handeye_result"].get("camera_in_gripper"), dict
    ):
        translation = payload["handeye_result"]["camera_in_gripper"].get("translation_m")
        quat = payload["handeye_result"]["camera_in_gripper"].get("rotation_quat_xyzw")
        frame = "camera_in_gripper"
    else:
        raise ValueError(
            f"{path.name} does not contain a camera_in_gripper transform that this viewer can use"
        )

    if not (isinstance(translation, list) and len(translation) >= 3):
        raise ValueError(f"{path.name} is missing translation_m")
    if not (isinstance(quat, list) and len(quat) >= 4):
        raise ValueError(f"{path.name} is missing rotation_quat_xyzw")

    return (
        np.asarray(translation[:3], dtype=np.float64),
        R.from_quat(np.asarray(quat[:4], dtype=np.float64)).as_matrix(),
        frame,
    )


def main() -> None:
    folder = Path(__file__).resolve().parent
    handeye_path = find_latest_handeye_file(folder)
    t_cg, r_cg, frame = load_camera_in_gripper(handeye_path)

    print(f"Loaded: {handeye_path.name}")
    print(f"Frame: {frame}")
    print(f"Translation_m: {t_cg.tolist()}")
    print(f"Blue axes: flange frame (origin at flange center, z forward)")
    print(f"Red axes: camera frame (origin at camera optical center, z forward)")

    cam_origin = np.zeros(3, dtype=np.float64)
    cam_axes = create_axes(cam_origin, np.eye(3), AXIS_LEN)
    grip_axes = create_axes(t_cg, r_cg, AXIS_LEN)

    cam_sphere = create_sphere(cam_origin, [0, 0, 1])
    grip_sphere = create_sphere(t_cg, [1, 0, 0])
    connection = create_line(cam_origin, t_cg, [1, 1, 0])

    o3d.visualization.draw_geometries(
        cam_axes + grip_axes + [cam_sphere, grip_sphere, connection]
    )


if __name__ == "__main__":
    main()
