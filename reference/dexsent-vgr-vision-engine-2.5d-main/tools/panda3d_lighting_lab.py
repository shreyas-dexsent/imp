from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import panda3d.core as p3d
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_MESH_PATH = WORKSPACE_ROOT / "data/stations/station-1/assets/asset-1/objects/barel2/barel2.obj"


DEFAULT_CONFIG: dict[str, Any] = {
    "ambient": {
        "enabled": True,
        "color": [1.0, 1.0, 1.0],
        "intensity": 0.02,
    },
    "key": {
        "enabled": True,
        "light_type": "spot",
        "color": [1.65, 1.62, 1.56],
        "intensity": 1.0,
        "offset": [-1.85, -1.35, 2.35],
        "radius_scale": 4.0,
        "camera_relative": False,
        "aim_at_center": True,
        "shadow_caster": True,
        "shadow_buffer_size": 2048,
        "lens_fov": 42.0,
        "ortho_film_scale": 8.0,
        "near_scale": 0.25,
        "far_scale": 12.0,
        "exponent": 18.0,
        "max_distance": 1000.0,
        "attenuation": [1.0, 0.0, 0.35],
    },
    "fill": {
        "enabled": True,
        "light_type": "point",
        "color": [0.16, 0.17, 0.20],
        "intensity": 1.0,
        "offset": [1.10, 0.35, 0.80],
        "radius_scale": 4.0,
        "camera_relative": False,
        "aim_at_center": False,
        "shadow_caster": False,
        "shadow_buffer_size": 1024,
        "lens_fov": 50.0,
        "ortho_film_scale": 8.0,
        "near_scale": 0.25,
        "far_scale": 12.0,
        "exponent": 12.0,
        "max_distance": 1000.0,
        "attenuation": [1.0, 0.0, 0.35],
    },
    "rim": {
        "enabled": False,
        "light_type": "directional",
        "color": [0.38, 0.40, 0.45],
        "intensity": 0.22,
        "offset": [-1.40, -0.35, 1.20],
        "radius_scale": 4.0,
        "camera_relative": True,
        "aim_at_center": True,
        "shadow_caster": False,
        "shadow_buffer_size": 1024,
        "lens_fov": 50.0,
        "ortho_film_scale": 8.0,
        "near_scale": 0.25,
        "far_scale": 12.0,
        "exponent": 12.0,
        "max_distance": 1000.0,
        "attenuation": [1.0, 0.0, 0.35],
    },
    "top": {
        "enabled": False,
        "light_type": "directional",
        "color": [0.52, 0.52, 0.50],
        "intensity": 0.18,
        "offset": [0.0, 0.55, 1.80],
        "radius_scale": 4.0,
        "camera_relative": True,
        "aim_at_center": True,
        "shadow_caster": False,
        "shadow_buffer_size": 1024,
        "lens_fov": 50.0,
        "ortho_film_scale": 8.0,
        "near_scale": 0.25,
        "far_scale": 12.0,
        "exponent": 12.0,
        "max_distance": 1000.0,
        "attenuation": [1.0, 0.0, 0.35],
    },
    "bounce": {
        "enabled": False,
        "light_type": "directional",
        "color": [0.28, 0.27, 0.25],
        "intensity": 0.16,
        "offset": [0.35, 0.85, -1.10],
        "radius_scale": 4.0,
        "camera_relative": True,
        "aim_at_center": True,
        "shadow_caster": False,
        "shadow_buffer_size": 1024,
        "lens_fov": 50.0,
        "ortho_film_scale": 8.0,
        "near_scale": 0.25,
        "far_scale": 12.0,
        "exponent": 12.0,
        "max_distance": 1000.0,
        "attenuation": [1.0, 0.0, 0.35],
    },
    "back": {
        "enabled": False,
        "light_type": "point",
        "color": [0.50, 0.50, 0.52],
        "intensity": 0.22,
        "offset": [0.0, -1.0, 0.0],
        "radius_scale": 4.0,
        "camera_relative": False,
        "aim_at_center": False,
        "shadow_caster": False,
        "shadow_buffer_size": 1024,
        "lens_fov": 50.0,
        "ortho_film_scale": 8.0,
        "near_scale": 0.25,
        "far_scale": 12.0,
        "exponent": 12.0,
        "max_distance": 1000.0,
        "attenuation": [1.0, 0.0, 0.35],
    },
    "material": {
        "enabled": True,
        "base_color": [0.58, 0.58, 0.58],
        "ambient": 0.12,
        "specular": 0.24,
        "roughness": 0.62,
        "shininess": 18.0,
        "metallic": 0.0,
        "two_sided": True,
    },
    "ground": {
        "enabled": False,
        "z_offset_scale": 0.02,
        "size_scale": 8.0,
        "color": [0.16, 0.16, 0.16],
        "ambient": 0.05,
        "specular": 0.05,
        "roughness": 1.0,
        "shininess": 2.0,
    },
    "view": {
        "background": [0.13, 0.13, 0.13],
        "camera_yaw_deg": 36.0,
        "camera_pitch_deg": 18.0,
        "camera_distance_scale": 5.8,
        "camera_fov_deg": 38.0,
        "look_at_offset": [0.0, 0.0, 0.0],
        "auto_spin": False,
        "spin_speed_deg": 18.0,
        "wireframe": False,
    },
}


LIGHT_GROUPS = ("key", "fill", "rim", "top", "bounce", "back")


@dataclass(frozen=True)
class ParamSpec:
    label: str
    path: tuple[Any, ...]
    kind: str
    step: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()


def _deep_copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)


def _merge_config(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = _deep_copy_config(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _merge_config(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    if isinstance(base, list) and isinstance(override, list):
        merged = list(base)
        for index, value in enumerate(override):
            if index < len(merged):
                merged[index] = _merge_config(merged[index], value)
            else:
                merged.append(copy.deepcopy(value))
        return merged
    return copy.deepcopy(override)


def _get_path_value(config: dict[str, Any], path: Sequence[Any]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def _set_path_value(config: dict[str, Any], path: Sequence[Any], value: Any) -> None:
    target: Any = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _spec_group(spec: ParamSpec) -> str:
    return str(spec.path[0])


def _rgb_specs(group: str, prefix: str) -> list[ParamSpec]:
    return [
        ParamSpec(f"{prefix}.r", (group, "color", 0), "float", 0.02, 0.0, 4.0),
        ParamSpec(f"{prefix}.g", (group, "color", 1), "float", 0.02, 0.0, 4.0),
        ParamSpec(f"{prefix}.b", (group, "color", 2), "float", 0.02, 0.0, 4.0),
    ]


def _material_rgb_specs() -> list[ParamSpec]:
    return [
        ParamSpec("material.base.r", ("material", "base_color", 0), "float", 0.02, 0.0, 1.5),
        ParamSpec("material.base.g", ("material", "base_color", 1), "float", 0.02, 0.0, 1.5),
        ParamSpec("material.base.b", ("material", "base_color", 2), "float", 0.02, 0.0, 1.5),
    ]


def _view_rgb_specs() -> list[ParamSpec]:
    return [
        ParamSpec("view.bg.r", ("view", "background", 0), "float", 0.02, 0.0, 1.0),
        ParamSpec("view.bg.g", ("view", "background", 1), "float", 0.02, 0.0, 1.0),
        ParamSpec("view.bg.b", ("view", "background", 2), "float", 0.02, 0.0, 1.0),
    ]


def _ground_rgb_specs() -> list[ParamSpec]:
    return [
        ParamSpec("ground.color.r", ("ground", "color", 0), "float", 0.02, 0.0, 1.5),
        ParamSpec("ground.color.g", ("ground", "color", 1), "float", 0.02, 0.0, 1.5),
        ParamSpec("ground.color.b", ("ground", "color", 2), "float", 0.02, 0.0, 1.5),
    ]


def _look_at_specs() -> list[ParamSpec]:
    return [
        ParamSpec("view.look_at.x", ("view", "look_at_offset", 0), "float", 0.05, -5.0, 5.0),
        ParamSpec("view.look_at.y", ("view", "look_at_offset", 1), "float", 0.05, -5.0, 5.0),
        ParamSpec("view.look_at.z", ("view", "look_at_offset", 2), "float", 0.05, -5.0, 5.0),
    ]


def _light_specs(group: str) -> list[ParamSpec]:
    return [
        ParamSpec(f"{group}.enabled", (group, "enabled"), "bool"),
        ParamSpec(
            f"{group}.light_type",
            (group, "light_type"),
            "enum",
            choices=("spot", "point", "directional"),
        ),
        ParamSpec(f"{group}.intensity", (group, "intensity"), "float", 0.05, 0.0, 8.0),
        *_rgb_specs(group, f"{group}.color"),
        ParamSpec(f"{group}.offset.x", (group, "offset", 0), "float", 0.05, -8.0, 8.0),
        ParamSpec(f"{group}.offset.y", (group, "offset", 1), "float", 0.05, -8.0, 8.0),
        ParamSpec(f"{group}.offset.z", (group, "offset", 2), "float", 0.05, -8.0, 8.0),
        ParamSpec(f"{group}.radius_scale", (group, "radius_scale"), "float", 0.1, 0.1, 24.0),
        ParamSpec(f"{group}.camera_relative", (group, "camera_relative"), "bool"),
        ParamSpec(f"{group}.aim_at_center", (group, "aim_at_center"), "bool"),
        ParamSpec(f"{group}.shadow_caster", (group, "shadow_caster"), "bool"),
        ParamSpec(
            f"{group}.shadow_buffer_size",
            (group, "shadow_buffer_size"),
            "int",
            256.0,
            256.0,
            8192.0,
        ),
        ParamSpec(f"{group}.lens_fov", (group, "lens_fov"), "float", 1.0, 5.0, 120.0),
        ParamSpec(
            f"{group}.ortho_film_scale",
            (group, "ortho_film_scale"),
            "float",
            0.2,
            0.5,
            40.0,
        ),
        ParamSpec(f"{group}.near_scale", (group, "near_scale"), "float", 0.05, 0.01, 8.0),
        ParamSpec(f"{group}.far_scale", (group, "far_scale"), "float", 0.25, 0.5, 80.0),
        ParamSpec(f"{group}.exponent", (group, "exponent"), "float", 1.0, 0.0, 128.0),
        ParamSpec(
            f"{group}.max_distance",
            (group, "max_distance"),
            "float",
            25.0,
            1.0,
            10000.0,
        ),
        ParamSpec(f"{group}.attenuation.c", (group, "attenuation", 0), "float", 0.05, 0.0, 8.0),
        ParamSpec(f"{group}.attenuation.l", (group, "attenuation", 1), "float", 0.05, 0.0, 8.0),
        ParamSpec(f"{group}.attenuation.q", (group, "attenuation", 2), "float", 0.05, 0.0, 8.0),
    ]


PARAM_SPECS: list[ParamSpec] = [
    ParamSpec("ambient.enabled", ("ambient", "enabled"), "bool"),
    ParamSpec("ambient.intensity", ("ambient", "intensity"), "float", 0.01, 0.0, 2.0),
    *_rgb_specs("ambient", "ambient.color"),
    *_light_specs("key"),
    *_light_specs("fill"),
    *_light_specs("rim"),
    *_light_specs("top"),
    *_light_specs("bounce"),
    *_light_specs("back"),
    ParamSpec("material.enabled", ("material", "enabled"), "bool"),
    *_material_rgb_specs(),
    ParamSpec("material.ambient", ("material", "ambient"), "float", 0.02, 0.0, 1.0),
    ParamSpec("material.specular", ("material", "specular"), "float", 0.02, 0.0, 3.0),
    ParamSpec("material.roughness", ("material", "roughness"), "float", 0.02, 0.0, 1.0),
    ParamSpec("material.shininess", ("material", "shininess"), "float", 1.0, 0.0, 256.0),
    ParamSpec("material.metallic", ("material", "metallic"), "float", 0.02, 0.0, 1.0),
    ParamSpec("material.two_sided", ("material", "two_sided"), "bool"),
    ParamSpec("ground.enabled", ("ground", "enabled"), "bool"),
    ParamSpec("ground.z_offset_scale", ("ground", "z_offset_scale"), "float", 0.01, -4.0, 4.0),
    ParamSpec("ground.size_scale", ("ground", "size_scale"), "float", 0.1, 0.5, 40.0),
    *_ground_rgb_specs(),
    ParamSpec("ground.ambient", ("ground", "ambient"), "float", 0.02, 0.0, 1.0),
    ParamSpec("ground.specular", ("ground", "specular"), "float", 0.02, 0.0, 2.0),
    ParamSpec("ground.roughness", ("ground", "roughness"), "float", 0.02, 0.0, 1.0),
    ParamSpec("ground.shininess", ("ground", "shininess"), "float", 1.0, 0.0, 128.0),
    *_view_rgb_specs(),
    ParamSpec("view.camera_yaw_deg", ("view", "camera_yaw_deg"), "float", 2.0, -180.0, 180.0),
    ParamSpec("view.camera_pitch_deg", ("view", "camera_pitch_deg"), "float", 2.0, -89.0, 89.0),
    ParamSpec(
        "view.camera_distance_scale",
        ("view", "camera_distance_scale"),
        "float",
        0.1,
        0.5,
        40.0,
    ),
    ParamSpec("view.camera_fov_deg", ("view", "camera_fov_deg"), "float", 1.0, 5.0, 120.0),
    *_look_at_specs(),
    ParamSpec("view.auto_spin", ("view", "auto_spin"), "bool"),
    ParamSpec("view.spin_speed_deg", ("view", "spin_speed_deg"), "float", 1.0, -180.0, 180.0),
    ParamSpec("view.wireframe", ("view", "wireframe"), "bool"),
]


GROUP_ORDER = (
    "ambient",
    "key",
    "fill",
    "rim",
    "top",
    "bounce",
    "back",
    "material",
    "ground",
    "view",
)


def _default_preset_path(mesh_path: Path) -> Path:
    return mesh_path.parent / f"{mesh_path.stem}.lighting_lab.json"


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (WORKSPACE_ROOT / path).resolve()


def _detect_candidate_x_displays() -> list[str]:
    display_paths = sorted(Path("/tmp/.X11-unix").glob("X*"))
    candidates: list[str] = []
    for path in display_paths:
        match = re.fullmatch(r"X(\d+)", path.name)
        if match is not None:
            candidates.append(f":{match.group(1)}")
    return candidates


def _current_user() -> str:
    return str(os.environ.get("USER") or os.environ.get("LOGNAME") or "current-user")


def _display_owner_lines() -> list[str]:
    try:
        output = subprocess.check_output(["who"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    lines: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        lines.append(line)
    return lines


def _display_help_message() -> str:
    display = str(os.environ.get("DISPLAY") or "").strip()
    wayland = str(os.environ.get("WAYLAND_DISPLAY") or "").strip()
    session_type = str(os.environ.get("XDG_SESSION_TYPE") or "").strip() or "unknown"
    current_user = _current_user()
    candidates = _detect_candidate_x_displays()
    owners = _display_owner_lines()
    lines = [
        "No GUI display is available for Panda3D onscreen rendering.",
        "",
        f"Current shell: DISPLAY={display or '<empty>'}, WAYLAND_DISPLAY={wayland or '<empty>'}, XDG_SESSION_TYPE={session_type}.",
    ]
    if candidates:
        lines.append(f"Detected X displays on this machine: {', '.join(candidates)}.")
    if owners:
        lines.append("Logged-in display sessions:")
        for line in owners:
            lines.append(f"  {line}")
    lines.extend(
        [
            "",
            "How to run the viewer:",
            "1. Best option: open a terminal inside the desktop session itself and run the tool there.",
        ]
    )
    if candidates:
        example_display = candidates[0]
        lines.extend(
            [
                "2. Or authorize this user from the desktop session, then reuse that display from this shell:",
                f"   In the desktop terminal: xhost +SI:localuser:{current_user}",
                f"   In this shell: export DISPLAY={example_display}",
                "   Then run: python tools/panda3d_lighting_lab.py",
            ]
        )
    lines.extend(
        [
            "",
            "If you still get a cookie/auth error after setting DISPLAY, the desktop session has not granted access yet.",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Panda3D lighting/material lab for CAD meshes."
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=DEFAULT_MESH_PATH,
        help="Mesh to inspect. Defaults to barel2.obj in the shared object library.",
    )
    parser.add_argument(
        "--preset",
        type=Path,
        default=None,
        help="Optional JSON preset file to load.",
    )
    parser.add_argument(
        "--save-preset",
        type=Path,
        default=None,
        help="Where the O hotkey saves the current config.",
    )
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        default=(1440, 900),
        metavar=("WIDTH", "HEIGHT"),
        help="Window size in pixels.",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="Print every editable parameter and exit.",
    )
    parser.add_argument(
        "--dump-default-config",
        action="store_true",
        help="Print the default config JSON and exit.",
    )
    return parser.parse_args()


def _color_with_intensity(rgb: Sequence[float], intensity: float) -> tuple[float, float, float, float]:
    return (
        float(rgb[0]) * float(intensity),
        float(rgb[1]) * float(intensity),
        float(rgb[2]) * float(intensity),
        1.0,
    )


def _vec3_to_np(value: p3d.LPoint3f | p3d.LVector3f) -> np.ndarray:
    return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-9:
        return fallback
    return vec / norm


class LightingLab(ShowBase):
    def __init__(
        self,
        mesh_path: Path,
        config: dict[str, Any],
        preset_path: Path,
        window_size: tuple[int, int],
    ) -> None:
        self._load_prc(window_size)
        try:
            super().__init__(windowType="onscreen")
        except Exception as exc:
            raise RuntimeError(_display_help_message()) from exc

        self.disableMouse()
        self.mesh_path = mesh_path
        self.lab_config = config
        self.preset_path = preset_path
        self.selected_index = 0
        self.hud_visible = True
        self.status_message = ""
        self._status_expire_time = 0.0
        self._light_nodes: list[p3d.NodePath] = []
        self._ground_node: p3d.NodePath | None = None
        self._mesh_radius = 0.05
        self._mesh_center = np.zeros(3, dtype=np.float64)
        self._mesh_min_z = -0.05

        self.render.set_shader_auto()
        self.render.set_antialias(p3d.AntialiasAttrib.MAuto)
        self.render.set_two_sided(True)
        self.set_background_color(*self.lab_config["view"]["background"])

        props = p3d.WindowProperties()
        props.setTitle("Panda3D Lighting Lab")
        self.win.requestProperties(props)

        self.model = self._load_model(mesh_path)
        self.model.reparentTo(self.render)
        self._center_model()
        self._create_hud()
        self._bind_keys()

        self._apply_model_modes()
        self._apply_material()
        self._rebuild_ground()
        self._rebuild_lights()
        self._apply_camera()
        self._refresh_hud()
        self.taskMgr.add(self._update_task, "lighting-lab-update")

    @staticmethod
    def _load_prc(window_size: tuple[int, int]) -> None:
        if platform.system().lower() == "linux":
            default_display = "pandagl"
        else:
            default_display = "pandagl"
        load_display = str(
            p3d.ExecutionEnvironment.getEnvironmentVariable("MEGAPOSE_PANDA3D_DISPLAY") or ""
        ).strip() or default_display
        p3d.load_prc_file_data(
            __file__,
            "\n".join(
                [
                    f"load-display {load_display}",
                    f"win-size {int(window_size[0])} {int(window_size[1])}",
                    "window-type onscreen",
                    "background-color 0.0 0.0 0.0 1.0",
                    "notify-level-display fatal",
                    "notify-level-assimp fatal",
                    "notify-level-device fatal",
                    "texture-minfilter mipmap",
                    "texture-anisotropic-degree 16",
                    "framebuffer-multisample 0",
                    "multisamples 0",
                    "audio-library-name null",
                    "load-file-type p3assimp",
                    "show-frame-rate-meter 0",
                ]
            ),
        )

    def _load_model(self, mesh_path: Path) -> p3d.NodePath:
        filename = p3d.Filename.fromOsSpecific(str(mesh_path))
        model = self.loader.loadModel(filename, noCache=True)
        if model.isEmpty():
            raise FileNotFoundError(f"Failed to load mesh: {mesh_path}")
        model.clearModelNodes()
        return model

    def _center_model(self) -> None:
        mins, maxs = self.model.getTightBounds()
        if mins is None or maxs is None:
            raise ValueError(f"Could not compute bounds for {self.mesh_path}")
        mins_np = _vec3_to_np(mins)
        maxs_np = _vec3_to_np(maxs)
        center = 0.5 * (mins_np + maxs_np)
        extents = maxs_np - mins_np
        radius = max(float(np.linalg.norm(extents)) * 0.5, 1e-3)
        self.model.setPos(
            float(-center[0]),
            float(-center[1]),
            float(-center[2]),
        )
        mins2, maxs2 = self.model.getTightBounds()
        if mins2 is None or maxs2 is None:
            raise ValueError(f"Could not recompute centered bounds for {self.mesh_path}")
        mins2_np = _vec3_to_np(mins2)
        maxs2_np = _vec3_to_np(maxs2)
        self._mesh_center = 0.5 * (mins2_np + maxs2_np)
        self._mesh_radius = radius
        self._mesh_min_z = float(mins2_np[2])

    def _bind_keys(self) -> None:
        self.accept("escape", self.userExit)
        self.accept("[", lambda: self._cycle_param(-1))
        self.accept("]", lambda: self._cycle_param(1))
        self.accept("j", lambda: self._adjust_selected(-1.0))
        self.accept("k", lambda: self._adjust_selected(1.0))
        self.accept("u", lambda: self._adjust_selected(-5.0))
        self.accept("i", lambda: self._adjust_selected(5.0))
        self.accept("space", self._toggle_selected)
        self.accept("t", lambda: self._cycle_selected_enum(1))

        self.accept("a", lambda: self._adjust_camera(yaw_delta=-4.0))
        self.accept("d", lambda: self._adjust_camera(yaw_delta=4.0))
        self.accept("w", lambda: self._adjust_camera(pitch_delta=4.0))
        self.accept("s", lambda: self._adjust_camera(pitch_delta=-4.0))
        self.accept("q", lambda: self._adjust_camera(distance_delta=-0.25))
        self.accept("e", lambda: self._adjust_camera(distance_delta=0.25))
        self.accept("wheel_up", lambda: self._adjust_camera(distance_delta=-0.15))
        self.accept("wheel_down", lambda: self._adjust_camera(distance_delta=0.15))
        self.accept("c", self._reset_view)
        self.accept("v", self._toggle_auto_spin)

        self.accept("m", self._toggle_material_enabled)
        self.accept("g", self._toggle_ground_enabled)
        self.accept("y", self._toggle_key_shadows)
        self.accept("h", self._toggle_hud)
        self.accept("p", self._print_config)
        self.accept("o", self._save_preset)
        self.accept("l", self._load_preset)
        self.accept("x", self._save_screenshot)
        self.accept("r", self._reset_config)

    def _create_hud(self) -> None:
        self.hud_text = OnscreenText(
            text="",
            parent=self.aspect2d,
            align=p3d.TextNode.ALeft,
            fg=(0.95, 0.95, 0.95, 1.0),
            pos=(-1.31, 0.95),
            scale=0.040,
            mayChange=True,
        )
        self.status_text = OnscreenText(
            text="",
            parent=self.aspect2d,
            align=p3d.TextNode.ALeft,
            fg=(0.98, 0.85, 0.40, 1.0),
            pos=(-1.31, -0.92),
            scale=0.046,
            mayChange=True,
        )

    def _target_center(self) -> np.ndarray:
        return self._mesh_center + np.array(self.lab_config["view"]["look_at_offset"], dtype=np.float64)

    def _apply_camera(self) -> None:
        view_cfg = self.lab_config["view"]
        target = self._target_center()
        yaw = math.radians(float(view_cfg["camera_yaw_deg"]))
        pitch = math.radians(float(view_cfg["camera_pitch_deg"]))
        distance = max(0.05, self._mesh_radius * float(view_cfg["camera_distance_scale"]))
        direction = np.array(
            [
                math.cos(pitch) * math.cos(yaw),
                -math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
            ],
            dtype=np.float64,
        )
        camera_pos = target + (direction * distance)
        self.camera.setPos(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2]))
        self.camera.lookAt(float(target[0]), float(target[1]), float(target[2]))
        self.camLens.setFov(float(view_cfg["camera_fov_deg"]))
        self.camLens.setNearFar(max(0.001, self._mesh_radius * 0.02), max(10.0, self._mesh_radius * 80.0))
        self._update_light_poses()

    def _apply_model_modes(self) -> None:
        if bool(self.lab_config["view"]["wireframe"]):
            self.model.setRenderModeWireframe()
        else:
            self.model.clearRenderMode()
        self.model.set_two_sided(bool(self.lab_config["material"]["two_sided"]))

    def _apply_material(self) -> None:
        material_cfg = self.lab_config["material"]
        if not bool(material_cfg["enabled"]):
            self.model.clearMaterial()
            self.model.clearColor()
            return
        base = material_cfg["base_color"]
        material = p3d.Material()
        base_color = p3d.Vec4(float(base[0]), float(base[1]), float(base[2]), 1.0)
        ambient_scale = float(material_cfg["ambient"])
        specular_scale = float(material_cfg["specular"])
        material.set_base_color(base_color)
        material.set_ambient(
            p3d.Vec4(
                float(base[0]) * ambient_scale,
                float(base[1]) * ambient_scale,
                float(base[2]) * ambient_scale,
                1.0,
            )
        )
        material.set_diffuse(base_color)
        material.set_specular(p3d.Vec3(specular_scale, specular_scale, specular_scale))
        material.set_roughness(float(material_cfg["roughness"]))
        material.set_shininess(float(material_cfg["shininess"]))
        material.set_metallic(float(material_cfg["metallic"]))
        material.set_twoside(bool(material_cfg["two_sided"]))
        self.model.setColor(base_color)
        self.model.setMaterial(material, 1)
        self.model.set_two_sided(bool(material_cfg["two_sided"]))

    def _make_ground_material(self) -> p3d.Material:
        cfg = self.lab_config["ground"]
        color = cfg["color"]
        material = p3d.Material()
        base_color = p3d.Vec4(float(color[0]), float(color[1]), float(color[2]), 1.0)
        material.set_base_color(base_color)
        material.set_ambient(
            p3d.Vec4(
                float(color[0]) * float(cfg["ambient"]),
                float(color[1]) * float(cfg["ambient"]),
                float(color[2]) * float(cfg["ambient"]),
                1.0,
            )
        )
        material.set_diffuse(base_color)
        material.set_specular(
            p3d.Vec3(float(cfg["specular"]), float(cfg["specular"]), float(cfg["specular"]))
        )
        material.set_roughness(float(cfg["roughness"]))
        material.set_shininess(float(cfg["shininess"]))
        material.set_metallic(0.0)
        material.set_twoside(True)
        return material

    def _rebuild_ground(self) -> None:
        if self._ground_node is not None:
            self._ground_node.removeNode()
            self._ground_node = None
        if not bool(self.lab_config["ground"]["enabled"]):
            return
        card = p3d.CardMaker("ground")
        card.setFrame(-1.0, 1.0, -1.0, 1.0)
        ground = self.render.attachNewNode(card.generate())
        ground.setP(-90.0)
        ground.setScale(max(0.05, self._mesh_radius * float(self.lab_config["ground"]["size_scale"])))
        ground_z = self._mesh_min_z - (float(self.lab_config["ground"]["z_offset_scale"]) * self._mesh_radius)
        ground.setPos(float(self._mesh_center[0]), float(self._mesh_center[1]), float(ground_z))
        ground.setMaterial(self._make_ground_material(), 1)
        ground.setColor(
            float(self.lab_config["ground"]["color"][0]),
            float(self.lab_config["ground"]["color"][1]),
            float(self.lab_config["ground"]["color"][2]),
            1.0,
        )
        ground.set_two_sided(True)
        self._ground_node = ground

    def _build_ambient_light(self) -> p3d.NodePath | None:
        cfg = self.lab_config["ambient"]
        if not bool(cfg["enabled"]):
            return None
        light = p3d.AmbientLight("ambient")
        light.set_color(_color_with_intensity(cfg["color"], float(cfg["intensity"])))
        node = self.render.attachNewNode(light)
        self.render.setLight(node)
        return node

    def _build_shaped_light(self, name: str, cfg: dict[str, Any]) -> p3d.NodePath | None:
        if not bool(cfg["enabled"]):
            return None
        light_type = str(cfg["light_type"]).strip().lower()
        if light_type == "point":
            light: Any = p3d.PointLight(name)
        elif light_type == "spot":
            light = p3d.Spotlight(name)
            lens = p3d.PerspectiveLens()
            lens.setFov(float(cfg["lens_fov"]))
            light.setLens(lens)
            light.setExponent(float(cfg["exponent"]))
            light.setMaxDistance(float(cfg["max_distance"]))
        elif light_type == "directional":
            light = p3d.DirectionalLight(name)
            lens = p3d.OrthographicLens()
            light.setLens(lens)
        else:
            raise ValueError(f"Unsupported light type: {light_type}")

        rgba = _color_with_intensity(cfg["color"], float(cfg["intensity"]))
        light.set_color(rgba)
        if hasattr(light, "set_specular_color"):
            light.set_specular_color(rgba)
        if hasattr(light, "set_attenuation"):
            light.set_attenuation(tuple(float(x) for x in cfg["attenuation"]))
        if bool(cfg["shadow_caster"]) and light_type in {"spot", "directional"}:
            size = int(cfg["shadow_buffer_size"])
            light.set_shadow_caster(True, size, size)
        node = self.render.attachNewNode(light)
        node.setTag("lighting_lab_name", name)
        node.setTag("lighting_lab_type", light_type)
        self.render.setLight(node)
        return node

    def _rebuild_lights(self) -> None:
        self.render.clearLight()
        for node in self._light_nodes:
            node.removeNode()
        self._light_nodes = []
        ambient_node = self._build_ambient_light()
        if ambient_node is not None:
            self._light_nodes.append(ambient_node)
        for name in LIGHT_GROUPS:
            node = self._build_shaped_light(name, self.lab_config[name])
            if node is not None:
                self._light_nodes.append(node)
        self._update_light_poses()

    def _light_world_position(self, cfg: dict[str, Any]) -> np.ndarray:
        target = self._target_center()
        offset = np.array(cfg["offset"], dtype=np.float64)
        distance_scale = self._mesh_radius * float(cfg["radius_scale"])
        if bool(cfg["camera_relative"]):
            camera_pos = _vec3_to_np(self.camera.getPos(self.render))
            toward_camera = _normalize(camera_pos - target, np.array([0.0, -1.0, 0.0], dtype=np.float64))
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            right = _normalize(
                np.cross(toward_camera, world_up),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
            up = _normalize(np.cross(right, toward_camera), world_up)
            return target + (
                distance_scale
                * (
                    (offset[0] * right)
                    + (offset[1] * toward_camera)
                    + (offset[2] * up)
                )
            )
        return target + (distance_scale * offset)

    def _update_light_pose(self, node: p3d.NodePath, cfg: dict[str, Any]) -> None:
        name = node.getTag("lighting_lab_name")
        if not name:
            return
        light_type = node.getTag("lighting_lab_type")
        world_pos = self._light_world_position(cfg)
        target = self._target_center()
        node.setPos(float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))
        if bool(cfg["aim_at_center"]) or light_type in {"spot", "directional"}:
            node.lookAt(float(target[0]), float(target[1]), float(target[2]))
        light = node.node()
        lens = None
        if hasattr(light, "getLens"):
            lens = light.getLens()
        elif hasattr(light, "get_lens"):
            lens = light.get_lens()
        if lens is None:
            return
        distance = float(np.linalg.norm(world_pos - target))
        near_far = (
            max(0.01, self._mesh_radius * float(cfg["near_scale"])),
            max(
                self._mesh_radius * float(cfg["far_scale"]),
                distance + (self._mesh_radius * 3.0),
            ),
        )
        if light_type == "spot":
            lens.setNearFar(float(near_far[0]), float(near_far[1]))
            if hasattr(lens, "setFov"):
                lens.setFov(float(cfg["lens_fov"]))
        elif light_type == "directional":
            film_size = max(0.05, self._mesh_radius * float(cfg["ortho_film_scale"]))
            if hasattr(lens, "setFilmSize"):
                lens.setFilmSize(float(film_size), float(film_size))
            lens.setNearFar(float(near_far[0]), float(near_far[1]))

    def _update_light_poses(self) -> None:
        for node in self._light_nodes:
            name = node.getTag("lighting_lab_name")
            if name in LIGHT_GROUPS:
                self._update_light_pose(node, self.lab_config[name])

    def _adjust_camera(
        self,
        yaw_delta: float = 0.0,
        pitch_delta: float = 0.0,
        distance_delta: float = 0.0,
    ) -> None:
        view_cfg = self.lab_config["view"]
        view_cfg["camera_yaw_deg"] = float(view_cfg["camera_yaw_deg"]) + float(yaw_delta)
        view_cfg["camera_pitch_deg"] = max(
            -89.0,
            min(89.0, float(view_cfg["camera_pitch_deg"]) + float(pitch_delta)),
        )
        view_cfg["camera_distance_scale"] = max(
            0.5,
            float(view_cfg["camera_distance_scale"]) + float(distance_delta),
        )
        self._apply_camera()
        self._refresh_hud()

    def _cycle_param(self, delta: int) -> None:
        self.selected_index = (self.selected_index + int(delta)) % len(PARAM_SPECS)
        self._refresh_hud()

    def _adjust_selected(self, scale: float) -> None:
        spec = PARAM_SPECS[self.selected_index]
        current = _get_path_value(self.lab_config, spec.path)
        if spec.kind == "bool":
            self._toggle_selected()
            return
        if spec.kind == "enum":
            self._cycle_selected_enum(1 if scale >= 0 else -1)
            return
        delta = float(spec.step) * float(scale)
        updated = float(current) + delta
        if spec.minimum is not None:
            updated = max(float(spec.minimum), updated)
        if spec.maximum is not None:
            updated = min(float(spec.maximum), updated)
        if spec.kind == "int":
            updated_value: Any = int(round(updated))
        else:
            updated_value = updated
        _set_path_value(self.lab_config, spec.path, updated_value)
        self._apply_full_config()

    def _toggle_selected(self) -> None:
        spec = PARAM_SPECS[self.selected_index]
        if spec.kind != "bool":
            return
        current = bool(_get_path_value(self.lab_config, spec.path))
        _set_path_value(self.lab_config, spec.path, not current)
        self._apply_full_config()

    def _cycle_selected_enum(self, delta: int) -> None:
        spec = PARAM_SPECS[self.selected_index]
        if spec.kind != "enum" or not spec.choices:
            return
        current = str(_get_path_value(self.lab_config, spec.path))
        try:
            idx = spec.choices.index(current)
        except ValueError:
            idx = 0
        idx = (idx + int(delta)) % len(spec.choices)
        _set_path_value(self.lab_config, spec.path, spec.choices[idx])
        self._apply_full_config()

    def _toggle_material_enabled(self) -> None:
        self.lab_config["material"]["enabled"] = not bool(self.lab_config["material"]["enabled"])
        self._apply_full_config()
        self._set_status(f"material enabled: {self.lab_config['material']['enabled']}")

    def _toggle_ground_enabled(self) -> None:
        self.lab_config["ground"]["enabled"] = not bool(self.lab_config["ground"]["enabled"])
        self._apply_full_config()
        self._set_status(f"ground enabled: {self.lab_config['ground']['enabled']}")

    def _toggle_key_shadows(self) -> None:
        self.lab_config["key"]["shadow_caster"] = not bool(self.lab_config["key"]["shadow_caster"])
        self._apply_full_config()
        self._set_status(f"key.shadow_caster: {self.lab_config['key']['shadow_caster']}")

    def _toggle_auto_spin(self) -> None:
        self.lab_config["view"]["auto_spin"] = not bool(self.lab_config["view"]["auto_spin"])
        self._refresh_hud()
        self._set_status(f"auto spin: {self.lab_config['view']['auto_spin']}")

    def _toggle_hud(self) -> None:
        self.hud_visible = not self.hud_visible
        self.hud_text.setText("" if not self.hud_visible else self.hud_text.getText())
        self._refresh_hud()

    def _reset_view(self) -> None:
        for key in ("camera_yaw_deg", "camera_pitch_deg", "camera_distance_scale", "camera_fov_deg"):
            self.lab_config["view"][key] = _deep_copy_config(DEFAULT_CONFIG)["view"][key]
        self.lab_config["view"]["look_at_offset"] = copy.deepcopy(DEFAULT_CONFIG["view"]["look_at_offset"])
        self._apply_camera()
        self._refresh_hud()
        self._set_status("view reset")

    def _reset_config(self) -> None:
        self.lab_config = _deep_copy_config(DEFAULT_CONFIG)
        self._apply_full_config()
        self._set_status("config reset to defaults")

    def _apply_full_config(self) -> None:
        self.set_background_color(*self.lab_config["view"]["background"])
        self._apply_model_modes()
        self._apply_material()
        self._rebuild_ground()
        self._rebuild_lights()
        self._apply_camera()
        self._refresh_hud()

    def _save_preset(self) -> None:
        self.preset_path.parent.mkdir(parents=True, exist_ok=True)
        self.preset_path.write_text(json.dumps(self.lab_config, indent=2) + "\n", encoding="utf-8")
        self._set_status(f"saved preset: {self.preset_path}")

    def _load_preset(self) -> None:
        if not self.preset_path.exists():
            self._set_status(f"preset not found: {self.preset_path}")
            return
        payload = json.loads(self.preset_path.read_text(encoding="utf-8"))
        self.lab_config = _merge_config(DEFAULT_CONFIG, payload)
        self._apply_full_config()
        self._set_status(f"loaded preset: {self.preset_path}")

    def _print_config(self) -> None:
        print(json.dumps(self.lab_config, indent=2))
        self._set_status("printed config to terminal")

    def _save_screenshot(self) -> None:
        screenshot_path = self.preset_path.with_name(
            f"{self.mesh_path.stem}_lighting_lab_{time.strftime('%Y%m%d_%H%M%S')}.png"
        )
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.win.saveScreenshot(p3d.Filename.fromOsSpecific(str(screenshot_path)))
        self._set_status(f"saved screenshot: {screenshot_path}")

    def _set_status(self, message: str) -> None:
        self.status_message = message
        self._status_expire_time = time.time() + 6.0
        self.status_text.setText(message)
        print(message)

    def _group_lines(self, group: str) -> list[str]:
        lines: list[str] = []
        for index, spec in enumerate(PARAM_SPECS):
            if _spec_group(spec) != group:
                continue
            marker = ">" if index == self.selected_index else " "
            value = _format_scalar(_get_path_value(self.lab_config, spec.path))
            lines.append(f"{marker} {spec.label:<28} {value}")
        return lines

    def _refresh_hud(self) -> None:
        if not self.hud_visible:
            self.hud_text.setText("")
            return
        spec = PARAM_SPECS[self.selected_index]
        group = _spec_group(spec)
        view_cfg = self.lab_config["view"]
        lines = [
            "Panda3D Lighting Lab",
            f"mesh: {self.mesh_path}",
            f"preset: {self.preset_path}",
            "",
            "select: [ ]   tweak: j/k small  u/i large",
            "types: space toggle   t cycle enum",
            "camera: a/d yaw  w/s pitch  q/e zoom  v auto-spin  c reset",
            "quick: m material  g ground  y key-shadows",
            "io: o save preset  l load preset  p print config  x screenshot",
            "other: h hide hud  r reset all  esc quit",
            "",
            f"selected: {spec.label} = {_format_scalar(_get_path_value(self.lab_config, spec.path))}",
            f"group: {group}",
            "",
            *self._group_lines(group),
            "",
            (
                "camera "
                f"yaw={float(view_cfg['camera_yaw_deg']):.1f} "
                f"pitch={float(view_cfg['camera_pitch_deg']):.1f} "
                f"dist={float(view_cfg['camera_distance_scale']):.2f} "
                f"fov={float(view_cfg['camera_fov_deg']):.1f}"
            ),
            f"radius: {self._mesh_radius:.5f}",
        ]
        self.hud_text.setText("\n".join(lines))

    def _update_task(self, task: Any) -> int:
        if bool(self.lab_config["view"]["auto_spin"]):
            dt = globalClock.getDt()
            self.lab_config["view"]["camera_yaw_deg"] = float(self.lab_config["view"]["camera_yaw_deg"]) + (
                float(self.lab_config["view"]["spin_speed_deg"]) * float(dt)
            )
            self._apply_camera()
            self._refresh_hud()
        elif any(
            bool(self.lab_config[name]["camera_relative"]) and bool(self.lab_config[name]["enabled"])
            for name in LIGHT_GROUPS
        ):
            self._update_light_poses()
        if self.status_message and time.time() > self._status_expire_time:
            self.status_message = ""
            self.status_text.setText("")
        return task.cont


def _list_params() -> None:
    for spec in PARAM_SPECS:
        if spec.kind == "enum":
            extra = f" choices={','.join(spec.choices)}"
        elif spec.kind in {"float", "int"}:
            extra = f" step={spec.step} min={spec.minimum} max={spec.maximum}"
        else:
            extra = ""
        print(f"{spec.label:<32} kind={spec.kind}{extra}")


def _load_initial_config(preset_path: Path | None) -> tuple[dict[str, Any], Path]:
    if preset_path is None:
        raise ValueError("preset_path must be provided")
    if preset_path.exists():
        payload = json.loads(preset_path.read_text(encoding="utf-8"))
        return _merge_config(DEFAULT_CONFIG, payload), preset_path
    return _deep_copy_config(DEFAULT_CONFIG), preset_path


def main() -> None:
    args = _parse_args()
    if args.list_params:
        _list_params()
        return
    if args.dump_default_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2))
        return

    mesh_path = _resolve_path(args.mesh)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    preset_arg = _resolve_path(args.preset) if args.preset is not None else None
    if preset_arg is None:
        preset_arg = _default_preset_path(mesh_path)
    config, derived_preset = _load_initial_config(preset_arg)
    save_preset_path = _resolve_path(args.save_preset) if args.save_preset is not None else derived_preset

    lab = LightingLab(
        mesh_path=mesh_path,
        config=config,
        preset_path=save_preset_path,
        window_size=(int(args.window_size[0]), int(args.window_size[1])),
    )
    lab.run()


if __name__ == "__main__":
    main()
