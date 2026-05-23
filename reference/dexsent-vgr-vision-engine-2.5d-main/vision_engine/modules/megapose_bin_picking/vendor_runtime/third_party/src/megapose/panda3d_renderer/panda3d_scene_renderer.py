"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



# Standard Library
import builtins
import os
import platform
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Set
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Set

# Third Party
import numpy as np
import panda3d as p3d
from direct.showbase.ShowBase import ShowBase
from tqdm import tqdm

# MegaPose
from megapose.datasets.object_dataset import RigidObjectDataset

# Local Folder
from .types import (
    CameraRenderingData,
    Panda3dCamera,
    Panda3dCameraData,
    Panda3dLightData,
    Panda3dObjectData,
    Resolution,
    RgbaColor,
)
from .utils import make_rgb_texture_normal_map, np_to_lmatrix4


@dataclass
class Panda3dDebugData:
    timings: Dict[str, float]


_TWO_SIDED_CAD_SHADER = None


def _get_two_sided_cad_shader() -> p3d.core.Shader:
    global _TWO_SIDED_CAD_SHADER
    if _TWO_SIDED_CAD_SHADER is not None:
        return _TWO_SIDED_CAD_SHADER

    _TWO_SIDED_CAD_SHADER = p3d.core.Shader.make(
        p3d.core.Shader.SL_GLSL,
        vertex="""
#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;

out vec3 v_normal_eye;
out vec3 v_pos_eye;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_normal_eye = normalize(p3d_NormalMatrix * p3d_Normal);
    v_pos_eye = (p3d_ModelViewMatrix * p3d_Vertex).xyz;
}
""",
        fragment="""
#version 150

uniform vec4 u_base_color;

in vec3 v_normal_eye;
in vec3 v_pos_eye;

out vec4 p3d_FragColor;

// Matte dark-rubber look with a shadowed cavity.
//
// Tuned to match a real dark-grey rubber part photographed under diffuse
// workshop lighting: medium-dark mid-tones, visible surface detail, no
// plasticky highlights. The cavity stays darker than the outer silhouette
// so the coarse network can discriminate front/back hypotheses.
void main() {
    vec3 n_raw = normalize(v_normal_eye);
    // Camera sits at origin in eye space, looking down -Z.
    vec3 view_dir = normalize(-v_pos_eye);

    // Flip the sign of the normal only when it points away from the
    // camera (inner-cavity walls, two-sided rendering) so the cavity-
    // darkening math below sees a consistent view-facing normal.
    float view_sign = sign(dot(n_raw, view_dir));
    if (view_sign == 0.0) view_sign = 1.0;
    vec3 n = n_raw * view_sign;

    // Soft three-point rig in eye space (x right, y up, z toward camera).
    // Brighter fill/rim so the object reads as mid-tone rather than black.
    vec3 key_dir  = normalize(vec3(-0.55,  0.75,  0.55));
    vec3 fill_dir = normalize(vec3( 0.80,  0.25,  0.45));
    vec3 rim_dir  = normalize(vec3( 0.10,  0.85, -0.55));

    vec3 key_col  = vec3(1.00, 0.98, 0.94);
    vec3 fill_col = vec3(0.55, 0.57, 0.62);
    vec3 rim_col  = vec3(0.45, 0.47, 0.52);

    float key_n  = max(dot(n, key_dir),  0.0);
    float fill_n = max(dot(n, fill_dir), 0.0);
    float rim_n  = max(dot(n, rim_dir),  0.0);

    vec3 diffuse = (key_col * key_n) + (fill_col * fill_n) + (rim_col * rim_n);

    // Low, broad specular — rubber has a weak soft highlight, not a
    // plastic point-highlight. Keeps the surface reading as matte.
    vec3 h = normalize(key_dir + view_dir);
    float spec_ndoth = max(dot(n, h), 0.0);
    float spec_exp = 12.0;
    float spec_amt = pow(spec_ndoth, spec_exp) * 0.12;
    vec3 specular = key_col * spec_amt;

    // Cavity darkening: front-facing pixels stay full brightness, pixels
    // where the normal grazes the view (inner walls) get attenuated.
    float view_facing = max(dot(n_raw * view_sign, view_dir), 0.0);
    float cavity_term = 0.25 + (0.75 * view_facing);

    float ambient = 0.22;
    vec3 shaded = (u_base_color.rgb * (vec3(ambient) + diffuse) + specular) * cavity_term;

    // Soft tonemap + mild gamma so mid-tones sit where a real photo lands.
    shaded = shaded / (shaded + vec3(1.0));
    shaded = pow(shaded, vec3(1.0 / 1.8));

    p3d_FragColor = vec4(shaded, u_base_color.a);
}
""",
    )
    return _TWO_SIDED_CAD_SHADER


class App(ShowBase):
    """Panda3d App."""

    def __init__(self) -> None:
        display_backend = str(os.environ.get("MEGAPOSE_PANDA3D_DISPLAY", "")).strip()
        if not display_backend:
            if platform.system().lower() == "linux":
                display_backend = "p3headlessgl"
            else:
                display_backend = "pandagl"
        p3d.core.load_prc_file_data(
            __file__,
            f"load-display {display_backend}\n"
            "notify-level-display fatal\n"
            "notify-level-assimp fatal\n"
            "notify-level-egldisplay fatal\n"
            "notify-level-glgsg fatal\n"
            "notify-level-glxdisplay fatal\n"
            "notify-level-x11display fatal\n"
            "notify-level-device fatal\n"
            "texture-minfilter mipmap\n"
            "texture-anisotropic-degree 16\n"
            "framebuffer-multisample 0\n"
            "multisamples 0\n"
            "background-color 1.0 1.0 1.0 1.0\n"
            "load-file-type p3assimp\n"
            "track-memory-usage 1\n"
            "transform-cache 0\n"
            "state-cache 0\n"
            "audio-library-name null\n"
            "model-cache-dir\n",
        )
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if len(devices) == 1 and "EGL_VISIBLE_DEVICES" not in os.environ:
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--id=" + str(devices[0]), "-q", "--xml-format"]
                    )
                    tree = ET.fromstring(out)
                    gpu = tree.findall("gpu")[0]
                    assert gpu is not None
                    minor_number_el = gpu.find("minor_number")
                    assert minor_number_el is not None
                    dev_id = minor_number_el.text
                    os.environ["EGL_VISIBLE_DEVICES"] = str(dev_id)
                except Exception:
                    pass

        super().__init__(windowType="offscreen")
        self.render.set_shader_auto()
        self.render.set_antialias(p3d.core.AntialiasAttrib.MAuto)
        self.render.set_two_sided(True)


def _scene_center_radius_camera(
    root_node: p3d.core.NodePath,
) -> tuple[np.ndarray, float, Optional[p3d.core.NodePath]]:
    bounds = root_node.getBounds()
    radius = float(getattr(bounds, "radius", 0.0) or 0.0)
    if not np.isfinite(radius) or radius <= 1e-4:
        radius = 0.05
    center_pt = bounds.getApproxCenter()
    center = np.array(
        [float(center_pt[0]), float(center_pt[1]), float(center_pt[2])],
        dtype=np.float32,
    )
    camera_np = root_node.find("**/+Camera")
    if camera_np.isEmpty():
        return center, radius, None
    return center, radius, camera_np


def _camera_relative_basis(
    root_node: p3d.core.NodePath,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    center, radius, camera_np = _scene_center_radius_camera(root_node)
    if camera_np is None:
        toward_camera = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        cam_pos_pt = camera_np.getPos(root_node)
        cam_pos = np.array(
            [float(cam_pos_pt[0]), float(cam_pos_pt[1]), float(cam_pos_pt[2])],
            dtype=np.float32,
        )
        toward_camera = cam_pos - center
        toward_camera_norm = float(np.linalg.norm(toward_camera))
        if not np.isfinite(toward_camera_norm) or toward_camera_norm <= 1e-6:
            toward_camera = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        else:
            toward_camera = toward_camera / toward_camera_norm

    helper_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(toward_camera, helper_up))) > 0.95:
        helper_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right = np.cross(helper_up, toward_camera)
    right_norm = float(np.linalg.norm(right))
    if not np.isfinite(right_norm) or right_norm <= 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / right_norm

    up = np.cross(toward_camera, right)
    up_norm = float(np.linalg.norm(up))
    if not np.isfinite(up_norm) or up_norm <= 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        up = up / up_norm

    return center, radius, toward_camera, right, up


def _camera_relative_light_position(
    root_node: p3d.core.NodePath,
    offset: tuple[float, float, float],
    radius_scale: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    center, radius, toward_camera, right, up = _camera_relative_basis(root_node)
    scale = radius * float(radius_scale)
    position = center + (
        scale
        * (
            (float(offset[0]) * right)
            + (float(offset[1]) * toward_camera)
            + (float(offset[2]) * up)
        )
    )
    return position.astype(np.float32), center.astype(np.float32), radius


def _camera_aligned_light_position(
    root_node: p3d.core.NodePath,
    distance_scale: float = 4.8,
) -> tuple[np.ndarray, np.ndarray, float]:
    center, radius, toward_camera, _, _ = _camera_relative_basis(root_node)
    scale = radius * float(distance_scale)
    position = center + (scale * toward_camera)
    return position.astype(np.float32), center.astype(np.float32), radius


def make_scene_lights(
    ambient_light_color: RgbaColor = (0.16, 0.16, 0.16, 1.0),
    point_lights_color: RgbaColor = (1.0, 1.0, 1.0, 1.0),
) -> List[Panda3dLightData]:
    """Creates a camera-relative rig tuned for stable feature visibility.

    MegaPose renders the object from many synthetic views. Static world-space
    lights inevitably leave some of those views with poor relief. This rig is
    anchored to the active render camera instead: one oblique shadow-casting
    key plus a few weak camera-side helpers. The result is much more
    homogeneous across the 72-view render bank while still keeping embossed
    features readable.
    """

    def _set_light_pose(
        root_node: p3d.core.NodePath,
        light_node: p3d.core.NodePath,
        offset: tuple[float, float, float],
        radius_scale: float = 4.0,
        aim_at_center: bool = True,
    ) -> None:
        position, center, radius = _camera_relative_light_position(
            root_node=root_node,
            offset=offset,
            radius_scale=radius_scale,
        )
        light_node.setPos(tuple(position.tolist()))
        if aim_at_center:
            light_node.lookAt(float(center[0]), float(center[1]), float(center[2]))
        light = light_node.node()
        lens = light.get_lens() if hasattr(light, "get_lens") else None
        if lens is not None:
            distance = float(np.linalg.norm(position - center))
            lens.set_near_far(
                max(0.01, radius * 0.30),
                max(radius * 10.0, distance + (radius * 3.5)),
            )

    def frontal_pos_fn(root_node: p3d.core.NodePath, light_node: p3d.core.NodePath) -> None:
        position, center, radius = _camera_aligned_light_position(
            root_node=root_node,
            distance_scale=4.6,
        )
        light_node.setPos(tuple(position.tolist()))
        light_node.lookAt(float(center[0]), float(center[1]), float(center[2]))
        light = light_node.node()
        lens = light.get_lens() if hasattr(light, "get_lens") else None
        if lens is not None:
            distance = float(np.linalg.norm(position - center))
            lens.set_near_far(
                max(0.01, radius * 0.30),
                max(radius * 10.0, distance + (radius * 3.5)),
            )

    def rake_left_pos_fn(root_node: p3d.core.NodePath, light_node: p3d.core.NodePath) -> None:
        _set_light_pose(
            root_node=root_node,
            light_node=light_node,
            offset=(-0.95, 1.05, 0.45),
            radius_scale=4.4,
            aim_at_center=True,
        )

    def rake_right_pos_fn(root_node: p3d.core.NodePath, light_node: p3d.core.NodePath) -> None:
        _set_light_pose(
            root_node=root_node,
            light_node=light_node,
            offset=(0.95, 1.05, 0.45),
            radius_scale=4.4,
            aim_at_center=True,
        )

    def lower_fill_pos_fn(root_node: p3d.core.NodePath, light_node: p3d.core.NodePath) -> None:
        _set_light_pose(
            root_node=root_node,
            light_node=light_node,
            offset=(0.0, 0.85, -0.35),
            radius_scale=4.0,
            aim_at_center=False,
        )

    return [
        Panda3dLightData(light_type="ambient", color=ambient_light_color),
        Panda3dLightData(
            light_type="spot",
            color=(0.20, 0.20, 0.20, 1.0),
            positioning_function=frontal_pos_fn,
            shadow_caster=False,
            lens_fov=72.0,
            exponent=2.0,
            max_distance=1000.0,
        ),
        Panda3dLightData(
            light_type="spot",
            color=(0.48, 0.48, 0.46, 1.0),
            positioning_function=rake_left_pos_fn,
            shadow_caster=False,
            lens_fov=36.0,
            exponent=10.0,
            max_distance=1000.0,
        ),
        Panda3dLightData(
            light_type="spot",
            color=(0.48, 0.48, 0.46, 1.0),
            positioning_function=rake_right_pos_fn,
            shadow_caster=False,
            lens_fov=36.0,
            exponent=10.0,
            max_distance=1000.0,
        ),
        Panda3dLightData(
            light_type="point",
            color=(0.08, 0.08, 0.08, 1.0),
            positioning_function=lower_fill_pos_fn,
            attenuation=(1.0, 0.0, 0.05),
        ),
    ]


class Panda3dSceneRenderer:
    """A class used to render scenes including objects, cameras, lights.

    Rendering is done using panda3d.
    """

    def __init__(
        self,
        asset_dataset: RigidObjectDataset,
        preload_labels: Set[str] = set(),
        debug: bool = False,
        verbose: bool = False,
    ):

        self._asset_dataset = asset_dataset
        self._label_to_node: Dict[str, p3d.core.NodePath] = dict()
        self.verbose = verbose
        self.debug = debug
        self.debug_data = Panda3dDebugData(timings=dict())

        self._cameras_pool: Dict[Resolution, List[Panda3dCamera]] = defaultdict(list)
        if hasattr(builtins, "base"):
            self._app = builtins.base  # type: ignore
        else:
            self._app = App()
        self._app.cam.node().setActive(0)
        self._app.render.clear_light()
        self._rgb_texture = make_rgb_texture_normal_map(size=32)

        assert isinstance(preload_labels, set)
        for label in tqdm(preload_labels, disable=not verbose):
            self.get_object_node(label)

    def create_new_camera(self, resolution: Resolution) -> Panda3dCamera:
        idx = sum([len(x) for x in self._cameras_pool.values()])
        cam = Panda3dCamera.create(f"camera={idx}", resolution=resolution, app=self._app)
        self._cameras_pool[resolution].append(cam)
        return cam

    def get_cameras(self, data_cameras: List[Panda3dCameraData]) -> List[Panda3dCamera]:
        resolution_to_data_cameras: Dict[Resolution, List[Panda3dCameraData]] = defaultdict(list)
        for data_camera in data_cameras:
            resolution_to_data_cameras[data_camera.resolution].append(data_camera)

        for resolution_, data_cameras_ in resolution_to_data_cameras.items():
            for idx in range(len(data_cameras_)):
                if idx >= len(self._cameras_pool[resolution_]):
                    self.create_new_camera(resolution_)

        cameras = []
        available_cameras = {k: v.copy() for k, v in self._cameras_pool.items()}
        for data_camera in data_cameras:
            camera = available_cameras[data_camera.resolution].pop()
            cameras.append(camera)
        return cameras

    def get_object_node(self, label: str) -> p3d.core.NodePath:
        if label in self._label_to_node:
            return self._label_to_node[label]

        asset = self._asset_dataset.get_object_by_label(label)
        scale = asset.scaling_factor_mesh_units_to_meters * asset.scaling_factor
        y, p, r = asset.ypr_offset_deg

        mesh_path = p3d.core.Filename.from_os_specific(str(asset.mesh_path))
        mesh_path.make_true_case()
        node = self._app.loader.load_model(mesh_path, noCache=True)
        node.setScale(scale)
        node.setPos(0, 0, 0)
        node.setHpr(y, p, r)
        self._apply_default_cad_material(node)
        self._label_to_node[label] = node
        return node

    def _apply_default_cad_material(self, node: p3d.core.NodePath) -> None:
        node.setMaterialOff(1)
        node.setTextureOff(1)

        material = p3d.core.Material()
        base = p3d.core.Vec4(0.30, 0.30, 0.30, 1.0)
        material.set_base_color(base)
        material.set_ambient(p3d.core.Vec4(0.10, 0.10, 0.10, 1.0))
        material.set_diffuse(base)
        material.set_specular(p3d.core.Vec3(0.12, 0.12, 0.12))
        material.set_roughness(0.85)
        material.set_shininess(12.0)
        material.set_metallic(0.0)
        material.set_twoside(True)
        node.set_color(base)
        node.set_material(material, 1)
        node.set_two_sided(True)
        node.set_shader(_get_two_sided_cad_shader(), 1)
        node.set_shader_input("u_base_color", base)

    def use_normals_texture(self, obj_node: p3d.core.NodePath) -> p3d.core.NodePath:
        obj_node.setMaterialOff(1)
        obj_node.set_color(p3d.core.Vec4((1.0, 1.0, 1.0, 1.0)))
        obj_node.setTextureOff(1)
        obj_node.setTexGen(p3d.core.TextureStage.getDefault(), p3d.core.TexGenAttrib.MEyeNormal)
        obj_node.setTexture(self._rgb_texture)
        return obj_node

    def setup_scene(
        self, root_node: p3d.core.NodePath, data_objects: List[Panda3dObjectData]
    ) -> List[p3d.core.NodePath]:
        obj_nodes = []
        for n, data_obj in enumerate(data_objects):
            label = data_obj.label
            obj_node = root_node.attach_new_node(f"label={label}-object={n}")
            self.get_object_node(label).instanceTo(obj_node)
            if data_obj.remove_mesh_material:
                obj_node.setMaterialOff(1)
            TWO = np_to_lmatrix4(data_obj.TWO.toHomogeneousMatrix())
            obj_node.setMat(TWO)
            if data_obj.positioning_function is not None:
                data_obj.positioning_function(root_node, obj_node)
            obj_node.setScale(data_obj.scale)
            if data_obj.color is not None:
                data_obj.set_node_material_and_transparency(obj_node)
            obj_nodes.append(obj_node)
        return obj_nodes

    def setup_cameras(
        self, root_node: p3d.core.NodePath, data_cameras: List[Panda3dCameraData]
    ) -> List[Panda3dCamera]:
        cameras = self.get_cameras(data_cameras)

        for data_camera, camera in zip(data_cameras, cameras):
            camera_node_path = camera.node_path
            camera_node_path.node().setActive(1)
            camera_node_path.reparentTo(root_node)

            data_camera.set_lens_parameters(camera_node_path.node().getLens())
            view_mat = data_camera.compute_view_mat()
            camera_node_path.setMat(view_mat)
            if data_camera.positioning_function is not None:
                data_camera.positioning_function(root_node, camera_node_path)
        return cameras

    def render_images(
        self, cameras: List[Panda3dCamera], copy_arrays: bool = True, render_depth: bool = False
    ) -> List[CameraRenderingData]:

        self._app.graphicsEngine.renderFrame()
        self._app.graphicsEngine.syncFrame()

        renderings = []
        for camera in cameras:
            rgb = camera.get_rgb_image()
            if copy_arrays:
                rgb = rgb.copy()
            rendering = CameraRenderingData(rgb)

            if render_depth:
                rendering.depth = camera.get_depth_image()
            renderings.append(rendering)
        return renderings

    def setup_lights(
        self, root_node: p3d.core, light_datas: List[Panda3dLightData]
    ) -> List[p3d.core.NodePath]:
        light_node_paths = []
        for n, light_data in enumerate(light_datas):
            if light_data.light_type == "point":
                light_node = p3d.core.PointLight(f"{n}_point")
                assert light_data.positioning_function is not None
            elif light_data.light_type == "ambient":
                light_node = p3d.core.AmbientLight(f"{n}_ambient")
            elif light_data.light_type == "directional":
                light_node = p3d.core.DirectionalLight("{n}_directional")
                assert light_data.positioning_function is not None
            elif light_data.light_type == "spot":
                light_node = p3d.core.Spotlight(f"{n}_spot")
                lens = p3d.core.PerspectiveLens()
                if light_data.lens_fov is not None:
                    lens.set_fov(float(light_data.lens_fov))
                if light_data.near_far is not None:
                    lens.set_near_far(float(light_data.near_far[0]), float(light_data.near_far[1]))
                light_node.set_lens(lens)
                if light_data.exponent is not None:
                    light_node.set_exponent(float(light_data.exponent))
                if light_data.max_distance is not None:
                    light_node.set_max_distance(float(light_data.max_distance))
                assert light_data.positioning_function is not None
            else:
                raise NotImplementedError(light_data.light_type)

            light_node.set_color(light_data.color)
            if light_data.light_type == "point" and light_data.attenuation is not None:
                light_node.set_attenuation(light_data.attenuation)
                light_node.set_specular_color(light_data.color)
            elif light_data.light_type == "spot":
                light_node.set_specular_color(light_data.color)
            if light_data.shadow_caster:
                shadow_buffer_size = light_data.shadow_buffer_size or (1024, 1024)
                light_node.set_shadow_caster(True, int(shadow_buffer_size[0]), int(shadow_buffer_size[1]))
            light_node_path = root_node.attach_new_node(light_node)
            root_node.set_light(light_node_path)
            if light_data.positioning_function is not None:
                light_data.positioning_function(root_node, light_node_path)
            light_node_paths.append(light_node_path)
        return light_node_paths

    def render_scene(
        self,
        object_datas: List[Panda3dObjectData],
        camera_datas: List[Panda3dCameraData],
        light_datas: List[Panda3dLightData],
        render_depth: bool = False,
        copy_arrays: bool = True,
        render_binary_mask: bool = False,
        render_normals: bool = False,
        clear: bool = True,
    ) -> List[CameraRenderingData]:

        start = time.time()
        root_node = self._app.render.attachNewNode("world")
        object_nodes = self.setup_scene(root_node, object_datas)
        cameras = self.setup_cameras(root_node, camera_datas)
        light_nodes = self.setup_lights(root_node, light_datas)
        if any(bool(getattr(light_data, "shadow_caster", False)) for light_data in light_datas):
            for object_node in object_nodes:
                try:
                    object_node.setDepthOffset(2)
                except Exception:
                    pass
        setup_time = time.time() - start

        start = time.time()
        renderings = self.render_images(cameras, copy_arrays=copy_arrays, render_depth=render_depth)
        if render_normals:
            for object_node in object_nodes:
                self.use_normals_texture(object_node)
                root_node.clear_light()
                light_data = Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))
                light_nodes += self.setup_lights(root_node, [light_data])
            normals_renderings = self.render_images(cameras, copy_arrays=copy_arrays)
            for n, rendering in enumerate(renderings):
                rendering.normals = normals_renderings[n].rgb

        if render_binary_mask:
            for rendering_n in renderings:
                assert rendering_n.depth is not None
                h, w = rendering_n.depth.shape[:2]
                binary_mask = np.zeros((h, w), dtype=np.bool_)
                binary_mask[rendering_n.depth[..., 0] > 0] = 1
                rendering.binary_mask = binary_mask

        render_time = time.time() - start

        if clear:
            for camera in cameras:
                camera.node_path.node().setActive(0)
            for object_node in object_nodes:
                object_node.clear_texture()  # TODO: Is this necessary ?
                object_node.clear_light()  # TODO: Is this necessary ?
                object_node.detach_node()
            for light_node in light_nodes:
                light_node.detach_node()
            root_node.clear_light()
            root_node.detach_node()

            for _ in range(3):
                # TODO: Is this necessary ?
                p3d.core.RenderState.garbageCollect()
                p3d.core.TransformState.garbageCollect()

        self.debug_data.timings["setup_time"] = setup_time
        self.debug_data.timings["render_time"] = render_time
        return renderings
