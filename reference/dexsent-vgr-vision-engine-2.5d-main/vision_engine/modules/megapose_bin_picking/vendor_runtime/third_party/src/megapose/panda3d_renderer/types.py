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
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

# Third Party
import builtins
import time

import numpy as np
import panda3d as p3d
import panda3d.core
from direct.showbase.ShowBase import ShowBase

# MegaPose
from megapose.lib3d.transform import Transform

# Local Folder
from .utils import depth_image_from_depth_buffer

RgbaColor = Tuple[float, float, float, float]
NodeFunction = Callable[
    [p3d.core.NodePath, p3d.core.NodePath], None
]  # (root_node_path, object_node_path)
Resolution = Tuple[int, int]

TCCGL = Transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float))


@dataclass
class CameraRenderingData:
    """
    rgb: (h, w, 3) uint8
    normals: (h, w, 3) uint8
    depth: (h, w, 1) float32
    binary_mask: (h, w, 1) np.bool_
    """

    rgb: np.ndarray
    normals: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None


@dataclass
class Panda3dCameraData:
    K: np.ndarray
    resolution: Tuple[int, int]
    TWC: Transform = Transform((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    z_near: float = 0.01
    z_far: float = 10
    node_name: str = "camera"
    positioning_function: Optional[NodeFunction] = None

    def compute_view_mat(self) -> p3d.core.LMatrix4f:
        assert self.TWC is not None
        TWCGL = self.TWC * TCCGL
        view_mat = TWCGL.toHomogeneousMatrix()
        view_mat = p3d.core.LMatrix4f(*view_mat.transpose().flatten().tolist())
        return view_mat

    def set_lens_parameters(self, lens: p3d.core.Lens) -> p3d.core.Lens:
        # NOTE: inspired from http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
        # https://discourse.panda3d.org/t/lens-camera-for-opencv-style-camera-parameterisation/15413
        near, far = self.z_near, self.z_far
        lens.set_far(far)
        lens.set_near(near)

        h, w = self.resolution
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = h - self.K[1, 2]
        A = (far + near) / (far - near)
        B = -2 * (far * near) / (far - near)
        user_mat = np.array(
            [
                [fx, 0, 0, 0],
                [0, 0, A, 1],
                [0, fy, 0, 0],
                [0, 0, B, 0],
            ]
        )

        lens.setFilmSize(w, h)
        lens.setUserMat(p3d.core.LMatrix4f(*user_mat.flatten().tolist()))
        lens.setFilmOffset(w * 0.5 - cx, h * 0.5 - cy)
        return


@dataclass
class Panda3dLightData:
    """Data used to to define a light in a panda3d scene.
    light_type: ambient, point, directional, or spot
    NOTE: Alpha is largely irrelevant
    https://docs.panda3d.org/1.10/python/programming/render-attributes/lighting#colored-lights
    """

    light_type: str
    color: RgbaColor = (1.0, 1.0, 1.0, 1.0)
    positioning_function: Optional[NodeFunction] = None
    attenuation: Optional[tuple[float, float, float]] = None
    shadow_caster: bool = False
    shadow_buffer_size: Optional[tuple[int, int]] = None
    lens_fov: Optional[float] = None
    near_far: Optional[tuple[float, float]] = None
    exponent: Optional[float] = None
    max_distance: Optional[float] = None


@dataclass
class Panda3dObjectData:
    label: str
    TWO: Transform = Transform((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    color: Optional[RgbaColor] = None
    material: Optional[p3d.core.Material] = None
    remove_mesh_material: bool = False
    scale: float = 1
    positioning_function: Optional[NodeFunction] = None

    def set_node_material_and_transparency(
        self,
        node_path: p3d.core.NodePath,
    ) -> p3d.core.NodePath:
        assert self.color is not None
        material = p3d.core.Material()
        material.set_ambient(p3d.core.Vec4(*self.color))
        material.set_diffuse(p3d.core.Vec4(*self.color))
        material.set_specular(p3d.core.Vec3(1, 1, 1))
        material.set_roughness(0.4)
        material.set_twoside(True)

        node_path.set_color(p3d.core.Vec4(*self.color))  # TODO: Is this necessary ?
        node_path.set_material(material, 1)
        node_path.set_two_sided(True)
        if self.color[3] < 1:
            node_path.set_transparency(p3d.core.TransparencyAttrib.M_alpha)
        return node_path


@dataclass
class Panda3dCamera:
    node_path: p3d.core.Camera
    display_region: p3d.core.DisplayRegion
    window_properties: p3d.core.WindowProperties
    graphics_buffer: p3d.core.GraphicsOutput
    resolution: Resolution
    texture: p3d.core.Texture
    depth_texture: p3d.core.Texture

    def _refresh_render_once(self) -> None:
        app = getattr(builtins, "base", None)
        if app is None:
            return
        try:
            app.graphicsEngine.renderFrame()
            app.graphicsEngine.syncFrame()
        except Exception:
            return

    @staticmethod
    def _buffer_length(raw: object) -> int:
        if raw is None:
            return 0
        try:
            return len(memoryview(raw))
        except Exception:
            try:
                return len(raw)  # type: ignore[arg-type]
            except Exception:
                return 0

    @staticmethod
    def create(
        name: str,
        resolution: Resolution,
        app: Optional[ShowBase] = None,
    ) -> "Panda3dCamera":
        if app is None:
            app = base  # type: ignore # noqa: F821
        window_props = p3d.core.WindowProperties.getDefault()
        resolution_ = (resolution[1], resolution[0])
        window_props.setSize(*resolution_)

        frame_buffer_props = p3d.core.FrameBufferProperties()
        frame_buffer_props.setRgbColor(True)
        frame_buffer_props.setRgbaBits(8, 8, 8, 8)
        frame_buffer_props.setDepthBits(24)
        frame_buffer_props.setStencilBits(8)
        frame_buffer_props.setMultisamples(0)

        graphics_buffer = app.graphicsEngine.make_output(
            app.pipe,
            f"Graphics Buffer [{name}]",
            -2,
            frame_buffer_props,
            window_props,
            p3d.core.GraphicsPipe.BFRefuseWindow,
            app.win.getGsg(),
            app.win,
        )
        if graphics_buffer is None:
            fallback_props = p3d.core.FrameBufferProperties()
            fallback_props.setRgbColor(True)
            fallback_props.setDepthBits(24)
            graphics_buffer = app.graphicsEngine.make_output(
                app.pipe,
                f"Graphics Buffer Fallback [{name}]",
                -2,
                fallback_props,
                window_props,
                p3d.core.GraphicsPipe.BFRefuseWindow,
                app.win.getGsg(),
                app.win,
            )
        if graphics_buffer is None:
            raise RuntimeError(
                "Panda3D could not create an offscreen graphics buffer on this system."
            )

        texture = p3d.core.Texture()
        graphics_buffer.addRenderTexture(
            texture,
            p3d.core.GraphicsOutput.RTMCopyRam,
        )

        depth_texture = p3d.core.Texture()
        depth_texture.setFormat(p3d.core.Texture.FDepthComponent)
        graphics_buffer.addRenderTexture(
            depth_texture, p3d.core.GraphicsOutput.RTMCopyRam, p3d.core.GraphicsOutput.RTPDepth
        )

        cam_node = p3d.core.Camera(f"Camera [{name}]")
        lens = p3d.core.MatrixLens()
        cam_node.setLens(lens)

        cam_node_path = app.camera.attachNewNode(cam_node)
        cam_node_path.reparentTo(app.render)

        display_region = graphics_buffer.make_display_region(0, 1, 0, 1)
        display_region.set_sort(0)
        display_region.set_camera(cam_node_path)

        return Panda3dCamera(
            node_path=cam_node_path,
            display_region=display_region,
            window_properties=window_props,
            graphics_buffer=graphics_buffer,
            resolution=resolution,
            texture=texture,
            depth_texture=depth_texture,
        )

    def get_rgb_image(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: (h, w, 3) uint8 array
        """
        xsize = int(self.texture.getXSize() or self.resolution[1])
        ysize = int(self.texture.getYSize() or self.resolution[0])
        expected = xsize * ysize * 3
        for attempt in range(4):
            image = self.texture.get_ram_image_as("rgb")
            if self._buffer_length(image) == expected:
                array = np.frombuffer(image, dtype=np.uint8).reshape((ysize, xsize, 3))
                return np.flipud(array)
            screenshot = self.display_region.get_screenshot()
            sx = screenshot.get_x_size()
            sy = screenshot.get_y_size()
            screenshot_image = screenshot.get_ram_image_as("rgb")
            if self._buffer_length(screenshot_image) == sx * sy * 3:
                array = np.frombuffer(screenshot_image, dtype=np.uint8).reshape((sy, sx, 3))
                return np.flipud(array)
            self._refresh_render_once()
            time.sleep(0.01 * (attempt + 1))
        raise RuntimeError(
            f"panda3d_empty_rgb_buffer: expected={expected} bytes got={self._buffer_length(image)}"
        )

    def _get_depth_buffer(self) -> np.ndarray:
        """Extracts a depth buffer image from the depth texture.

        See link below for explanation of values in depth_buffer. This is
        NOT a depth image in the computer vision sense.

        https://developer.nvidia.com/content/depth-precision-visualized#:~:text=GPU%20hardware%20depth%20buffers%20don,reciprocal%20of%20world%2Dspace%20depth.

        Returns:
            depth_buffer: [H,W,1] numpy array with values in [0,1]

        """
        width = int(self.depth_texture.getXSize() or self.resolution[1])
        height = int(self.depth_texture.getYSize() or self.resolution[0])
        components = max(1, int(self.depth_texture.getNumComponents() or 1))
        expected = width * height * components * np.dtype(np.float32).itemsize
        data = None
        for attempt in range(4):
            data = self.depth_texture.getRamImage()
            if self._buffer_length(data) == expected:
                depth_buffer = np.frombuffer(data, np.float32)
                depth_buffer.shape = (height, width, components)
                return np.flipud(depth_buffer)
            self._refresh_render_once()
            time.sleep(0.01 * (attempt + 1))
        raise RuntimeError(
            "panda3d_empty_depth_buffer: "
            f"expected={expected} bytes got={self._buffer_length(data)}"
        )

    def get_depth_image(self, eps: float = 0.001) -> np.ndarray:
        depth_buffer = self._get_depth_buffer()
        lens = self.node_path.node().getLens()
        z_near, z_far = lens.getNear(), lens.getFar()
        return depth_image_from_depth_buffer(depth_buffer, z_near, z_far, eps=eps)
