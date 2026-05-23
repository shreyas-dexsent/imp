from __future__ import annotations

import numpy as np

from vision_engine.modules.megapose_bin_picking.runtime import (
    _tensor_image_to_u8,
    build_camera_data,
    resolve_object_assets,
    select_detection_candidate,
)


def test_resolve_object_assets_prefers_object_named_files(tmp_path):
    object_dir = tmp_path / "widget"
    object_dir.mkdir()
    (object_dir / "object.pt").write_bytes(b"pt")
    (object_dir / "object.glb").write_bytes(b"mesh")

    assets = resolve_object_assets(object_dir)

    assert assets.object_folder == object_dir.resolve()
    assert assets.segmentation_model_path.name == "object.pt"
    assert assets.mesh_path.name == "object.glb"
    assert assets.label == "widget"


def test_resolve_object_assets_falls_back_to_folder_name(tmp_path):
    object_dir = tmp_path / "gear"
    object_dir.mkdir()
    (object_dir / "gear.pt").write_bytes(b"pt")
    (object_dir / "gear.obj").write_text("v 0 0 0\n", encoding="utf-8")

    assets = resolve_object_assets(object_dir, label_override="part_gear")

    assert assets.segmentation_model_path.name == "gear.pt"
    assert assets.mesh_path.name == "gear.obj"
    assert assets.label == "part_gear"


def test_resolve_object_assets_honors_megapose_config_mesh_override(tmp_path):
    object_dir = tmp_path / "seal"
    object_dir.mkdir()
    (object_dir / "seal.pt").write_bytes(b"pt")
    (object_dir / "seal.stl").write_text("solid seal\nendsolid seal\n", encoding="utf-8")
    (object_dir / "converted.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (object_dir / "megapose.json").write_text(
        '{"mesh_path": "converted.obj"}',
        encoding="utf-8",
    )

    assets = resolve_object_assets(object_dir)

    assert assets.mesh_path.name == "converted.obj"
    assert assets.segmentation_model_path.name == "seal.pt"


def test_select_detection_candidate_defaults_to_highest_segmentation_score():
    detections = [
        {"rank": 0, "score": 0.95, "area_px": 1000.0, "bbox_xywh": [0, 0, 10, 10]},
        {"rank": 1, "score": 0.90, "area_px": 3000.0, "bbox_xywh": [0, 0, 20, 15]},
        {"rank": 2, "score": 0.85, "area_px": 2000.0, "bbox_xywh": [0, 0, 15, 12]},
        {"rank": 3, "score": 0.80, "area_px": 9000.0, "bbox_xywh": [0, 0, 50, 20]},
    ]

    selected = select_detection_candidate(
        detections,
        image_shape=(64, 64),
        top_k=3,
        min_confidence=0.7,
    )

    assert selected is not None
    assert selected["rank"] == 0
    assert len(selected["selection_pool"]) == 3


def test_select_detection_candidate_ignores_tiny_near_depth_outliers():
    depth_m = np.full((20, 20), 0.30, dtype=np.float32)
    depth_m[0, 0] = 0.10
    depth_m[0, 1] = 0.10
    depth_m[10:15, 10:15] = 0.25
    detections = [
        {
            "rank": 0,
            "score": 0.95,
            "area_px": 100.0,
            "bbox_xyxy": [0, 0, 10, 10],
            "bbox_xywh": [0, 0, 10, 10],
            "mask": np.pad(np.ones((10, 10), dtype=bool), ((0, 10), (0, 10))),
        },
        {
            "rank": 1,
            "score": 0.92,
            "area_px": 100.0,
            "bbox_xyxy": [10, 10, 20, 20],
            "bbox_xywh": [10, 10, 10, 10],
            "mask": np.pad(np.ones((10, 10), dtype=bool), ((10, 0), (10, 0))),
        },
    ]

    selected = select_detection_candidate(
        detections,
        image_shape=depth_m.shape,
        depth_m=depth_m,
        K=np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        top_k=2,
        min_confidence=0.8,
        selection_mode="high_confidence_highest_z",
        high_confidence_gate=0.8,
    )

    assert selected is not None
    assert selected["rank"] == 1
    assert selected["closest_region_distance_m"] < selected["robust_region_distance_m"]


def test_build_camera_data_scales_intrinsics_to_frame_shape():
    camera = build_camera_data(
        {
            "intrinsics": {
                "fx": 640.0,
                "fy": 640.0,
                "cx": 320.0,
                "cy": 240.0,
                "resolution": {"width": 640, "height": 480},
            }
        },
        (960, 1280),
    )

    k = camera["K"]
    np.testing.assert_allclose(k[0, 0], 1280.0)
    np.testing.assert_allclose(k[1, 1], 1280.0)
    np.testing.assert_allclose(k[0, 2], 640.0)
    np.testing.assert_allclose(k[1, 2], 480.0)


def test_build_camera_data_accepts_camera_data_resolution_as_height_width():
    camera = build_camera_data(
        {
            "camera_data": {
                "K": [
                    [615.0, 0.0, 320.0],
                    [0.0, 615.0, 240.0],
                    [0.0, 0.0, 1.0],
                ],
                "resolution": [480, 640],
                "depth_scale_m_per_unit": 0.001,
            }
        },
        (480, 640),
    )

    k = camera["K"]
    np.testing.assert_allclose(k[0, 0], 615.0)
    np.testing.assert_allclose(k[1, 1], 615.0)
    np.testing.assert_allclose(k[0, 2], 320.0)
    np.testing.assert_allclose(k[1, 2], 240.0)


def test_build_camera_data_prefers_explicit_intrinsics_over_frame_camera_data():
    camera = build_camera_data(
        {
            "K": [
                [657.59, 0.0, 635.67],
                [0.0, 658.35, 354.13],
                [0.0, 0.0, 1.0],
            ],
            "intrinsics": {
                "fx": 657.59,
                "fy": 658.35,
                "cx": 635.67,
                "cy": 354.13,
                "resolution": {"width": 1280, "height": 720},
            },
            "camera_data": {
                "K": [
                    [651.84, 0.0, 631.87],
                    [0.0, 650.17, 355.31],
                    [0.0, 0.0, 1.0],
                ],
                "resolution": [720, 1280],
                "source": "pyrealsense2",
            },
        },
        (720, 1280),
    )

    k = camera["K"]
    np.testing.assert_allclose(k[0, 0], 657.59)
    np.testing.assert_allclose(k[1, 1], 658.35)
    np.testing.assert_allclose(k[0, 2], 635.67)
    np.testing.assert_allclose(k[1, 2], 354.13)


def test_tensor_image_to_u8_decodes_multichannel_channel_first_render():
    render = np.zeros((6, 24, 32), dtype=np.float32)
    render[0, :, :] = 0.10
    render[1, :, :] = 0.40
    render[2, :, :] = 0.90
    render[3, :, :] = 0.75

    image = _tensor_image_to_u8(render)

    assert image.shape == (24, 32, 3)
    np.testing.assert_array_equal(image[0, 0], np.array([26, 102, 230], dtype=np.uint8))
