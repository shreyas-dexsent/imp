from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vision_engine.modules.ppf_icp_bin_picking import runtime as ppf_runtime


def test_ppf_dependency_guard_reports_missing_contrib(monkeypatch):
    monkeypatch.setattr(ppf_runtime, "cv2", SimpleNamespace(), raising=False)

    assert ppf_runtime.ppf_dependency_available() is False


def test_build_cad_model_cloud_uses_mesh_units_and_scale(tmp_path):
    import trimesh

    mesh_path = tmp_path / "box.obj"
    mesh = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
    mesh.export(mesh_path)

    cloud = ppf_runtime.build_cad_model_cloud(
        mesh_path,
        mesh_units="mm",
        mesh_scale=2.0,
        sample_points=512,
    )

    assert cloud.shape == (512, 6)
    assert np.isfinite(cloud).all()
    np.testing.assert_array_less(np.max(np.abs(cloud[:, :3]), axis=0), [0.011, 0.021, 0.031])
    np.testing.assert_allclose(np.linalg.norm(cloud[:, 3:6], axis=1), 1.0, atol=1e-5)


def test_build_scene_cloud_masks_depth_caps_points_and_generates_normals(monkeypatch):
    def fake_compute_normals(points, *_args):
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(points), 1))
        return np.concatenate([points, normals], axis=1).astype(np.float32)

    monkeypatch.setattr(
        ppf_runtime,
        "_lookup_ppf_api",
        lambda: ppf_runtime.PpfApi(
            detector_ctor=object,
            icp_ctor=object,
            compute_normals=fake_compute_normals,
        ),
    )
    depth_m = np.full((4, 4), 0.25, dtype=np.float32)
    depth_m[0, 0] = 0.0
    depth_m[3, 3] = 2.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:4, 1:4] = True
    k = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    scene, info = ppf_runtime.build_scene_cloud(
        depth_m,
        k,
        mask,
        {
            "depth_max_m": 1.0,
            "ppf_scene_voxel_size_m": 0.0,
            "ppf_scene_max_points": 3,
            "ppf_normal_neighbors": 8,
        },
    )

    assert scene.shape == (3, 6)
    assert info["raw_point_count"] == 8
    assert info["point_count"] == 3
    np.testing.assert_allclose(scene[:, 5], 1.0)


def test_mocked_ppf_icp_pose_emits_match_schema(monkeypatch, tmp_path):
    pose_a = np.eye(4, dtype=np.float32)
    pose_a[:3, 3] = [0.10, 0.20, 0.30]
    pose_b = np.eye(4, dtype=np.float32)
    pose_b[:3, 3] = [0.11, 0.20, 0.30]

    class FakePose:
        def __init__(self, pose, votes):
            self.pose = pose
            self.numVotes = votes
            self.residual = 0.2

    class FakeDetector:
        def match(self, *_args):
            return [FakePose(pose_b, 2), FakePose(pose_a, 5)]

    class FakeIcp:
        def registerModelToScene(self, *_args):
            return True, 0.001, np.eye(4, dtype=np.float32)

    monkeypatch.setattr(
        ppf_runtime,
        "_lookup_ppf_api",
        lambda: ppf_runtime.PpfApi(
            detector_ctor=FakeDetector,
            icp_ctor=FakeIcp,
            compute_normals=lambda points, *_args: points,
        ),
    )
    model_pc = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.01, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    bundle = ppf_runtime.PpfModelBundle(
        detector=FakeDetector(),
        model_pc=model_pc,
        mesh_path=tmp_path / "object.obj",
        mesh_units="m",
        mesh_scale=1.0,
        cache_key=("fake",),
        lock=ppf_runtime.threading.Lock(),
    )
    scene_pc = model_pc.copy()

    hyp = ppf_runtime._run_ppf_icp(bundle=bundle, scene_pc=scene_pc, params={"icp_enabled": True})

    assert hyp is not None
    assert hyp.icp_applied is True
    assets = SimpleNamespace(
        label="part",
        mesh_path=tmp_path / "object.obj",
        segmentation_model_path=tmp_path / "object.pt",
    )
    camera_data = {
        "K": np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0},
    }
    mask = np.ones((4, 4), dtype=bool)
    match = ppf_runtime._make_match(
        params={"object_id": "part", "axis": [0, 0, 1]},
        assets=assets,
        candidate={"rank": 0, "score": 0.9, "area_px": 16.0, "bbox_xywh": [0, 0, 4, 4]},
        candidate_data={
            "bbox_xyxy": np.array([0, 0, 4, 4], dtype=np.float32),
            "bbox_xywh": [0, 0, 4, 4],
            "selection_mask": mask,
            "segmentation_contours_uv": [[[0, 0], [3, 0], [3, 3]]],
            "cluster_filter_info": None,
        },
        pose_hypothesis=hyp,
        camera_data=camera_data,
        annotation_meta={
            "object_center_object_m": [0.0, 0.0, 0.0],
            "object_extents_m": [0.01, 0.01, 0.01],
            "annotation_axis_length_m": 0.02,
        },
        duplicate_filter_info={"kept_count": 1},
        scene_info={"point_count": 2},
        model_bundle=bundle,
        depth_m=np.full((4, 4), 0.3, dtype=np.float32),
        request_id="test",
    )

    assert match["method"] == "ppf_icp_bin_picking"
    assert match["center_xyz_m"] == match["pose_origin_xyz_m"]
    assert len(match["pose_quat_xyzw"]) == 4
    assert match["ppf_icp"]["icp_residual"] == 0.001
    assert "safety_pcd" in match


def test_no_candidate_returns_retry_error(monkeypatch, tmp_path):
    fake_assets = SimpleNamespace(
        object_folder=tmp_path,
        label="part",
        mesh_path=tmp_path / "object.obj",
        segmentation_model_path=tmp_path / "object.pt",
    )
    fake_bundle = SimpleNamespace(
        model_pc=np.zeros((8, 6), dtype=np.float32),
        cache_key=("fake",),
    )
    monkeypatch.setattr(
        ppf_runtime,
        "_lookup_ppf_api",
        lambda: ppf_runtime.PpfApi(
            detector_ctor=object,
            icp_ctor=object,
            compute_normals=lambda points, *_args: points,
        ),
    )
    monkeypatch.setattr(ppf_runtime, "resolve_object_assets", lambda *args, **kwargs: fake_assets)
    monkeypatch.setattr(ppf_runtime, "get_ppf_model_bundle", lambda *args, **kwargs: fake_bundle)
    monkeypatch.setattr(
        ppf_runtime,
        "get_mesh_annotation_meta",
        lambda *args, **kwargs: {"object_center_object_m": [0, 0, 0], "object_extents_m": [1, 1, 1]},
    )
    monkeypatch.setattr(ppf_runtime, "run_segmentation", lambda *args, **kwargs: [])

    result = ppf_runtime.run_ppf_icp_bin_picking(
        bgr=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.full((4, 4), 300, dtype=np.uint16),
        params={
            "object_folder": str(tmp_path),
            "K": [[100, 0, 2], [0, 100, 2], [0, 0, 1]],
            "depth_scale": 0.001,
        },
        request_id="test",
    )

    assert result["valid"] is False
    assert result["terminal"] is True
    assert result["error"] == "ppf_icp_no_candidate"

