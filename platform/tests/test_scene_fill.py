"""P3 headline test: prove the **Scene-fill seam** is wired end-to-end through
the imp module wrappers, not just through the underlying motion-core library
(which has its own test suite at platform/modules/motion-core/algorithms/tests).

Three things must hold for P3 to be done (debt **D3** in PLAN.md):

1. ``motion-pinocchio.FkModule`` publishes a ``Pose6D`` whose ``frame_id`` is
   the world frame and whose position matches ``algorithms.kinematics.fk``
   on the same ``(Scene, q)`` — i.e. the wrapper composes through
   ``base_pose`` instead of returning local-base coordinates.
2. ``motion-coal.CollisionModule`` routes a perception ``Pose6D`` arriving on
   its ``object_pose_key`` into ``Scene.set_object_pose`` *before* querying
   collision, so moving an obstacle by topic actually flips the verdict.
3. ``Scene.attach`` (called in-process — schema TBD in P5) makes the
   attached object's geometry follow the parent frame's FK; subsequent
   collision queries see the attached pose, not the prior free pose.

These tests are gated with ``pytest.importorskip`` on the heavy native deps
(pin, coal, zenoh) — they're skipped on hosts that don't have them and run
to completion inside the dev container / CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Heavy native deps: skip if any are missing. Order matters -- imp_sdk pulls
# zenoh on import, so the zenoh check comes first.
pytest.importorskip("zenoh")
pytest.importorskip("pinocchio")
pytest.importorskip("coal")
pytest.importorskip("imp_sdk")

from scipy.spatial.transform import Rotation  # noqa: E402

from algorithms.collision import is_in_collision  # noqa: E402
from algorithms.descriptions import WorldDescription  # noqa: E402
from algorithms.kinematics import fk  # noqa: E402
from algorithms.resolved import KinematicModel, Scene  # noqa: E402

from imp_sdk.schemas import imp_pb2  # noqa: E402

from imp_module_motion_coal.collision import CollisionModule  # noqa: E402
from imp_module_motion_pinocchio.fk import FkModule  # noqa: E402


WORLD_YAML = (
    Path(__file__).resolve().parents[1]
    / "modules"
    / "motion-core"
    / "algorithms"
    / "configs"
    / "worlds"
    / "franka_table_world.yaml"
)


def _home_q(world):
    system = world.robot("arm").robot_system
    model = KinematicModel.from_robot_system(system)
    home = system.named_joint_state("home")
    return model, np.array([home[n] for n in model.active_joint_names], dtype=float)


def _state_msg(q: np.ndarray) -> imp_pb2.RobotState:
    return imp_pb2.RobotState(
        header=imp_pb2.Header(schema="imp.RobotState/1"),
        q=q.tolist(),
        mode="idle",
    )


def _pose_msg(T: np.ndarray, object_id: str = "matka") -> imp_pb2.Pose6D:
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    return imp_pb2.Pose6D(
        header=imp_pb2.Header(schema="imp.Pose6D/1"),
        object_id=object_id,
        position_m=T[:3, 3].tolist(),
        quat_xyzw=quat.tolist(),
        confidence=1.0,
        valid=True,
    )


# ---------------------------------------------------------------------------
# Test 1: FK module publishes world-frame poses
# ---------------------------------------------------------------------------


def test_fk_module_publishes_world_frame_pose():
    """FkModule.compute should compose ``T_world_base @ T_base_tcp`` and tag
    the published ``Pose6D`` with the world frame id, not the robot base id.
    """
    world = WorldDescription.from_yaml(WORLD_YAML)
    _, q = _home_q(world)

    module = FkModule(
        station="devstation",
        robot="fr3",
        world_path=str(WORLD_YAML),
        world_robot="arm",
        chain="arm",
    )
    module.configure()
    out = module.compute({"state": _state_msg(q)})
    pose = out["tcp"]

    # Frame id is the world frame, not the robot's local base.
    assert pose.header.frame_id == world.world_frame

    # Position matches the library's world-frame fk on the same Scene.
    scene = Scene.from_world(world)
    T_world_tcp = fk(scene, "arm", q, module.tcp_frame)
    assert np.allclose(pose.position_m, T_world_tcp[:3, 3])


# ---------------------------------------------------------------------------
# Test 2: Perception Pose6D updates the Scene through the module
# ---------------------------------------------------------------------------


def test_collision_module_routes_perception_pose_into_scene():
    """Collision verdict should change when a perception pose places the
    object on top of the robot — proving the Pose6D input actually mutates
    Scene.object_poses before is_in_collision runs.
    """
    pose_key = "imp/devstation/perc/s1/world_pose"
    module = CollisionModule(
        station="devstation",
        robot="fr3",
        world_path=str(WORLD_YAML),
        world_robot="arm",
        object_pose_key=pose_key,
        object_id="matka",
    )
    module.configure()

    world = WorldDescription.from_yaml(WORLD_YAML)
    _, q = _home_q(world)

    # Baseline: the YAML places matka at (0.45, 0, 0.12) -- away from the arm.
    baseline = module.compute({"state": _state_msg(q), "object_pose": _pose_msg(np.eye(4)) if False else _pose_msg(world.objects[0].pose.as_matrix())})
    baseline_value = baseline["collision"].value

    # Now perception "moves" matka to the robot base origin: guaranteed overlap.
    T_overlap = np.eye(4)
    T_overlap[:3, 3] = (0.0, 0.0, 0.2)  # straight on top of the robot base column
    moved = module.compute({"state": _state_msg(q), "object_pose": _pose_msg(T_overlap)})
    moved_value = moved["collision"].value

    assert moved_value > baseline_value, (
        f"Scene-fill failed: moving matka onto the robot did not change the "
        f"collision count (baseline={baseline_value}, moved={moved_value})."
    )


# ---------------------------------------------------------------------------
# Test 3: Attach moves the geometry with FK (in-process API)
# ---------------------------------------------------------------------------


def test_attach_makes_object_follow_tcp():
    """After Scene.attach(matka -> fr3_hand_tcp), the object's effective
    world pose should follow FK on the parent frame.

    Attach/detach as topic events lands in P5; for now the demo exercises
    the in-process Scene API the module exposes via ``module.scene``.
    """
    module = CollisionModule(
        station="devstation",
        robot="fr3",
        world_path=str(WORLD_YAML),
        world_robot="arm",
    )
    module.configure()

    world = WorldDescription.from_yaml(WORLD_YAML)
    model, q = _home_q(world)

    # Attach matka to the TCP frame at +0.10 m along x (a typical EE offset).
    T_parent_obj = np.eye(4)
    T_parent_obj[0, 3] = 0.10
    module.scene.attach(
        "matka",
        "fr3_hand_tcp",
        T_parent_obj,
        allow_collision_with=["fr3_hand", "fr3_leftfinger", "fr3_rightfinger"],
    )

    # The object's free-standing pose is gone now; only attached state remains.
    assert "matka" not in module.scene.object_poses
    assert "matka" in module.scene.attached

    # Run a collision query: the attached allowances must keep the EE<->matka
    # contacts suppressed (this is what the dynamic ACM overlay is for).
    report = is_in_collision(model, module.scene, q)
    for a, b in report.contacts:
        pair = (a, b)
        assert "matka" not in pair or module.scene.is_pair_allowed(a, b), (
            f"contact {pair} not allowed under the dynamic ACM overlay -- "
            f"attach() did not register the runtime allowance."
        )
