from pathlib import Path

import pytest
from pydantic import ValidationError

from algorithms.descriptions import (
    MeshGeometrySpec,
    RobotSystemDescription,
    WorldDescription,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def robot_path() -> Path:
    return REPO_ROOT / "configs" / "robots" / "franka_fr3_with_franka_hand.yaml"


def robot_only_path() -> Path:
    return REPO_ROOT / "configs" / "robots" / "franka_fr3_robot_only.yaml"


def world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "franka_table_world.yaml"


def robot_only_world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "franka_robot_only_world.yaml"


def two_robot_world_path() -> Path:
    return REPO_ROOT / "configs" / "worlds" / "two_franka_table_world.yaml"


# ---------------------------------------------------------------------------
# Robot system
# ---------------------------------------------------------------------------


def test_robot_system_with_gripper_loads_cleanly():
    system = RobotSystemDescription.from_yaml(robot_path())

    assert system.robot.id == "franka_fr3"
    assert system.gripper is not None
    assert system.gripper.id == "franka_hand"
    assert system.resolve_path(system.robot.urdf_path).exists()


def test_robot_system_chains_are_top_level_and_named():
    system = RobotSystemDescription.from_yaml(robot_path())

    chain_ids = [chain.id for chain in system.kinematic_chains]
    assert chain_ids == ["arm", "arm_with_gripper"]

    arm = system.chain("arm")
    assert arm.tip_frame == "fr3_link8"
    assert len(arm.joints) == 7
    assert "fr3_finger_joint1" not in arm.joints

    arm_with_gripper = system.chain("arm_with_gripper")
    # Mimic follower fr3_finger_joint2 is intentionally hidden from active chains.
    assert "fr3_finger_joint2" not in arm_with_gripper.joints
    assert "fr3_finger_joint1" in arm_with_gripper.joints
    assert len(arm_with_gripper.joints) == 8


def test_robot_system_tcps_are_top_level():
    system = RobotSystemDescription.from_yaml(robot_path())

    tcp_ids = [tcp.id for tcp in system.tcps]
    assert tcp_ids == ["robot_tcp", "hand_tcp"]
    assert system.tcp("hand_tcp").transform.child_frame == "fr3_hand_tcp"


def test_named_joint_state_is_a_dict():
    system = RobotSystemDescription.from_yaml(robot_path())

    home = system.named_joint_state("home")
    assert isinstance(home, dict)
    assert home["fr3_finger_joint1"] == 0.01
    # Mimic joint is intentionally absent.
    assert "fr3_finger_joint2" not in home


def test_robot_only_description_has_no_gripper():
    system = RobotSystemDescription.from_yaml(robot_only_path())

    assert system.gripper is None
    assert system.chain("arm").tip_frame == "fr3_link8"
    assert "fr3_finger_joint1" not in system.named_joint_state("home")


def test_robot_system_rejects_legacy_chains_under_robot():
    data = {
        "schema": "dexsent.algorithms.robot_system",
        "version": 2,
        "id": "x",
        "name": "x",
        "robot": {
            "id": "r",
            "urdf_path": "r.urdf",
            "base_frame": "base",
            "chains": [{"id": "arm", "base_frame": "base", "tip_frame": "tip", "joints": []}],
        },
    }
    with pytest.raises(ValidationError):
        RobotSystemDescription.model_validate(data)


def test_robot_system_rejects_legacy_parallel_list_joint_state():
    data = {
        "schema": "dexsent.algorithms.robot_system",
        "version": 2,
        "id": "x",
        "name": "x",
        "robot": {"id": "r", "urdf_path": "r.urdf", "base_frame": "base"},
        "named_joint_states": {
            "home": {"names": ["j1"], "positions": [0.0]},
        },
    }
    with pytest.raises(ValidationError):
        RobotSystemDescription.model_validate(data)


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


def test_world_loads_robot_system_and_object_geometry():
    world = WorldDescription.from_yaml(world_path())

    assert len(world.robots) == 1
    robot = world.robots[0]
    assert robot.id == "arm"
    assert robot.namespace is None
    assert robot.robot_system.id == "franka_fr3_with_franka_hand"

    assert len(world.objects) == 1
    obj = world.objects[0]
    assert obj.id == "matka"
    assert obj.type == "workpiece"
    assert obj.visual is not None
    assert isinstance(obj.visual.geometry, MeshGeometrySpec)
    assert world.resolve_path(obj.visual.geometry.path).exists()


def test_world_robot_only_loads_with_no_objects():
    world = WorldDescription.from_yaml(robot_only_world_path())

    assert world.objects == []
    assert world.robots[0].robot_system.gripper is None


def test_multi_robot_world_requires_namespaces():
    world = WorldDescription.from_yaml(two_robot_world_path())

    assert [r.id for r in world.robots] == ["left_arm", "right_arm"]
    assert world.robot("left_arm").frame_name("base") == "left/base"
    assert world.robot("right_arm").frame_name("fr3_link8") == "right/fr3_link8"


def test_multi_robot_world_rejects_null_namespace(tmp_path: Path):
    # Same as two_franka_table_world but with the right arm's namespace removed.
    bad_yaml = """
schema: dexsent.algorithms.world
version: 2
id: bad_world
world_frame: world
robots:
  - id: left_arm
    robot_system: """ + str(robot_path()) + """
    namespace: left
    base_pose:
      parent_frame: world
      child_frame: left/base
      matrix:
        - [1.0, 0.0, 0.0, 0.0]
        - [0.0, 1.0, 0.0, 0.0]
        - [0.0, 0.0, 1.0, 0.0]
        - [0.0, 0.0, 0.0, 1.0]
  - id: right_arm
    robot_system: """ + str(robot_path()) + """
    namespace: null
    base_pose:
      parent_frame: world
      child_frame: right/base
      matrix:
        - [1.0, 0.0, 0.0, 0.0]
        - [0.0, 1.0, 0.0, 0.0]
        - [0.0, 0.0, 1.0, 0.0]
        - [0.0, 0.0, 0.0, 1.0]
"""
    p = tmp_path / "bad.yaml"
    p.write_text(bad_yaml)
    with pytest.raises(ValueError, match="non-null namespace"):
        WorldDescription.from_yaml(p)


def test_multi_robot_world_rejects_duplicate_namespace(tmp_path: Path):
    bad_yaml = """
schema: dexsent.algorithms.world
version: 2
id: bad_world
world_frame: world
robots:
  - id: a
    robot_system: """ + str(robot_path()) + """
    namespace: same
  - id: b
    robot_system: """ + str(robot_path()) + """
    namespace: same
"""
    p = tmp_path / "bad.yaml"
    p.write_text(bad_yaml)
    with pytest.raises(ValueError, match="duplicate world robot namespaces"):
        WorldDescription.from_yaml(p)


def test_namespace_format_validator_rejects_uppercase():
    with pytest.raises(ValidationError):
        from algorithms.descriptions import WorldRobotDescription

        WorldRobotDescription(
            id="x",
            robot_system="r.yaml",
            namespace="Left",
        )


def test_visual_and_collision_origin_default_to_none():
    """When YAML omits `origin`, the geometry is colocated at the object's local frame."""
    world = WorldDescription.from_yaml(world_path())
    obj = world.objects[0]
    assert obj.visual is not None
    assert obj.visual.origin is None
    assert obj.collision is not None
    assert obj.collision.origin is None


def test_visual_and_collision_origin_loads_when_provided():
    from algorithms.descriptions import (
        CollisionGeometrySpec,
        MeshGeometrySpec,
        VisualSpec,
    )

    visual = VisualSpec.model_validate({
        "enabled": True,
        "geometry": {"type": "mesh", "path": "v.obj", "scale": [1.0, 1.0, 1.0]},
        "origin": {
            "parent_frame": "matka",
            "child_frame": "matka_visual",
            "matrix": [[1, 0, 0, 0.01], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        },
    })
    assert visual.origin is not None
    assert visual.origin.as_matrix()[0, 3] == 0.01

    collision = CollisionGeometrySpec.model_validate({
        "enabled": True,
        "geometry": {"type": "mesh", "path": "c.stl", "scale": [1.0, 1.0, 1.0]},
        "origin": {
            "parent_frame": "matka",
            "child_frame": "matka_collision",
            "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.02], [0, 0, 0, 1]],
        },
        "processing": {"type": "convex_decomposition", "max_hulls": 8},
    })
    assert collision.origin is not None
    assert collision.origin.as_matrix()[2, 3] == -0.02
    assert collision.processing.max_hulls == 8


def test_object_type_rejects_attached_at_yaml_load():
    bad = {
        "id": "x",
        "type": "attached",
        "pose": {
            "parent_frame": "world",
            "child_frame": "x",
            "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        },
    }
    from algorithms.descriptions import WorldObjectDescription

    with pytest.raises(ValidationError):
        WorldObjectDescription.model_validate(bad)
