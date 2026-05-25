"""Collision-check imp module (spec §9).

Compute-Runtime wrapper over motion-core. Builds the resolved
``CollisionModel`` + ``Scene`` from a ``world.yaml`` at ``configure``, then on
each ``RobotState`` runs ``algorithms.collision.is_in_collision`` for that
configuration and publishes the contact count (0 = collision-free) as a
``Scalar``.

The **Scene-fill seam** (spec §9, debt D3 in PLAN.md) is wired here when the
optional ``object_pose_key`` is configured: a perception ``Pose6D`` published
on that key is routed into ``Scene.set_object_pose(object_id, ...)`` each tick
*before* the collision query, so a moving obstacle observed by perception
flips the verdict without any orchestrator code in between.

Attach / detach is intentionally not yet a topic-driven action -- the schema
for grasp events lands with the task layer in P5. Until then the Scene's
``attach`` / ``detach`` are callable in-process and tested in the P3
integration test.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.collision import is_in_collision
from algorithms.descriptions import WorldDescription
from algorithms.resolved import CollisionModel, KinematicModel, Scene


def _pose_to_matrix(position_m, quat_xyzw) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(position_m, dtype=float)
    return T


class CollisionModule(Module):
    name = "motion-coal-collision"

    def __init__(
        self,
        station: str,
        robot: str,
        world_path: str,
        world_robot: str = "arm",
        plan: str = "collision",
        *,
        object_pose_key: Optional[str] = None,
        object_id: Optional[str] = None,
    ):
        if object_pose_key and not object_id:
            raise ValueError(
                "object_pose_key requires object_id (which Scene object the "
                "incoming pose updates)"
            )
        self.station = station
        self.robot = robot
        self.world_path = world_path
        self.world_robot = world_robot
        self.object_pose_key = object_pose_key
        self.object_id = object_id
        self.state_key = keyexpr.hal(station, robot, "state")
        self.out_key = keyexpr.motion(station, plan, "collision")
        self.model: Optional[KinematicModel] = None

    def inputs(self):
        specs = [Input("state", self.state_key, imp_pb2.RobotState)]
        if self.object_pose_key:
            specs.append(Input("object_pose", self.object_pose_key, imp_pb2.Pose6D))
        return specs

    def outputs(self):
        return [Output("collision", self.out_key, imp_pb2.Scalar, QosClass.STATE)]

    def configure(self) -> None:
        world = WorldDescription.from_yaml(self.world_path)
        self.collision_model = CollisionModel.from_world(world)
        self.scene = Scene.from_world(world, self.collision_model)
        self.model = KinematicModel.from_robot_system(world.robot(self.world_robot).robot_system)
        self.n_active = len(self.model.active_joint_names)
        self.world_frame = world.world_frame

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        # Scene-fill: route perception pose updates into the Scene before query.
        if "object_pose" in latest:
            pose: imp_pb2.Pose6D = latest["object_pose"]  # type: ignore[assignment]
            if pose.valid and self.object_id in self.scene.known_object_ids():
                T_world_obj = _pose_to_matrix(pose.position_m, pose.quat_xyzw)
                self.scene.set_object_pose(self.object_id, T_world_obj)

        q = np.asarray(latest["state"].q, dtype=float)
        if q.shape[0] != self.n_active:
            return {}

        # Bookkeeping: keep robot_states consistent with what we're querying.
        self.scene.set_robot_state(self.world_robot, q)

        report = is_in_collision(self.model, self.scene, q)
        value = float(len(report.contacts)) if report.in_collision else 0.0
        return {
            "collision": imp_pb2.Scalar(
                header=imp_pb2.Header(schema="imp.Scalar/1"),
                value=value,
            )
        }
