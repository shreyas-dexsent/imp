"""Path-processing imp module: shortcut smoothing then optional spline fit.

Compute-Runtime wrapper over motion-core. Subscribes a joint-space ``Path``
(typically from ``motion-ompl`` or ``motion-cartesian``), reconstructs the
internal ``algorithms.planning.path.Path``, runs ``shortcut_smooth`` against
the configured world+model, optionally fits a polynomial spline through the
shortened waypoints, and republishes the result on a ``processed`` topic.

Both ops live in motion-core (``algorithms.optimization``); this module only
marshals topic messages to and from the library.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from algorithms.descriptions import WorldDescription
from algorithms.optimization import shortcut_smooth, spline_fit
from algorithms.planning.path import Path as MotionPath
from algorithms.resolved import CollisionModel, KinematicModel, Scene


class PathProcessorModule(Module):
    name = "motion-path-processor"

    def __init__(
        self,
        station: str,
        robot: str,
        world_path: str,
        world_robot: str = "arm",
        plan_in: str = "plan",
        plan_out: str = "processed",
        *,
        shortcut_iters: int = 100,
        max_joint_step: float = 0.05,
        spline_order: Optional[str] = "quintic",
        spline_samples: int = 200,
        random_seed: int = 0,
    ):
        if spline_order not in (None, "cubic", "quintic"):
            raise ValueError(f"spline_order must be None, 'cubic', or 'quintic'; got {spline_order!r}")
        self.station = station
        self.robot = robot
        self.world_path = world_path
        self.world_robot = world_robot
        self.shortcut_iters = shortcut_iters
        self.max_joint_step = max_joint_step
        self.spline_order = spline_order
        self.spline_samples = spline_samples
        self.random_seed = random_seed
        self.in_key = keyexpr.motion(station, plan_in, "path")
        self.out_key = keyexpr.motion(station, plan_out, "path")
        self.model: Optional[KinematicModel] = None

    def inputs(self):
        return [Input("path", self.in_key, imp_pb2.Path)]

    def outputs(self):
        return [Output("path", self.out_key, imp_pb2.Path, QosClass.STATE)]

    def configure(self) -> None:
        world = WorldDescription.from_yaml(self.world_path)
        self.scene = Scene.from_world(world, CollisionModel.from_world(world))
        self.model = KinematicModel.from_robot_system(world.robot(self.world_robot).robot_system)
        self.joint_names = tuple(self.model.active_joint_names)

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        msg: imp_pb2.Path = latest["path"]  # type: ignore[assignment]
        n_dof = int(msg.n_dof)
        if n_dof == 0 or not msg.q_wp:
            return {}  # planner failed upstream; nothing to process
        flat = np.asarray(msg.q_wp, dtype=float)
        if flat.size % n_dof != 0:
            return {}
        wp = flat.reshape(-1, n_dof)
        if wp.shape[0] < 2 or wp.shape[1] != len(self.joint_names):
            return {}

        path = MotionPath(waypoints=wp, joint_names=self.joint_names)

        shortened, _stats = shortcut_smooth(
            path,
            self.model,
            self.scene,
            iterations=self.shortcut_iters,
            max_joint_step=self.max_joint_step,
            random_seed=self.random_seed,
        )

        final = (
            spline_fit(shortened, order=self.spline_order, samples=self.spline_samples)
            if self.spline_order is not None and shortened.num_waypoints >= 2
            else shortened
        )

        out_wp = np.asarray(final.waypoints, dtype=float)
        return {
            "path": imp_pb2.Path(
                header=imp_pb2.Header(schema="imp.Path/1"),
                q_wp=out_wp.flatten().tolist(),
                n_dof=out_wp.shape[1],
            )
        }
