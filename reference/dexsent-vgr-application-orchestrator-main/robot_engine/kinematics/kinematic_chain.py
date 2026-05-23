from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from robot_engine.interfaces.schemas import KinematicChainConfig, Transform3D
from robot_engine.math_utils import as_matrix, joint_motion_matrix, to_transform


@dataclass
class ChainState:
    transforms: Dict[str, np.ndarray]
    joint_names: List[str]


class KinematicChain:
    def __init__(self, config: KinematicChainConfig):
        self.config = config
        self.movable_joints = [joint for joint in config.joints if joint.joint_type != "fixed"]
        self.joint_names = [joint.name for joint in self.movable_joints]

    def clamp(self, q: Dict[str, float]) -> Dict[str, float]:
        out = dict(q)
        for joint in self.movable_joints:
            value = float(out.get(joint.name, 0.0))
            out[joint.name] = min(joint.upper, max(joint.lower, value))
        return out

    def violates_limits(self, q: Dict[str, float]) -> List[str]:
        bad = []
        for joint in self.movable_joints:
            value = float(q.get(joint.name, 0.0))
            if value < joint.lower - 1e-12 or value > joint.upper + 1e-12:
                bad.append(joint.name)
        return bad

    def forward_matrices(self, q: Dict[str, float]) -> ChainState:
        transforms = {self.config.base_frame: np.eye(4)}
        for joint in self.config.joints:
            parent = transforms[joint.parent_frame]
            origin = as_matrix(joint.origin)
            motion = joint_motion_matrix(joint.joint_type, joint.axis, float(q.get(joint.name, 0.0)))
            transforms[joint.child_frame] = parent @ origin @ motion
        tip = transforms[self.config.tip_frame]
        if self.config.tcp is not None:
            transforms[self.config.tcp.transform.child_frame] = tip @ as_matrix(self.config.tcp.transform)
        return ChainState(transforms=transforms, joint_names=self.joint_names)

    def forward_transforms(self, q: Dict[str, float]) -> Dict[str, Transform3D]:
        state = self.forward_matrices(q)
        return {
            frame: to_transform(self.config.base_frame, frame, matrix)
            for frame, matrix in state.transforms.items()
            if frame != self.config.base_frame
        }

    @property
    def tcp_frame(self) -> str:
        return self.config.tcp.transform.child_frame if self.config.tcp else self.config.tip_frame
