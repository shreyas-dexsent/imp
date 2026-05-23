from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from robot_engine.interfaces.schemas import KinematicChainConfig, Transform3D
from robot_engine.kinematics.kinematic_chain import KinematicChain
from robot_engine.math_utils import as_matrix, to_transform


@dataclass
class KinematicChainModel:
    config: KinematicChainConfig
    base_frame: str
    flange_frame: str
    gripper_frame: str | None = None
    tcp_frame: str | None = None
    custom_tcp_transform: Transform3D | None = None

    @classmethod
    def from_config(cls, config: KinematicChainConfig) -> "KinematicChainModel":
        tcp = config.tcp.transform if config.tcp else None
        return cls(config, config.base_frame, config.tip_frame, tcp_frame=tcp.child_frame if tcp else config.tip_frame, custom_tcp_transform=tcp)

    def get_chain_transform(self, q: Dict[str, float]) -> Transform3D:
        return to_transform(self.base_frame, self.flange_frame, KinematicChain(self.config).forward_matrices(q).transforms[self.flange_frame])

    def get_tcp_transform(self, q: Dict[str, float]) -> Transform3D:
        chain = KinematicChain(self.config)
        frame = chain.tcp_frame
        return to_transform(self.base_frame, frame, chain.forward_matrices(q).transforms[frame])

    def update_tcp_transform(self, T_flange_tcp: Transform3D) -> None:
        self.custom_tcp_transform = T_flange_tcp
        self.tcp_frame = T_flange_tcp.child_frame


KinematicChain = KinematicChain

