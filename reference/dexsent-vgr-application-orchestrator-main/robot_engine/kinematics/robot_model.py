from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_engine.interfaces.schemas import AlgorithmError, RobotModelConfig


@dataclass
class RobotModel:
    config: RobotModelConfig
    pin_model: object = None
    pin_data: object = None
    error: AlgorithmError | None = None

    @classmethod
    def load_from_urdf(cls, urdf_path: str, package_dirs=None) -> "RobotModel":
        return load_robot_model(RobotModelConfig(robot_id=Path(urdf_path).stem, urdf_path=urdf_path, package_dirs=package_dirs or []))

    @classmethod
    def load_with_pinocchio(cls, urdf_path: str, package_dirs=None) -> "RobotModel":
        return cls.load_from_urdf(urdf_path, package_dirs)

    def get_joint_names(self) -> list[str]:
        if self.pin_model is None:
            return []
        return [name for name in self.pin_model.names[1:]]

    def get_joint_limits(self):
        if self.pin_model is None:
            return [], []
        return self.pin_model.lowerPositionLimit.tolist(), self.pin_model.upperPositionLimit.tolist()

    def get_velocity_limits(self):
        if self.pin_model is None:
            return []
        return self.pin_model.velocityLimit.tolist()

    def get_acceleration_limits(self):
        if self.pin_model is None:
            return []
        limit = getattr(self.pin_model, "effortLimit", np.ones(self.pin_model.nv))
        return np.asarray(limit, dtype=float).tolist()

    def get_frame_names(self) -> list[str]:
        if self.pin_model is None:
            return []
        return [frame.name for frame in self.pin_model.frames]

    def get_link_names(self) -> list[str]:
        if self.pin_model is None:
            return []
        return list(dict.fromkeys(frame.name for frame in self.pin_model.frames if "joint" not in frame.name.lower()))

    def get_neutral_configuration(self):
        if self.pin_model is None:
            return []
        try:
            import pinocchio as pin

            return pin.neutral(self.pin_model).tolist()
        except Exception:
            return np.zeros(self.pin_model.nq).tolist()

    def validate_configuration(self, q) -> AlgorithmError | None:
        if self.pin_model is None:
            return None
        q = np.asarray(q, dtype=float)
        if q.shape != (self.pin_model.nq,) or not np.isfinite(q).all():
            return AlgorithmError(code="INVALID_TRANSFORM", message="Configuration has invalid shape or non-finite values.")
        return self.validate_joint_limits(q)

    def validate_joint_limits(self, q) -> AlgorithmError | None:
        if self.pin_model is None:
            return None
        q = np.asarray(q, dtype=float)
        lower = np.asarray(self.pin_model.lowerPositionLimit, dtype=float)
        upper = np.asarray(self.pin_model.upperPositionLimit, dtype=float)
        if np.any(q < lower) or np.any(q > upper):
            return AlgorithmError(code="JOINT_LIMIT_VIOLATION", message="Configuration violates robot joint limits.")
        return None


def load_robot_model(config: RobotModelConfig) -> RobotModel:
    if not config.urdf_path:
        return RobotModel(config=config)
    try:
        import pinocchio as pin

        model = pin.buildModelFromUrdf(config.urdf_path)
        return RobotModel(config=config, pin_model=model, pin_data=model.createData())
    except Exception as exc:
        return RobotModel(config=config, error=AlgorithmError(code="MODEL_LOAD_FAILED", message=str(exc)))
