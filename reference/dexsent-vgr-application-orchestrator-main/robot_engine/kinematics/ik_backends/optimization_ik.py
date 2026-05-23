from __future__ import annotations

from robot_engine.kinematics.ik_backends.lm_ik import LMIKBackend


class OptimizationIKBackend(LMIKBackend):
    backend_name = "OPTIMIZATION"

