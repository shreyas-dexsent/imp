"""spatial-transform: lift perception poses from camera frame to robot base.

Subscribes a perception ``Pose6D`` plus ``imp/<station>/tf`` (and optionally
the robot's ``RobotState``), composes the chain through a ``TfGraph``, and
publishes a ``PoseTarget`` in the configured base frame (spec §9).

Eye-to-hand mode (default): a static hand-eye edge ``base -> camera`` arrives
on tf once at startup; every pose is a single lookup.

Eye-in-hand mode (``eye_in_hand=True`` + ``robot_system_path`` set): FK is
computed from each ``RobotState`` and injected as a transient ``base -> tcp``
edge in the in-process graph; hand-eye is then ``tcp -> camera``. Same lookup,
two-hop composition.

``TransformModule`` is lazily imported so the pure ``lift_pose`` helpers
remain usable without ``imp_sdk`` / ``zenoh`` installed (e.g. inside unit
tests that only exercise the SE(3) math).
"""

from .lift import lift_pose, matrix_to_pose, pose_to_matrix

__all__ = ["TransformModule", "lift_pose", "matrix_to_pose", "pose_to_matrix"]


def __getattr__(name):
    if name == "TransformModule":
        from .transform import TransformModule  # local import: needs imp_sdk + zenoh

        return TransformModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
