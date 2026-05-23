# Interfaces

`interfaces/` is the UI/backend boundary. Core algorithms do not import UI code.

- `schemas.py`: Pydantic data contracts for transforms, assets, collision, kinematics, planning, trajectory, and motion primitives.
- `error_codes.py`: structured error names used by API-facing results.
- `ui_api.py`: high-level functions that backend code can call.

Expected integration flow:

1. UI submits robot/gripper/object/bin assets and frame transforms.
2. Backend builds a `RobotEngineContext`.
3. Backend calls collision, FK/Jacobian/IK, path planning, trajectory, or motion primitive functions.
4. Results return success state, reason codes, debug fields, path/trajectory data, and rejection details.

