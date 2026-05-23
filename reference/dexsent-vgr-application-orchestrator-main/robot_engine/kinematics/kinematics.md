# Kinematics

Kinematics is separated into model loading, FK, Jacobian computation, IK backends, singularity metrics, and redundancy handling.

- `robot_model.py`: Pinocchio URDF loading through the PyPI package `pin`.
- `kinematic_chain.py`: lightweight UI-supplied serial chain for tests and backend integration.
- `fk_solver.py`: selected/all-frame FK, including externally supplied TCP transform.
- `jacobian_solver.py`: numerical frame/TCP Jacobian validation path.
- `ik_solver.py`: backend registry and multi-seed ranking.
- `ik_backends/`: DLS, LM, optimization, Pinocchio adapter, analytical adapter, EAIK lazy adapter, and explicit SQP interface.

Implemented backends are DLS and SciPy least-squares LM/optimization. SQP and generic analytical IK return clear non-success results until real robot-specific solvers are provided.

