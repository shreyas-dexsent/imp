# motion-core

**Kind:** Module (shared library) &nbsp;|&nbsp; **Status:** phase 3 — ported + verified

The `robot-algorithms` three-layer motion library, vendored intact as the canonical
motion source (spec §9/§22). Import package: **`algorithms`**; distribution:
`imp-motion-core`. The thin `motion-*` plugins wrap its operations.

- **descriptions** — pydantic models from YAML (`RobotSystemDescription`, `WorldDescription`).
- **resolved** — `KinematicModel`, `CollisionModel`, mutable `Scene` (the live-state seam).
- **operations** — stateless ops: FK, IK (DLS/QP/analytical), collision (Coal), planning
  (OMPL/cartesian), optimization (shortcut/spline), trajectory (Ruckig/poly).

```bash
pip install pin coal ompl open3d numpy scipy trimesh pydantic pyyaml pytest
cd algorithms && PYTHONPATH=. python -m pytest tests -q     # 226 passed
```

Layout mirrors upstream so the locked test suite passes verbatim: `algorithms/`
(package + `configs/` + `tests/` + `docs/`) with a sibling `assets/` (robot/gripper/
object meshes + URDFs). `meshcat` (viz only) is optional. Vendored from
`reference/robot-algorithms`.
