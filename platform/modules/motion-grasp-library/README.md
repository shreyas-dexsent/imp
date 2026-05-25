# motion-grasp-library

**Kind:** Module &nbsp;|&nbsp; **Status:** phase 3 — `synthesize_grasps` implemented (feasibility = P5)

Grasp candidate library + the `synthesize_grasps` op (spec §9). Ported from
`reference/.../orchestrator/robot_engine/planning/{grasp_library,grasp_candidate,grasp_feasibility}.py`
— grasping is deliberately **not** in `robot-algorithms`, so this module is
imp's own.

## What's in the box

| Symbol | Where | Role |
|---|---|---|
| `Grasp` | `library.py` | Dataclass: `grasp_id`, `score`, `t_obj_gripper` (4x4 in the object frame). |
| `GraspLibrary` | `library.py` | Keyed store of `Grasp` records; `from_json(...)` + score-sorted `list()`. |
| `synthesize_grasps(library, T_world_object)` | `synthesize.py` | Pure: compose each candidate's `t_obj_gripper` with the live object pose; returns `WorldGrasp` (with `t_world_gripper`). |
| `SynthesizeGraspsModule` | `module.py` | Compute-Runtime wrapper: subscribes the object's world-frame `Pose6D`, publishes a `Grasps` message with every candidate lifted into world coordinates (sorted by score). |

## Feasibility (`grasp_feasibility`)

Per-candidate IK + collision feasibility lives best as a **service** that
calls into `motion-pinocchio` IK and `motion-coal` collision for each
candidate — it's not a per-tick compute. The schema + the service shell
land with the task layer in **P5**; for now `synthesize_grasps` produces
the ranked candidates and consumers can run feasibility themselves.

## `grasps.json` format

```json
{
  "object_id": "matka",
  "grasps": [
    {"grasp_id": "top_pinch_0", "score": 0.93,
     "t_obj_gripper": [[1,0,0,0],[0,1,0,0],[0,0,1,0.05],[0,0,0,1]]},
    {"grasp_id": "side_grab_left", "score": 0.71,
     "t_obj_gripper": [1, 0, 0, 0, 0, 1, 0, -0.04, 0, 0, 1, 0.05, 0, 0, 0, 1]}
  ]
}
```

`t_obj_gripper` accepts nested 4x4 or flat row-major 16-float.

## Run

```bash
PYTHONPATH=sdk/py:modules/motion-grasp-library \
  python -m imp_module_motion_grasp_library \
    --object-pose-key imp/devstation/perc/s1/world_pose \
    --grasps workspace/stations/st1/processes/p1/objects/matka/grasps.json
```

## Tests

```bash
cd platform/modules/motion-grasp-library
PYTHONPATH=. pytest tests -q       # 8 passed
```

Covers Grasp validation, library upsert + sort, JSON round-trip, and
`synthesize_grasps` composition.
