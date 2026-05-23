# planning

Grasp candidate representation, feasibility evaluation, and library management.

---

## `grasp_candidate.py`

Re-exports `GraspCandidate` from `interfaces.schemas`:

```python
@dataclass
class GraspCandidate:
    grasp_id: str
    score: float
    grasp_transform: Transform3D   # T_object_gripper
```

`grasp_transform` encodes the 6-DOF gripper pose relative to the object frame at the moment of grasping.

---

## `grasp_feasibility.py`

### `evaluate_grasp_candidate(request: GraspFeasibilityRequest, world: CollisionWorld | None) → GraspFeasibilityResult`

Two independent checks run in sequence:

**1. IK feasibility**

If `request.ik_request` is provided:
```python
ik_result = solve_ik(request.ik_request)
```
The grasp is IK-infeasible if `not ik_result.ok` (joint limits, singularity, non-convergence).

**2. Collision feasibility**

If `world` is provided:
```python
collision_result = check_scene(world)
```
The grasp is collision-infeasible if `collision_result.collision`.

**Result**:
```python
GraspFeasibilityResult(
    grasp_id=...,
    feasible=(not rejection_reasons),
    ik=ik_result,
    collision=collision_result,
    rejection_reasons=[AlgorithmError, ...]
)
```

A grasp is `feasible=True` only if both checks pass (no rejection reasons accumulated).

---

## `grasp_library.py`

### `GraspLibrary`

In-memory keyed store of `GraspCandidate` objects.

#### `__init__(candidates: Iterable[GraspCandidate])`

Builds `Dict[grasp_id → GraspCandidate]`.

#### `add(candidate: GraspCandidate)`

Upserts by `grasp_id`.

#### `list() → List[GraspCandidate]`

Returns all candidates sorted by `score` descending — highest-quality grasps first.
