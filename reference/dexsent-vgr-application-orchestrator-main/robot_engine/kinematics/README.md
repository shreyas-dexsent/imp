# kinematics

Forward kinematics, Jacobian computation, inverse kinematics (multiple backends), singularity analysis, and redundancy resolution.

---

## `kinematic_chain.py`

### `KinematicChain`

The core FK engine. Operates directly on `KinematicChainConfig` — no URDF required.

#### `forward_matrices(q: Dict[str, float]) → ChainState`

Iterates joints in definition order:
```
T_child = T_parent @ T_origin @ T_motion(joint_type, axis, q_i)
```

`T_motion` (from `math_utils.joint_motion_matrix`):
- **revolute**: `exp([axis]× · q_i)` — rotation by angle `q_i` about `axis`
- **prismatic**: `T` with `t = axis · q_i` — translation along `axis`
- **fixed**: identity

TCP offset applied last if `config.tcp` is set:
```
T_tcp = T_tip @ T_flange_tcp
```

Returns `ChainState` with all frame matrices and the list of movable joint names.

#### `forward_transforms(q) → Dict[str, Transform3D]`

Wraps `forward_matrices` output in `Transform3D` objects (from base frame).

#### `clamp(q) → Dict[str, float]`

Element-wise `max(lower, min(upper, q_i))`.

#### `violates_limits(q) → List[str]`

Returns names of joints outside `[lower - 1e-12, upper + 1e-12]`.

#### `tcp_frame → str`

`config.tcp.transform.child_frame` if TCP is configured, else `config.tip_frame`.

---

## `fk_solver.py`

### `compute_fk(request: FKRequest) → FKResult`

1. Checks joint limits via `chain.violates_limits`.
2. Calls `chain.forward_transforms(q)`.
3. Optionally filters to `request.target_frame` only.

---

## `jacobian_solver.py`

### `compute_jacobian(request: JacobianRequest, eps=1e-6) → JacobianResult`

Numerical Jacobian via forward finite differences in joint space:

```
J[:, i] = pose_error(FK(q), FK(q + ε·eᵢ)) / ε
```

`pose_error` returns a 6-vector `[translation_error; rotation_error]`, giving a geometric Jacobian:

```
J ∈ ℝ^{6×n}
J_linear  = ∂p/∂qᵢ   (3 rows)
J_angular = ∂ω/∂qᵢ   (3 rows, from rotation_log)
```

Returns `condition_number = σ_max / σ_min` from SVD of J.

---

## `ik_solver.py`

### `solve_ik(request: IKRequest) → IKResult`

Damped Least-Squares (DLS) IK, iterating up to `max_iterations`:

```
err = pose_error(T_current, T_target)  ∈ ℝ⁶

step = Jᵀ (J Jᵀ + λ²I)⁻¹ err        Levenberg-Marquardt
q ← q + step
q ← clamp(q, lower, upper)
```

Convergence criterion: `‖err[:3]‖ ≤ position_tolerance` AND `‖err[3:]‖ ≤ orientation_tolerance`.

Termination reasons: `IK_CONVERGED`, `MAX_ITERATIONS`, `SINGULAR_JACOBIAN`, `IK_UNREACHABLE`, `JOINT_LIMIT_VIOLATION`.

### `solve_ik_with_backend(request, backend="auto") → IKResult`

Dispatches to a named backend from the `IKBackendRegistry`. `"auto"` defaults to DLS.

### `solve_ik_multi_seed(request, seeds, backend) → IKResult`

Tries each seed independently, returns the solution with minimum joint displacement `‖q - q_seed‖`.

### `rank_solutions(solutions, q_current) → List`

Sorts solutions by `‖q - q_current‖` — minimum joint motion first.

---

## IK Backends (`ik_backends/`)

All backends implement `BaseIKBackend.solve(request) → IKResult`.

### `dls_ik.py` — `DLSIKBackend`

Damped Least-Squares as above. Default backend.

### `lm_ik.py` — `LMIKBackend`

Levenberg-Marquardt with adaptive damping: increases `λ` on divergence, decreases on convergence (full LM schedule vs fixed-λ DLS).

### `optimization_ik.py` — `OptimizationIKBackend`

Wraps `scipy.optimize.minimize`. Cost function:
```
f(q) = ‖pose_error(FK(q), T_target)‖² + α ‖q - q_seed‖²
```
Gradient via `finite_difference_jacobian`.

### `sqp_ik.py` — `SQPIKBackend`

Sequential Quadratic Programming via `scipy.optimize.minimize(method='SLSQP')` with joint limits as bounds and a pose-error equality constraint.

### `analytical_ik.py` — `AnalyticalIKBackend`

Closed-form solver for 6-DOF robots with a spherical wrist (Pieper criterion). Decouples position (first 3 joints) from orientation (last 3 joints). Returns up to 8 solutions corresponding to shoulder/elbow/wrist flip configurations.

### `eaik_adapter.py` — `EAIKAdapterBackend`

Adapter for the EAIK (Efficient Analytical IK) library. Uses subgroup decomposition to reduce the 6-DOF IK to a sequence of simpler 1D or 2D problems.

### `pinocchio_ik.py` — `PinocchioIKBackend`

Uses Pinocchio's `computeJointJacobians` and its own task-space controller for IK when a URDF model is available.

---

## `singularity.py`

### `manipulability_index(J) → float`

Yoshikawa's manipulability measure:
```
w = √(det(J Jᵀ)) = ∏ σᵢ
```
where `σᵢ` are singular values of J. Zero at a singularity.

### `condition_number(J) → float`

```
κ(J) = σ_max / σ_min
```
High condition number (→∞) indicates near-singularity.

### `minimum_singular_value(J) → float`

`σ_min` — smallest singular value. Near zero at singularity.

### `is_near_singularity(J, threshold) → bool`

`κ(J) ≥ threshold`

### `singularity_report(J) → dict`

Returns all three metrics in one call.

---

## `redundancy.py`

### `select_minimum_joint_motion_solution(solutions, q_current, weights) → np.ndarray`

Selects the IK solution minimising `‖W(q - q_current)‖` — minimum weighted joint displacement.

### `joint_limit_avoidance_gradient(q, lower, upper) → np.ndarray`

Gradient of the joint-limit avoidance potential:
```
∂H/∂qᵢ = -2(qᵢ - midᵢ) / spanᵢ²
```
where `mid = (lower + upper)/2`, `span = upper - lower`. Pushes joints toward the centre of their range.

### `preferred_posture_cost(q, q_nominal, weights) → float`

```
H = ½ Σ wᵢ (qᵢ - q_nominal_i)²
```

### `manipulability_gradient_numeric(q, jacobian_fn, eps=1e-5) → np.ndarray`

Numerical gradient of `w(q)` (manipulability index) via central differences.

### `nullspace_secondary_objective(primary_update, J, secondary_gradient) → np.ndarray`

Projects a secondary task into the null space of J:
```
δq = δq_primary + N · ∇H
N = I - J⁺ J
```

---

## `constraints.py`

Frozen dataclasses defining kinematic constraints:

| Class | Fields | Description |
|---|---|---|
| `JointLimitConstraint` | `lower, upper` | `satisfied(q) → bool` |
| `PositionToleranceConstraint` | `tolerance` | Cartesian position tolerance |
| `OrientationToleranceConstraint` | `tolerance` | Rotation tolerance in radians |
| `PoseToleranceConstraint` | `position_tolerance, orientation_tolerance` | Combined |
| `SingularityConstraint` | `condition_number_threshold` | |
| `CollisionFreeConstraint` | `checker` | Callable collision checker |
| `PreferredPostureConstraint` | `q_nominal, weight` | Secondary objective |
| `MinimumJointMotionConstraint` | `q_current, weight` | Seed-proximity secondary objective |

---

## `chain_model.py`

### `KinematicChainModel`

Higher-level model wrapping `KinematicChainConfig` with named frame accessors for base, flange, TCP, and gripper frames. Provides `get_chain_transform(q)` and `get_tcp_transform(q)` returning `Transform3D` objects. TCP can be updated at runtime via `update_tcp_transform`.

---

## `robot_model.py`

### `RobotModel`

Optional Pinocchio-backed model loaded from URDF via `pin.buildModelFromUrdf`. Used when full dynamics or advanced FK (e.g., parallel kinematics) are needed.

Key methods: `get_joint_names`, `get_joint_limits`, `get_velocity_limits`, `get_neutral_configuration`, `validate_configuration`, `validate_joint_limits`.

---

## `frame_model.py`

### `FrameModel` (frozen dataclass)

Associates a `frame_id` and `parent_frame_id` with a `Transform3D` and a semantic `frame_type` label (`"base"`, `"flange"`, `"tcp"`, `"gripper"`, `"object"`, `"grasp"`, `"bin"`, `"camera"`, `"fixture"`, `"custom"`).
