# core_math

Foundational mathematical primitives used across the engine: rotations, SE(3) transforms, Lie groups, numerical differentiation, optimisation utilities, and tolerances.

Quaternion convention throughout: **`[x, y, z, w]`** (scipy / ROS standard).
Twist convention: **`[vx, vy, vz, wx, wy, wz]`** (linear then angular).

---

## `rotations.py`

### `normalize_quaternion(q) → np.ndarray`

Divides by L2 norm. Raises `ValueError` if `‖q‖ ≤ 1e-9`.

### `rotation_matrix_from_quaternion(q) → np.ndarray (3×3)`

`scipy.spatial.transform.Rotation.from_quat(q).as_matrix()`

### `quaternion_from_rotation_matrix(R) → np.ndarray (4,)`

`Rotation.from_matrix(R).as_quat()` after validating R.

### `validate_rotation_matrix(R)`

Checks: shape (3,3), finite, `R^T R ≈ I` (atol `1e-7`), `det(R) ≈ 1` (atol `1e-7`).

### `slerp_quaternion(q0, q1, alpha) → np.ndarray`

Spherical linear interpolation using `scipy.spatial.transform.Slerp`:

```
q(α) = q0 * (q0⁻¹ q1)^α
```

Clamps `α` to `[0, 1]`.

### `rotation_exp(w) → np.ndarray (3×3)`

Rodrigues exponential map: `Rotation.from_rotvec(w).as_matrix()`

```
exp([w]×) = I + sin(θ)/θ [w]× + (1 - cos(θ))/θ² [w]×²,  θ = ‖w‖
```

### `rotation_log(R) → np.ndarray (3,)`

Inverse Rodrigues: `Rotation.from_matrix(R).as_rotvec()`

```
w = θ * axis,  θ = arccos((tr(R) - 1) / 2)
```

### `orientation_error(current, target) → np.ndarray (3,)`

`rotation_log(target @ current^T)` — axis-angle residual.

### `angular_distance(current, target) → float`

`‖orientation_error(current, target)‖` — geodesic distance on SO(3) in radians.

### `euler_to_rotation_matrix(euler, convention="xyz") → np.ndarray`
### `rotation_matrix_to_euler(R, convention="xyz") → np.ndarray`

Thin wrappers on `scipy.Rotation.from_euler` / `.as_euler`.

---

## `transforms.py`

### `compose_transform(T_ab, T_bc) → np.ndarray (4×4)`

Matrix product `T_ab @ T_bc`. Both inputs validated.

### `invert_transform(T) → np.ndarray (4×4)`

Efficient rigid-body inverse exploiting block structure:

```
T⁻¹ = [ R^T  | -R^T t ]
       [  0   |    1   ]
```

### `adjoint_SE3(T) → np.ndarray (6×6)`

Adjoint of the rigid-body transformation for transforming twists:

```
Ad(T) = [ R   [p]× R ]
        [ 0      R  ]
```

where `[p]×` is the skew-symmetric matrix of the translation vector.

### `transform_twist(T, twist) → np.ndarray (6,)`

`Ad(T) @ twist` — maps a body twist to a spatial twist in frame T.

### `transform_wrench(T, wrench) → np.ndarray (6,)`

`Ad(T)^{-T} @ wrench` — dual (co-vector) transformation for forces/torques.

### `pose_to_matrix(position, quaternion) → np.ndarray (4×4)`

Builds a 4×4 homogeneous matrix from a 3-vector position and `[x,y,z,w]` quaternion.

### `matrix_to_pose(T) → (np.ndarray, np.ndarray)`

Returns `(translation, quaternion)` pair.

### `pose_error(T_current, T_target) → np.ndarray (6,)`

```
err = [ T_target[:3,3] - T_current[:3,3],
        rotation_log(T_target[:3,:3] @ T_current[:3,:3]^T) ]
```

6-DOF task-space residual used by IK solvers.

### `translation_error / rotation_error`

Components of `pose_error` returned individually.

---

## `lie_groups.py`

Implements SE(3) exponential/logarithm maps and Jacobians. The twist convention is `ξ = [v; ω] ∈ ℝ⁶`.

### `skew(v) → np.ndarray (3×3)` / `unskew(S) → np.ndarray (3,)`

```
[v]× = [  0  -v₃  v₂ ]
       [ v₃   0  -v₁ ]
       [-v₂  v₁   0  ]
```

### `left_jacobian_SO3(w) → np.ndarray (3×3)`

```
J_l(w) = I + (1 - cos θ)/θ² [w]× + (θ - sin θ)/θ³ [w]×²
```

Near-zero Taylor expansion: `I + ½[w]× + ⅙[w]×²`

### `inverse_left_jacobian_SO3(w) → np.ndarray (3×3)`

```
J_l⁻¹(w) = I - ½[w]× + (1/θ² - (1+cos θ)/(2θ sin θ)) [w]×²
```

Near-zero Taylor: `I - ½[w]× + ¹⁄₁₂[w]×²`

### `se3_exp(ξ) → np.ndarray (4×4)`

Exponential map from twist to SE(3):

```
T = exp([ξ]∧) = [ exp([ω]×)   J_l(ω) v ]
                [    0             1    ]
```

where `ξ = [v; ω]`, and the translational component is `J_l(ω) v` (not simply `v`).

### `se3_log(T) → np.ndarray (6,)`

Logarithm map — inverse of `se3_exp`:

```
ω = log_SO3(R)
v = J_l(ω)⁻¹ t
ξ = [v; ω]
```

### `body_twist_error(T_current, T_target) → np.ndarray (6,)`

`se3_log(T_current⁻¹ T_target)` — error expressed in the body frame.

### `spatial_twist_error(T_current, T_target) → np.ndarray (6,)`

`se3_log(T_target T_current⁻¹)` — error in the spatial (world) frame.

---

## `interpolation.py`

### `interpolate_joint(q0, q1, alpha) → np.ndarray`

Linear interpolation: `q0 + (q1 - q0) * α`

### `interpolate_pose_SE3(T0, T1, alpha) → np.ndarray (4×4)`

Geodesic interpolation on SE(3):

```
T(α) = T0 · exp(α · log(T0⁻¹ T1))
```

This follows the SE(3) geodesic — not a product of independent R and t interpolations.

### `interpolate_pose_position_slerp(T0, T1, alpha) → np.ndarray (4×4)`

Decoupled interpolation: linear position + SLERP orientation. Cheaper than SE(3) geodesic but not geometrically consistent.

### `sample_joint_path(q0, q1, max_joint_step) → List[np.ndarray]`

Computes `n = ceil(‖q1-q0‖_∞ / max_joint_step) + 1` uniformly spaced waypoints.

### `sample_cartesian_path(T0, T1, translation_step, rotation_step) → List[np.ndarray]`

```
n = ceil(max(‖p1-p0‖ / t_step, ‖ω‖ / r_step)) + 1
```

where `ω = se3_log(T0⁻¹ T1)[3:]` is the rotation magnitude.

### `compute_path_length_joint / compute_path_length_cartesian`

Sum of segment L2 norms in joint space / Cartesian translation space respectively.

---

## `metrics.py`

### `euclidean_distance(a, b) → float`
`‖a - b‖₂`

### `angular_distance(Ra, Rb) → float`
Delegates to `rotation_log`-based geodesic distance.

### `pose_distance(Ta, Tb, position_weight, orientation_weight) → float`
`w_pos · ‖p_a - p_b‖ + w_rot · angular_distance(Ra, Rb)`

### `joint_distance(qa, qb, weights) → float`
`‖W(qa - qb)‖₂`

### `max_joint_delta(q0, q1) → float`
`‖q1 - q0‖_∞` — used for joint step sizing.

### `rms_error(errors) → float`
`sqrt(mean(errors²))`

---

## `numerical_derivatives.py`

### `finite_difference_jacobian(f, x, eps=1e-6) → np.ndarray`

Forward differences: `J[:,i] = (f(x + ε·eᵢ) - f(x)) / ε`

### `central_difference_jacobian(f, x, eps=1e-6) → np.ndarray`

Central differences: `J[:,i] = (f(x + ε·eᵢ) - f(x - ε·eᵢ)) / (2ε)`

O(ε²) accuracy vs O(ε) for forward differences.

### `check_jacobian_analytic_vs_numeric(J_analytic, J_numeric, tol=1e-6) → dict`

Returns `{"ok", "error_norm", "max_abs_error"}`.

### `finite_difference_gradient(f, x, eps) → np.ndarray`
### `finite_difference_hessian(f, x, eps=1e-5) → np.ndarray`

Hessian via two applications of finite difference gradient.

---

## `optimization.py`

Pseudoinverse and nullspace tools used by IK and redundancy resolution.

### `pseudoinverse(J) → np.ndarray`
`J⁺ = pinv(J)` — Moore-Penrose pseudoinverse via SVD.

### `damped_pseudoinverse(J, λ=1e-3) → np.ndarray`

Levenberg-Marquardt damping for near-singular Jacobians:

```
J⁺_λ = Jᵀ (JJᵀ + λ²I)⁻¹
```

### `weighted_damped_pseudoinverse(J, W, λ=1e-3) → np.ndarray`

```
J⁺_W = W⁻¹ Jᵀ (J W⁻¹ Jᵀ + λ²I)⁻¹
```

W is the joint-space metric (e.g. inertia matrix). Minimises `‖δq‖_W`.

### `nullspace_projector(J) → np.ndarray (n×n)`

```
N = I - J⁺ J
```

Projects any vector into the null space of J; used for secondary objectives in redundancy resolution.

### `weighted_nullspace_projector(J, W) → np.ndarray`

`N_W = I - J⁺_W J` with the weighted pseudoinverse.

### `clamp_to_joint_limits(q, lower, upper) → np.ndarray`

Element-wise clip.

### `joint_limit_margin(q, lower, upper) → np.ndarray`

`min(q - lower, upper - q)` — distance to nearest limit per joint.

### `residual_norm(r, weights) → float`

`‖diag(w) r‖₂`

---

## `tolerances.py`

### `ToleranceConfig` (Pydantic BaseModel)

Central tolerance registry used across the engine:

| Field | Default | Usage |
|---|---|---|
| `numerical` | `1e-9` | Near-zero guards |
| `transform` | `1e-8` | Transform validity |
| `rotation_orthonormal` | `1e-7` | `R^T R ≈ I` check |
| `rotation_determinant` | `1e-7` | `det(R) ≈ 1` check |
| `ik_position` | `1e-4` | IK position convergence (m) |
| `ik_orientation` | `1e-3` | IK orientation convergence (rad) |
| `collision_clearance` | `1e-4` | Minimum safe clearance (m) |
| `trajectory` | `1e-8` | Trajectory continuity |

`DEFAULT_TOLERANCES = ToleranceConfig()` is the singleton imported by other modules.
