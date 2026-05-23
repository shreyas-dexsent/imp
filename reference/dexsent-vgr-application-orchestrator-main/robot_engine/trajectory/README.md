# trajectory

Polynomial trajectory generation (cubic, quintic), trapezoidal and S-curve motion profiles, time scaling, re-timing, and validation.

---

## `trajectory_base.py`

### `JointTrajectoryPoint` (dataclass)

| Field | Description |
|---|---|
| `time: float` | Time stamp (seconds) |
| `q: List[float]` | Joint positions |
| `q_dot: List[float]` | Joint velocities |
| `q_ddot: List[float]` | Joint accelerations |
| `q_jerk: Optional[List[float]]` | Joint jerks |

### `JointTrajectory` (dataclass)

| Property | Description |
|---|---|
| `duration` | `points[-1].time` |
| `dof` | `len(points[0].q)` |

#### `evaluate(t) → JointTrajectoryPoint`

Linear interpolation between bracketing points:
```
i = searchsorted(times, t) - 1
α = (t - times[i]) / (times[i+1] - times[i])
q(t) = (1-α) q_i + α q_{i+1}
```

#### `sample(dt) → List[JointTrajectoryPoint]`

Samples at uniform time steps `[0, dt, 2dt, ..., duration]`.

---

## `polynomial.py`

### `solve_cubic_coefficients(q0, q1, v0, v1, T) → np.ndarray (4×DOF)`

Solves the boundary-value problem for a cubic polynomial `q(t) = a0 + a1 t + a2 t² + a3 t³`:

Boundary conditions:
- `q(0) = q0`, `q(T) = q1`
- `q̇(0) = v0`, `q̇(T) = v1`

Closed-form solution:
```
a0 = q0
a1 = v0
a2 = 3(q1 - q0)/T² - (2v0 + v1)/T
a3 = -2(q1 - q0)/T³ + (v0 + v1)/T²
```

### `solve_quintic_coefficients(q0, q1, v0, v1, a0, a1, T) → np.ndarray (6×DOF)`

Solves `q(t) = Σ cₙ tⁿ` for n=0..5, with 6 boundary conditions:
- `q(0)=q0`, `q(T)=q1`
- `q̇(0)=v0`, `q̇(T)=v1`
- `q̈(0)=a0`, `q̈(T)=a1`

First 3 coefficients are directly:
```
c0 = q0,  c1 = v0,  c2 = a0/2
```

Remaining 3 solved from the linear system:
```
[T³   T⁴   T⁵ ] [c3]   [q1 - (c0 + c1T + c2T²)]
[3T²  4T³  5T⁴] [c4] = [v1 - (c1 + 2c2T)      ]
[6T   12T² 20T³][c5]   [a1 - 2c2               ]
```

### `evaluate_polynomial(coeffs, t) → np.ndarray`

`q(t) = Σ coeffs[i] · t^i` — standard Horner-form evaluation.

### `evaluate_polynomial_derivative(coeffs, t, order) → np.ndarray`

Symbolic differentiation of the coefficient array (multiply by power, shift), then evaluate. Applied `order` times.

---

## `cubic.py`

### `CubicTrajectory`

Subclass of `JointTrajectory` tagged with `generation_method="cubic"`.

### `cubic_joint_trajectory(q0, q1, v0=0, v1=0, duration=1.0, samples=101) → CubicTrajectory`

Single-segment cubic trajectory. Evaluates at `samples` uniform time points in `[0, T]`.

### `multi_joint_cubic_trajectory(q0, q1, v0, v1, duration, samples) → CubicTrajectory`

Multi-DOF version: operates on arrays.

---

## `quintic.py`

### `QuinticTrajectory`

Subclass of `JointTrajectory` tagged with `generation_method="quintic"`.

### `quintic_joint_trajectory(q0, q1, v0=0, v1=0, a0=0, a1=0, duration=1.0, samples=101) → QuinticTrajectory`

Single-segment quintic trajectory.

### `multi_joint_quintic_trajectory(...) → QuinticTrajectory`

Multi-DOF version.

### `quintic_segment_interpolation(q_waypoints, ..., duration_per_segment=1.0) → QuinticTrajectory`

Piecewise quintic interpolation through a list of waypoints. Each segment uses zero-velocity and zero-acceleration boundary conditions. Time offsets accumulate across segments.

---

## `trapezoidal.py`

### `TrapezoidalProfile` (dataclass)

Fields: `q0, q1, v_max, a_max, t_acc, t_cruise, t_dec, duration, triangular`.

#### `evaluate(t) → (q, v, a)`

Three-phase evaluation:
```
Phase 1 (t ≤ t_acc):       q = q0 + sign · ½ a_max t²,  v = sign·a_max·t,  a = sign·a_max
Phase 2 (t ≤ t_acc+t_cruise): q = q0 + sign·(d_acc + v_max·dt),  v = sign·v_max,  a = 0
Phase 3 (deceleration):      q = q0 + sign·(d_dec_start + v_max·dt - ½ a_max dt²),  a = -sign·a_max
```

### `trapezoidal_profile_1d(q0, q1, v_max, a_max) → TrapezoidalProfile`

Checks if the motion distance allows a cruise phase:

```
t_acc = v_max / a_max
d_acc = ½ a_max t_acc²

if 2·d_acc ≥ |q1 - q0|:  # distance too short to reach v_max
    → triangular_profile_1d (no cruise phase)
else:
    t_cruise = (|q1 - q0| - 2·d_acc) / v_max
    duration = 2·t_acc + t_cruise
```

### `triangular_profile_1d(q0, q1, v_max, a_max) → TrapezoidalProfile`

Triangular profile (acceleration + deceleration only):
```
t_acc = √(|q1 - q0| / a_max)
peak_v = a_max · t_acc
duration = 2 · t_acc
```

### `synchronized_multi_joint_trapezoidal(q0, q1, v_limits, a_limits) → List[TrapezoidalProfile]`

Plans per-joint trapezoidal profiles, then sets all durations to the slowest joint's `duration`. This synchronises all joints to arrive at `q1` simultaneously.

---

## `s_curve.py`

### `SCurveProfile` (dataclass)

Uses a **smoothstep** (5th-order Hermite polynomial) as the displacement shape function:
```
s(u) = 10u³ - 15u⁴ + 6u⁵,   u = t/T
```

Derivatives (all divided by appropriate powers of T):
```
ṡ(u) = 30u² - 60u³ + 30u⁴
s̈(u) = 60u - 180u² + 120u³
s⃛(u) = 60 - 360u + 360u²
```

#### `evaluate(t) → (q, v, a, jerk)`

```
q    = q0 + D·s(u)
v    = D·ṡ(u)/T
a    = D·s̈(u)/T²
jerk = D·s⃛(u)/T³
```

where `D = q1 - q0`.

### `s_curve_profile_1d(q0, q1, v_max, a_max, j_max) → SCurveProfile`

Computes the minimum duration T satisfying all three constraints simultaneously:
```
T = max(
    _SMOOTHSTEP_V_MAX · |D| / v_max,    # velocity constraint
    √(_SMOOTHSTEP_A_MAX · |D| / a_max), # acceleration constraint
    ∛(_SMOOTHSTEP_J_MAX · |D| / j_max)  # jerk constraint
)
```

Constants (`_SMOOTHSTEP_V_MAX = 1.875`, `_SMOOTHSTEP_A_MAX ≈ 5.77`, `_SMOOTHSTEP_J_MAX = 60.0`) are the peak ratios of the smoothstep derivatives.

Only supports rest-to-rest profiles (`v0=v1=a0=a1=0`).

### `synchronized_multi_joint_s_curve(q0, q1, v_limits, a_limits, j_limits, samples=101) → JointTrajectory`

Plans per-joint S-curve profiles, synchronises to the longest `duration`, then evaluates all joints at the same time samples to produce a `JointTrajectory`.

### `validate_s_curve_limits(profile, samples=1001) → dict`

Numerically validates peak velocity, acceleration, and jerk over 1001 samples.

---

## `time_scaling.py`

### `compute_segment_durations(q_waypoints, velocity_limits, acceleration_limits) → List[float]`

```
Δt_i = max_j(|q_{i+1,j} - q_{i,j}| / v_max_j)
```

### `time_scale_path_trapezoidal(q_waypoints, v_limits, a_limits) → List[float]`

Delegates to `compute_segment_durations` (simple velocity-limited timing; full trapezoidal profile used via `trapezoidal.py`).

### `synchronize_segment_times(segment_times) → List[float]`

`max(Δt_i, 1e-9)` — ensures all durations are positive.

---

## `retiming.py`

### `retime_joint_path(q_waypoints, velocity_limits, acceleration_limits, method="trapezoidal") → JointTrajectory`

1. Computes segment durations via `time_scale_path_trapezoidal`.
2. Assigns cumulative timestamps.
3. Computes velocities: `v_i = (q_{i+1} - q_i) / Δt_i`.
4. Returns `JointTrajectory` with zero accelerations (trapezoidal kinematic model).

### `synchronize_by_slowest_joint(q_waypoints, v_limits, a_limits) → JointTrajectory`

Alias for `retime_joint_path`.

### `validate_retiming_result(trajectory) → bool`

Checks non-empty points and monotonically increasing timestamps.

---

## `trajectory_sampler.py`

### `sample_trajectory(trajectory, dt) → List[JointTrajectoryPoint]`
### `sample_q_qdot_qddot(trajectory, dt) → List[Tuple]`

Returns `(q, q_dot, q_ddot)` tuples at uniform `dt` intervals.

### `sample_tcp_pose_over_time(trajectory, fk_solver, tcp_frame) → List`

Applies `fk_solver(q, tcp_frame)` at each trajectory waypoint to obtain the Cartesian TCP path.

### `compute_trajectory_duration(trajectory) → float`

`trajectory.duration`

---

## `trajectory_validator.py`

Per-point validation functions. Each returns `(ok: bool, failed_index: int | None, reason: str)`.

| Function | Check |
|---|---|
| `validate_joint_position_limits` | `lower ≤ q ≤ upper` per point |
| `validate_velocity_limits` | `|v| ≤ limits + 1e-9` per point |
| `validate_acceleration_limits` | `|a| ≤ limits + 1e-9` per point |
| `validate_jerk_limits` | `|jerk| ≤ limits + 1e-9` per point (if jerk is set) |
| `validate_trajectory_continuity` | Timestamps strictly increasing |
| `validate_collision_at_samples` | `collision_checker.check_state(q)` at `dt` intervals |
| `validate_clearance_margin` | `minimum_clearance ≥ threshold` per waypoint |
| `validate_singularity_margin` | `condition_number(J(q)) ≤ threshold` per waypoint |
| `validate_tcp_tracking_error` | Stub — always returns OK |
