# Trajectory

Trajectory generation assigns time to a geometric path.

- Cubic: position and velocity boundary constraints.
- Quintic: position, velocity, and acceleration boundary constraints.
- Trapezoidal/triangular: velocity and acceleration limited 1D profiles.
- Retiming: assigns timestamps to joint waypoints. It is separate from path planning.
- S-curve: implemented for rest-to-rest zero boundary velocity/acceleration profiles using a jerk-limited smoothstep profile. Nonzero boundary conditions return `NOT_IMPLEMENTED`.
