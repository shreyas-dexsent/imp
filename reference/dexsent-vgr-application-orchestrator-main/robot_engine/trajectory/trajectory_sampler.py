from __future__ import annotations


def sample_trajectory(trajectory, dt: float):
    return trajectory.sample(dt)


def sample_q_qdot_qddot(trajectory, dt: float):
    return [(p.q, p.q_dot, p.q_ddot) for p in trajectory.sample(dt)]


def sample_tcp_pose_over_time(trajectory, fk_solver, tcp_frame):
    return [fk_solver(point.q, tcp_frame) for point in trajectory.points]


def compute_trajectory_duration(trajectory):
    return trajectory.duration

