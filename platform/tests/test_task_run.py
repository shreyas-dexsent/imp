"""P5 headline: drive a synthetic chain end-to-end through ``imp_tasks``.

Same chain as ``test_smoke_motion_chain.py`` (perception Pose6D[cam] ->
spatial-transform -> IK -> ompl plan) but composed *as a task.yaml* and
run by :class:`imp_tasks.TaskRun`, not by hand-wired threads. Proves the
platform claim: a YAML task file is enough to wire and execute a motion
chain on the live bus, no Python orchestration code per task.

What the test asserts:

1. ``TaskSpec.from_yaml`` + ``compile_task`` succeed on the synthetic
   ``task.yaml`` written into ``tmp_path``.
2. ``TaskRun.run()`` reaches :class:`imp_tasks.RunStatus.SUCCEEDED` --
   i.e. the planner's ``Path`` lands on its keyexpr within the stage
   timeout, advancing the FSM to the terminal stage.
3. ``RunResult.stages_completed`` matches the sequence we wrote.
4. The ``run-store`` (when the test imports it) wrote
   ``runs/<run_id>/meta.json`` in the temp workspace.

Gated on ``pin + ompl + zenoh`` -- skips on hosts without the full env
just like the other heavy integration tests.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zenoh")
pytest.importorskip("pinocchio")
pytest.importorskip("ompl")

from scipy.spatial.transform import Rotation  # noqa: E402

from algorithms.descriptions import RobotSystemDescription  # noqa: E402
from algorithms.kinematics import fk_local  # noqa: E402
from algorithms.resolved import KinematicModel  # noqa: E402

from imp_sdk import Bus, QosClass, keyexpr  # noqa: E402
from imp_sdk.schemas import imp_pb2  # noqa: E402


PLATFORM = Path(__file__).resolve().parents[1]
WORLDS = PLATFORM / "modules" / "motion-core" / "algorithms" / "configs" / "worlds"
ROBOTS = PLATFORM / "modules" / "motion-core" / "algorithms" / "configs" / "robots"


def test_task_yaml_drives_synthetic_chain_to_succeeded(tmp_path: Path):
    """Compose transform -> IK -> plan as a task.yaml and run it."""
    from imp_tasks import RunStatus, TaskSpec, compile_task
    from imp_tasks.runtime import TaskRun

    station = "tasksmoke"
    rs_path = ROBOTS / "franka_fr3_with_franka_hand.yaml"
    world_path = WORLDS / "franka_robot_only_world.yaml"

    # ------------------------------------------------------------------
    # Pick a reachable target so IK + planner both succeed.
    # ------------------------------------------------------------------
    system = RobotSystemDescription.from_yaml(rs_path)
    model = KinematicModel.from_robot_system(system)
    spec_chain = model.chain("arm")
    tcp = spec_chain.tcp_frame or spec_chain.tip_frame
    home = system.named_joint_state("home")
    q_home = np.array([home[n] for n in model.active_joint_names], dtype=float)

    q_target = q_home.copy()
    q_target[0] += 0.10
    T_base_target = fk_local(model, q_target, tcp)
    target_pos = T_base_target[:3, 3]
    target_quat = Rotation.from_matrix(T_base_target[:3, :3]).as_quat()

    # eye-to-hand: camera at (0.5, 0, 0.5) in the base frame. The
    # synthetic perception pose is the same target expressed in the
    # camera frame (== T_camera_base @ p_base).
    T_base_camera = np.eye(4)
    T_base_camera[:3, 3] = (0.5, 0.0, 0.5)
    T_camera_base = np.eye(4)
    T_camera_base[:3, :3] = T_base_camera[:3, :3].T
    T_camera_base[:3, 3] = -T_base_camera[:3, :3].T @ T_base_camera[:3, 3]
    target_in_camera = (T_camera_base @ np.append(target_pos, 1.0))[:3]

    pose_key = f"imp/{station}/perc/s1/pose"
    plan_path_key = f"imp/{station}/motion/plan/path"

    # ------------------------------------------------------------------
    # Write task.yaml + run it.
    # ------------------------------------------------------------------
    task_yaml = f"""
schema: imp.task
version: 1
id: synthetic_pick
station: {station}

graph:
  nodes:
    - id: transform
      plugin: spatial-transform
      params:
        station: {station}
        pose_key: {pose_key}
        out_plan: ik
        base_frame: {system.robot.base_frame}
    - id: ik
      plugin: motion-pinocchio
      class: IkModule
      params:
        station: {station}
        robot: fr3
        robot_system_path: {rs_path}
    - id: plan
      plugin: motion-ompl
      params:
        station: {station}
        robot: fr3
        world_path: {world_path}
        random_seed: 0

sequence:
  - {{ stage: solve, until: {plan_path_key} }}
"""
    task_path = tmp_path / "task.yaml"
    task_path.write_text(task_yaml)

    # Compile happens against the actually-installed entry points.
    spec = TaskSpec.from_yaml(task_path)
    assert spec.id == "synthetic_pick"
    compiled = compile_task(spec)
    assert len(compiled.nodes) == 3

    # ------------------------------------------------------------------
    # Run the task, feeding the synthetic inputs from another thread.
    # ------------------------------------------------------------------
    run = TaskRun(compiled, stage_timeout_s=30.0)

    def _feed() -> None:
        time.sleep(0.8)  # wait for runtime spin-up + Zenoh discovery
        bus = Bus.open()
        try:
            state = imp_pb2.RobotState(
                header=imp_pb2.Header(schema="imp.RobotState/1"),
                q=q_home.tolist(),
                mode="idle",
            )
            bus.put(keyexpr.hal(station, "fr3", "state"), state, QosClass.STATE)
            time.sleep(0.05)
            bus.put(
                keyexpr.tf(station),
                imp_pb2.TfEdge(
                    header=imp_pb2.Header(schema="imp.TfEdge/1"),
                    parent_frame=system.robot.base_frame,
                    child_frame="camera",
                    matrix=T_base_camera.flatten().tolist(),
                ),
                QosClass.STATE,
            )
            time.sleep(0.05)
            bus.put(
                pose_key,
                imp_pb2.Pose6D(
                    header=imp_pb2.Header(schema="imp.Pose6D/1", frame_id="camera"),
                    object_id="target",
                    position_m=target_in_camera.tolist(),
                    quat_xyzw=target_quat.tolist(),
                    confidence=1.0,
                    valid=True,
                ),
                QosClass.STATE,
            )
            # Nudge the state a few times so the planner's compute fires
            # after the goal lands.
            for _ in range(5):
                bus.put(keyexpr.hal(station, "fr3", "state"), state, QosClass.STATE)
                time.sleep(0.2)
        finally:
            bus.close()

    threading.Thread(target=_feed, daemon=True).start()
    result = run.run()

    assert result.status == RunStatus.SUCCEEDED, (
        f"task did not reach SUCCEEDED: status={result.status} "
        f"reject_reason={result.reject_reason!r}"
    )
    assert result.stages_completed == ["solve"]
    assert result.elapsed_s < 30.0


def test_run_task_job_writes_meta_json(tmp_path: Path):
    """The ``run-task`` job persists meta.json to the workspace store."""
    from imp_job_run_task import RunTaskRequest, run_task

    # Write a tiny task that's certain to time out (no nodes do anything
    # in the time window), then check that the failure path still writes
    # meta.json -- that exercises the persistence wiring without paying
    # the cost of a full motion chain.
    station = "metawrite"
    task_yaml = f"""
schema: imp.task
version: 1
id: timeout_demo
station: {station}

graph:
  nodes:
    - id: tf
      plugin: spatial-tf
      params:
        station: {station}

sequence:
  - {{ stage: wait, until: imp/{station}/motion/tf/frames }}
"""
    task_path = tmp_path / "task.yaml"
    task_path.write_text(task_yaml)
    workspace = tmp_path / "ws"

    result = run_task(RunTaskRequest(
        task_path=str(task_path),
        workspace_root=str(workspace),
        stage_timeout_s=1.0,  # short -> finishes the test quickly
    ))

    # We expect timeout (no one published a TfEdge), but meta.json must
    # exist either way.
    assert result.status in ("timeout", "failed")
    assert result.meta_path is not None
    meta = json.loads(Path(result.meta_path).read_text())
    assert meta["task_id"] == "timeout_demo"
    assert meta["status"] == result.status
