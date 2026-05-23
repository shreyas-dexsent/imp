from __future__ import annotations

from robot_engine.collision.collision_checker import check_scene
from robot_engine.collision.collision_world import CollisionWorld
from robot_engine.interfaces.schemas import AlgorithmError, GraspFeasibilityRequest, GraspFeasibilityResult
from robot_engine.kinematics.ik_solver import solve_ik


def evaluate_grasp_candidate(request: GraspFeasibilityRequest, world: CollisionWorld | None = None) -> GraspFeasibilityResult:
    reasons = []
    ik_result = solve_ik(request.ik_request) if request.ik_request else None
    if ik_result and not ik_result.ok:
        reasons.append(ik_result.error or AlgorithmError(code=ik_result.reason, message="IK rejected grasp."))

    collision_result = check_scene(world) if world is not None else None
    if collision_result and collision_result.collision:
        reasons.append(AlgorithmError(code="COLLISION", message="Grasp scene is in collision.", details={"pairs": collision_result.colliding_pairs}))

    return GraspFeasibilityResult(
        grasp_id=request.candidate.grasp_id,
        feasible=not reasons,
        ik=ik_result,
        collision=collision_result,
        rejection_reasons=reasons,
    )
