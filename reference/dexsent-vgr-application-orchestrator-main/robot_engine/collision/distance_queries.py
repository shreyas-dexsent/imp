from __future__ import annotations

from typing import List

import numpy as np

from robot_engine.collision.collision_object import CollisionObject
from robot_engine.collision.collision_world import CollisionWorld
from robot_engine.interfaces.schemas import AlgorithmError, DistanceQueryResult


def minimum_distance_pair(a: CollisionObject, b: CollisionObject) -> DistanceQueryResult:
    coal_a = a.coal_object()
    coal_b = b.coal_object()
    if coal_a is not None and coal_b is not None:
        try:
            import coal

            req = coal.DistanceRequest()
            res = coal.DistanceResult()
            distance = float(coal.distance(coal_a, coal_b, req, res))
            try:
                nearest = res.nearest_points
            except Exception:
                nearest = None
            p1 = list(map(float, nearest[0])) if nearest is not None else None
            p2 = list(map(float, nearest[1])) if nearest is not None else None
            return DistanceQueryResult(object_a=a.object_id, object_b=b.object_id, distance=distance, nearest_point_a=p1, nearest_point_b=p2, in_collision=distance <= 0)
        except Exception as exc:
            return DistanceQueryResult(object_a=a.object_id, object_b=b.object_id, ok=False, error=AlgorithmError(code="COLLISION_BACKEND_UNAVAILABLE", message=str(exc)))
    return DistanceQueryResult(
        object_a=a.object_id,
        object_b=b.object_id,
        ok=False,
        error=AlgorithmError(
            code="COLLISION_BACKEND_UNAVAILABLE",
            message="Exact mesh distance is unavailable for this pair; AABB fallback is disabled.",
            details={
                "coal_ready_a": a.geometry.coal_geometry is not None,
                "coal_ready_b": b.geometry.coal_geometry is not None,
                "coal_error_a": getattr(a.geometry, "coal_error", None),
                "coal_error_b": getattr(b.geometry, "coal_error", None),
            },
        ),
    )


def minimum_distance_to_all(world: CollisionWorld, object_id: str) -> List[DistanceQueryResult]:
    if object_id not in world.objects:
        return [DistanceQueryResult(object_a=object_id, object_b="", ok=False, error=AlgorithmError(code="OBJECT_NOT_FOUND", message=f"Missing object: {object_id}"))]
    return [minimum_distance_pair(world.get(object_id), obj) for oid, obj in world.objects.items() if oid != object_id]


def minimum_distances_active_pairs(world: CollisionWorld) -> List[DistanceQueryResult]:
    return [minimum_distance_pair(world.get(a), world.get(b)) for a, b in world.active_pairs()]


def aabb_distance(bounds_a: np.ndarray, bounds_b: np.ndarray):
    min_a, max_a = bounds_a
    min_b, max_b = bounds_b
    sep = np.maximum(0.0, np.maximum(min_b - max_a, min_a - max_b))
    colliding = bool(np.all(sep == 0.0))
    if colliding:
        return 0.0, None, None, True
    point_a = np.clip((min_b + max_b) / 2.0, min_a, max_a)
    point_b = np.clip((min_a + max_a) / 2.0, min_b, max_b)
    return float(np.linalg.norm(point_b - point_a)), point_a.tolist(), point_b.tolist(), False
