from __future__ import annotations

from typing import List

from robot_engine.collision.collision_object import CollisionObject
from robot_engine.collision.collision_world import CollisionWorld
from robot_engine.interfaces.schemas import AlgorithmError, CollisionCheckResult


def _coal_check_pair(a: CollisionObject, b: CollisionObject):
    """Return (collided: bool, contacts: list) using Coal, or None if unavailable."""
    try:
        import coal
    except Exception:
        return None

    ca = a.coal_object()
    cb = b.coal_object()
    if ca is None or cb is None:
        return None

    req = coal.CollisionRequest()
    req.num_max_contacts = 16
    res = coal.CollisionResult()
    try:
        n = coal.collide(ca, cb, req, res)
    except Exception:
        return None

    contacts = []
    for contact in res.getContacts():
        contacts.append({
            "position": list(map(float, contact.pos)),
            "normal": list(map(float, contact.normal)),
            "penetration_depth": float(contact.penetration_depth),
        })
    return bool(n > 0), contacts


def check_pair(a: CollisionObject, b: CollisionObject) -> CollisionCheckResult:
    result = _coal_check_pair(a, b)
    if result is not None:
        collided, contacts = result
        return CollisionCheckResult(
            collision=collided,
            colliding_pairs=[[a.object_id, b.object_id]] if collided else [],
            contacts=contacts,
        )

    return CollisionCheckResult(
        collision=False,
        ok=False,
        error=AlgorithmError(
            code="COLLISION_BACKEND_UNAVAILABLE",
            message="Exact mesh collision is unavailable for this pair; AABB fallback is disabled.",
            details={
                "object_a": a.object_id,
                "object_b": b.object_id,
                "coal_ready_a": a.geometry.coal_geometry is not None,
                "coal_ready_b": b.geometry.coal_geometry is not None,
                "coal_error_a": getattr(a.geometry, "coal_error", None),
                "coal_error_b": getattr(b.geometry, "coal_error", None),
            },
        ),
    )


def check_active_pairs(world: CollisionWorld) -> CollisionCheckResult:
    all_contacts = []
    pairs = []
    for a_id, b_id in world.active_pairs():
        result = check_pair(world.get(a_id), world.get(b_id))
        if not result.ok:
            return result
        if result.collision:
            pairs.extend(result.colliding_pairs)
            all_contacts.extend(result.contacts)
    return CollisionCheckResult(collision=bool(pairs), colliding_pairs=pairs, contacts=all_contacts)


def check_scene(world: CollisionWorld) -> CollisionCheckResult:
    return check_active_pairs(world)
