from __future__ import annotations

from robot_engine.collision.collision_checker import check_pair
from robot_engine.collision.distance_queries import minimum_distance_pair


def pairwise_collision_query(object_a, object_b):
    return check_pair(object_a, object_b)


def pairwise_distance_query(object_a, object_b):
    return minimum_distance_pair(object_a, object_b)

