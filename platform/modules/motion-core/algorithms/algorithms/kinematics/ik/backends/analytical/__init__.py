"""Analytical IK backends."""

from algorithms.kinematics.ik.backends.analytical.base import AnalyticalIK
from algorithms.kinematics.ik.backends.analytical.opw import OPWIK
from algorithms.kinematics.ik.backends.analytical.registry import (
    clear,
    lookup,
    register,
)
from algorithms.kinematics.ik.backends.analytical.spherical_wrist_6r import (
    SphericalWrist6RIK,
)

# Friendly aliases for the top-level kinematics.ik re-export.
register_analytical = register
lookup_analytical = lookup

__all__ = [
    "AnalyticalIK",
    "OPWIK",
    "SphericalWrist6RIK",
    "clear",
    "lookup",
    "lookup_analytical",
    "register",
    "register_analytical",
]
