# Copyright (c) 2026 DexSent Robotics Pvt Ltd.
# All Rights Reserved.
#
# SPDX-License-Identifier: LicenseRef-Proprietary
# See LICENSE file for details.
#
# Author: Shreyas <shreyas.kumar@dexsentrobotics.com>
# Proprietary and confidential.

"""Pydantic schema for 4x4 homogeneous transforms.

A `TransformSpec` is the YAML-facing representation of an SE(3) rigid
transform. It carries explicit parent/child frame names and a 4x4 matrix
encoded as a nested list (YAML-friendly).

Validators enforce shape, finiteness, and the homogeneous bottom row.
Callers retrieve a NumPy view via :meth:`TransformSpec.as_matrix`.
"""
from __future__ import annotations

from typing import List

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class TransformSpec(BaseModel):
    """A 4x4 homogeneous transform with explicit parent and child frame names.

    Parameters
    ----------
    parent_frame : str
        Name of the frame the transform is expressed in (the "from" frame).
    child_frame : str
        Name of the frame the transform places (the "to" frame).
    matrix : list[list[float]]
        A 4x4 nested list. The bottom row must equal `[0, 0, 0, 1]`.
        Validated at load time; non-finite or mis-shaped matrices raise.
    """

    parent_frame: str
    child_frame: str
    matrix: List[List[float]]

    model_config = ConfigDict(extra="forbid")

    @field_validator("matrix")
    @classmethod
    def _validate_matrix(cls, value: List[List[float]]) -> List[List[float]]:
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("transform matrix must be 4x4")

        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("transform matrix contains non-finite values")
        if not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("transform bottom row must be [0, 0, 0, 1]")

        return value

    def as_matrix(self) -> np.ndarray:
        """Return the transform as a 4x4 `np.ndarray` of dtype `float64`."""
        return np.asarray(self.matrix, dtype=float)
