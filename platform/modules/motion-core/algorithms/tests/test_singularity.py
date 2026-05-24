"""Tests for kinematics.singularity - pure SVD metrics on a Jacobian."""
from __future__ import annotations

import numpy as np
import pytest

from algorithms.kinematics.singularity import (
    condition_number,
    inverse_condition_number,
    manipulability,
    min_singular_value,
    singularity_report,
)


# ---------------------------------------------------------------------------
# Well-conditioned matrices
# ---------------------------------------------------------------------------


def test_identity_jacobian_is_perfectly_conditioned():
    J = np.eye(6, 6)
    assert manipulability(J) == pytest.approx(1.0)
    assert condition_number(J) == pytest.approx(1.0)
    assert inverse_condition_number(J) == pytest.approx(1.0)
    assert min_singular_value(J) == pytest.approx(1.0)


def test_full_rank_6x7_jacobian_has_finite_metrics():
    rng = np.random.default_rng(7)
    J = rng.normal(size=(6, 7))
    rep = singularity_report(J)
    assert rep["rank"] == 6
    assert np.isfinite(rep["condition_number"])
    assert rep["min_singular_value"] > 0.0
    assert rep["manipulability"] > 0.0


# ---------------------------------------------------------------------------
# Singular and rank-deficient cases
# ---------------------------------------------------------------------------


def test_rank_deficient_jacobian_has_zero_min_singular_value():
    # Drop one row from an otherwise well-conditioned matrix; condition explodes.
    J = np.eye(6, 7)
    J[5, :] = 0.0  # rank 5
    assert min_singular_value(J) == pytest.approx(0.0, abs=1e-12)
    assert manipulability(J) == pytest.approx(0.0, abs=1e-12)
    assert condition_number(J) == float("inf")
    assert inverse_condition_number(J) == pytest.approx(0.0, abs=1e-12)


def test_singularity_report_detects_rank():
    J = np.zeros((6, 6))
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    rep = singularity_report(J)
    assert rep["rank"] == 2


def test_empty_jacobian_returns_safe_defaults():
    J = np.zeros((0, 0))
    rep = singularity_report(J)
    assert rep["rank"] == 0
    assert rep["condition_number"] == float("inf")
    assert rep["manipulability"] == 0.0
    assert manipulability(J) == 0.0
    assert min_singular_value(J) == 0.0


# ---------------------------------------------------------------------------
# Numerical consistency
# ---------------------------------------------------------------------------


def test_metric_identities():
    """The four scalar metrics should be self-consistent."""
    rng = np.random.default_rng(11)
    J = rng.normal(size=(6, 7))
    rep = singularity_report(J)

    # inverse_condition_number == 1 / condition_number (for finite cond).
    np.testing.assert_allclose(
        rep["inverse_condition_number"], 1.0 / rep["condition_number"], rtol=1e-12
    )

    # Manipulability == product of singular values.
    sigma = np.linalg.svd(J, compute_uv=False)
    np.testing.assert_allclose(rep["manipulability"], float(np.prod(sigma)), rtol=1e-12)


def test_scaling_jacobian_scales_metrics_predictably():
    """Scaling J by alpha scales sigma by alpha, leaves cond unchanged,
    multiplies manipulability by alpha^min(6, dof)."""
    rng = np.random.default_rng(13)
    J = rng.normal(size=(6, 6))
    alpha = 5.0
    rep_J = singularity_report(J)
    rep_aJ = singularity_report(alpha * J)

    np.testing.assert_allclose(rep_aJ["condition_number"], rep_J["condition_number"], rtol=1e-10)
    np.testing.assert_allclose(rep_aJ["min_singular_value"], alpha * rep_J["min_singular_value"], rtol=1e-10)
    np.testing.assert_allclose(rep_aJ["manipulability"], alpha**6 * rep_J["manipulability"], rtol=1e-10)
