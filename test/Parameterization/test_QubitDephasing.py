"""Tests for QubitDephasing (Phase 2.2)."""

import numpy as np


class TestQubitDephasing:
    def test_construct(self):
        from quanestimation.base.Parameterization.GeneralDynamics import (
            QubitDephasing,
        )

        r = np.array([1.0, 0.0, 0.0])
        tspan = np.linspace(0, 2, 20)
        dyn = QubitDephasing(r, "z", 0.1, tspan)
        assert dyn is not None

    def test_all_axes(self):
        from quanestimation.base.Parameterization.GeneralDynamics import (
            QubitDephasing,
        )

        r = np.array([1.0, 1.0, 1.0])
        tspan = np.linspace(0, 2, 10)
        for axis in ["x", "y", "z"]:
            dyn = QubitDephasing(r, axis, 0.01, tspan)
            assert dyn is not None
