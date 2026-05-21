import pytest
import numpy as np
from quanestimation import MeasurementOpt


def test_measurementopt_factory_methods():
    """Verify all supported factory methods exist."""
    for method in ["AD", "PSO", "DE"]:
        mopt = MeasurementOpt(
            savefile=False,
            method=method,
            max_episode=5,
            seed=1234,
        )
        assert mopt is not None


def test_measurementopt_invalid_method():
    """Verify factory raises on unsupported method."""
    with pytest.raises(ValueError, match="is not a valid value for method"):
        MeasurementOpt(savefile=False, method="INVALID")


def test_dynamics_invalid_dyn_method():
    """Verify invalid dyn_method raises."""
    mopt = MeasurementOpt(method="AD", max_episode=5, seed=1234)
    H0 = np.eye(2, dtype=np.complex128)
    rho0 = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
    dH = [np.zeros((2, 2), dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    with pytest.raises(Exception):
        mopt.dynamics(tspan, rho0, H0, dH, dyn_method="invalid")
