import pytest
import numpy as np
from quanestimation import StateOpt


def test_stateopt_factory_methods():
    """Verify all supported factory methods exist."""
    methods = {
        "AD": dict(savefile=False, method="AD", max_episode=5, seed=1234),
        "DE": dict(savefile=False, method="DE", max_episode=5, p_num=3, seed=1234),
        "PSO": dict(savefile=False, method="PSO", max_episode=5, p_num=3, seed=1234),
        "NM": dict(savefile=False, method="NM", max_episode=5, p_num=3, seed=1234),
        "RI": dict(savefile=False, method="RI", max_episode=5, seed=1234),
    }
    for method, kwargs in methods.items():
        sopt = StateOpt(**kwargs)
        assert sopt is not None


def test_stateopt_invalid_method():
    """Verify factory raises on unsupported method."""
    with pytest.raises(ValueError, match="is not a valid value for method"):
        StateOpt(savefile=False, method="INVALID")


def test_dynamics_dH_not_list():
    """Verify TypeError when dH is not a list."""
    sopt = StateOpt(method="AD", max_episode=5, seed=1234)
    H0 = np.eye(2, dtype=np.complex128)
    tspan = np.linspace(0, 1, 20)

    with pytest.raises(TypeError, match="The derivative of Hamiltonian should be a list"):
        sopt.dynamics(tspan, H0, dH=np.eye(2))


def test_dynamics_invalid_dyn_method():
    """Verify invalid dyn_method raises."""
    sopt = StateOpt(method="AD", max_episode=5, seed=1234)
    H0 = np.eye(2, dtype=np.complex128)
    dH = [np.zeros((2, 2), dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    with pytest.raises(Exception):
        sopt.dynamics(tspan, H0, dH, dyn_method="invalid")


def test_dynamics_basic_construction():
    """Verify basic dynamics setup completes without error."""
    sopt = StateOpt(method="AD", max_episode=5, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    sopt.dynamics(tspan, H0, dH)
    assert sopt.scheme is not None
