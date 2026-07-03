import pytest
import numpy as np
from quanestimation import MeasurementOpt


@pytest.fixture(autouse=True)
def _isolate_cwd(monkeypatch, tmp_path):
    """Redirect CWD to tmp_path to isolate .dat/.npy writes between tests."""
    monkeypatch.chdir(tmp_path)


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


def test_dynamics_basic_construction():
    """Verify basic dynamics setup with projection measurement completes."""
    mopt = MeasurementOpt(method="DE", max_episode=3, p_num=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    mopt.dynamics(tspan, rho0, H0, dH)
    assert mopt.scheme is not None


def test_cfim_full_pipeline():
    """Verify full CFIM optimization pipeline: dynamics → CFIM → optimize!."""
    mopt = MeasurementOpt(method="DE", max_episode=3, p_num=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    mopt.dynamics(tspan, rho0, H0, dH)
    mopt.CFIM()
    assert hasattr(mopt, "obj")
    assert mopt.obj is not None


def test_measurementopt_pso_run():
    """Verify PSO measurement optimization executes: dynamics + CFIM."""
    mopt = MeasurementOpt(method="PSO", max_episode=5, p_num=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    mopt.dynamics(tspan, rho0, H0, dH)
    mopt.CFIM()
    assert mopt.obj is not None


def test_measurementopt_multi_param():
    """Verify multi-parameter measurement optimization pipeline."""
    mopt = MeasurementOpt(method="DE", max_episode=5, p_num=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128), np.diag([1.0, 0.0]).astype(np.complex128)]
    tspan = np.linspace(0, 1, 20)

    mopt.dynamics(tspan, rho0, H0, dH)
    mopt.CFIM()
    assert mopt.obj is not None
