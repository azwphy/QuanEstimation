import pytest
import numpy as np
from quanestimation import ControlOpt


@pytest.fixture(autouse=True)
def _isolate_cwd(monkeypatch, tmp_path):
    """Redirect CWD to tmp_path to isolate .dat/.npy writes between tests."""
    monkeypatch.chdir(tmp_path)


def test_controlopt_factory_methods():
    """Verify all supported factory methods exist."""
    for method in ["auto-GRAPE", "GRAPE", "PSO", "DE"]:
        copt = ControlOpt(
            savefile=False,
            method=method,
            max_episode=10,
            seed=1234,
        )
        assert copt is not None


def test_controlopt_invalid_method():
    """Verify factory raises on unsupported method."""
    with pytest.raises(ValueError, match="is not a valid value for method"):
        ControlOpt(savefile=False, method="INVALID")


def test_dynamics_dH_not_list():
    """Verify TypeError when dH is not a list."""
    copt = ControlOpt(method="GRAPE", max_episode=10, seed=1234)
    H0 = np.eye(2, dtype=np.complex128)
    rho0 = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
    tspan = np.linspace(0, 1, 20)

    with pytest.raises(
        TypeError, match="The derivative of Hamiltonian should be a list"
    ):
        copt.dynamics(tspan, rho0, H0, dH=np.eye(2), Hc=[])


def test_dynamics_invalid_dyn_method():
    """Verify invalid dyn_method raises."""
    copt = ControlOpt(method="GRAPE", max_episode=10, seed=1234)
    H0 = np.eye(2, dtype=np.complex128)
    rho0 = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
    dH = [np.zeros((2, 2), dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    with pytest.raises(Exception):
        copt.dynamics(tspan, rho0, H0, dH, Hc=[], dyn_method="invalid")


def test_dynamics_basic_construction():
    """Verify basic dynamics setup completes without error."""
    copt = ControlOpt(method="GRAPE", max_episode=10, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    Hc = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    copt.dynamics(tspan, rho0, H0, dH, Hc)
    assert copt.dynamics_type == "lindblad"
    assert copt.scheme is not None


def test_de_algorithm_constructs():
    """Verify DE algorithm option constructs."""
    copt = ControlOpt(
        method="DE",
        max_episode=10,
        p_num=5,
        seed=1234,
    )
    assert copt is not None


def test_pso_algorithm_constructs():
    """Verify PSO algorithm option constructs."""
    copt = ControlOpt(
        method="PSO",
        max_episode=10,
        p_num=5,
        seed=1234,
    )
    assert copt is not None


def test_qfim_full_pipeline():
    """Verify full QFIM optimization pipeline: dynamics → QFIM → optimize!."""
    copt = ControlOpt(method="GRAPE", Adam=True, max_episode=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    Hc = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    copt.dynamics(tspan, rho0, H0, dH, Hc)
    copt.QFIM()
    assert hasattr(copt, "obj")
    assert copt.obj is not None


def test_cfim_full_pipeline():
    """Verify full CFIM optimization pipeline: dynamics → CFIM → optimize!."""
    copt = ControlOpt(method="GRAPE", Adam=True, max_episode=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    Hc = [np.eye(2, dtype=np.complex128)]
    M = [
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
        np.array([[0, 0], [0, 1]], dtype=np.complex128),
    ]
    tspan = np.linspace(0, 1, 20)

    copt.dynamics(tspan, rho0, H0, dH, Hc)
    copt.CFIM(M=M)
    assert hasattr(copt, "obj")
    assert copt.obj is not None


def test_controlopt_pso_run():
    """Verify PSO control optimization executes: dynamics + QFIM."""
    copt = ControlOpt(method="PSO", max_episode=5, p_num=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    Hc = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    copt.dynamics(tspan, rho0, H0, dH, Hc)
    copt.QFIM()
    assert copt.obj is not None


def test_controlopt_de_run():
    """Verify DE control optimization executes: dynamics + QFIM."""
    copt = ControlOpt(method="DE", max_episode=5, p_num=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    Hc = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    copt.dynamics(tspan, rho0, H0, dH, Hc)
    copt.QFIM()
    assert copt.obj is not None


def test_controlopt_cfim_run():
    """Verify CFIM target with PSO executes correctly."""
    copt = ControlOpt(method="PSO", max_episode=5, p_num=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    Hc = [np.eye(2, dtype=np.complex128)]
    M = [
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
        np.array([[0, 0], [0, 1]], dtype=np.complex128),
    ]
    tspan = np.linspace(0, 1, 20)

    copt.dynamics(tspan, rho0, H0, dH, Hc)
    copt.CFIM(M=M)
    assert copt.obj is not None


def test_controlopt_multi_ctrl():
    """Verify multi-control-channel pipeline with DE."""
    copt = ControlOpt(method="DE", max_episode=5, p_num=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    Hc = [np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    copt.dynamics(tspan, rho0, H0, dH, Hc)
    copt.QFIM()
    assert copt.obj is not None
