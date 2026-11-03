import pytest
import numpy as np
from quanestimation import StateOpt


@pytest.fixture(autouse=True)
def _isolate_cwd(monkeypatch, tmp_path):
    """Redirect CWD to tmp_path to isolate .dat/.npy writes between tests."""
    monkeypatch.chdir(tmp_path)


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

    with pytest.raises(
        TypeError, match="The derivative of Hamiltonian should be a list"
    ):
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


def test_qfim_full_pipeline():
    """Verify full QFIM optimization pipeline: dynamics → QFIM → optimize!."""
    sopt = StateOpt(method="AD", max_episode=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    sopt.dynamics(tspan, H0, dH)
    sopt.QFIM()
    assert hasattr(sopt, "obj")
    assert sopt.obj is not None


def test_stateopt_pso_run():
    """Verify PSO state optimization executes: dynamics + QFIM."""
    sopt = StateOpt(method="PSO", max_episode=5, p_num=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    sopt.dynamics(tspan, H0, dH)
    sopt.QFIM()
    assert sopt.obj is not None


def test_stateopt_de_run():
    """Verify DE state optimization executes: dynamics + QFIM."""
    sopt = StateOpt(method="DE", max_episode=5, p_num=3, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    sopt.dynamics(tspan, H0, dH)
    sopt.QFIM()
    assert sopt.obj is not None


def test_stateopt_nm_run():
    """Verify Nelder-Mead state optimization executes: dynamics + QFIM."""
    sopt = StateOpt(method="NM", max_episode=5, p_num=5, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    sopt.dynamics(tspan, H0, dH)
    sopt.QFIM()
    assert sopt.obj is not None


def test_stateopt_ri_run():
    """Verify RI state optimization executes via Kraus + QFIM."""
    gamma = 0.1
    K1 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K2 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)
    K = [K1, K2]
    dK1 = [
        np.array([[0.0, 0.0], [0.0, -0.5 / np.sqrt(1 - gamma)]], dtype=np.complex128)
    ]
    dK2 = [np.array([[0.0, 0.5 / np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)]
    dK = [dK1, dK2]

    sopt = StateOpt(method="RI", max_episode=5, seed=1234)
    sopt.Kraus(K, dK)
    sopt.QFIM()
    assert sopt.obj is not None


def test_stateopt_cfim_run():
    """Verify CFIM target with AD executes correctly."""
    sopt = StateOpt(method="AD", max_episode=5, seed=1234)
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    M = [
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
        np.array([[0, 0], [0, 1]], dtype=np.complex128),
    ]
    tspan = np.linspace(0, 1, 20)

    sopt.dynamics(tspan, H0, dH)
    sopt.CFIM(M=M)
    assert sopt.obj is not None
