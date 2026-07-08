import pytest
import numpy as np
from quanestimation import ComprehensiveOpt


@pytest.fixture(autouse=True)
def _isolate_cwd(monkeypatch, tmp_path):
    """Redirect CWD to tmp_path to isolate .dat/.npy writes between tests."""
    monkeypatch.chdir(tmp_path)


def _qubit_setup():
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    Hc = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)
    return H0, dH, rho0, Hc, tspan


def test_comprehensiveopt_factory_methods():
    """Verify all supported factory methods exist."""
    for method in ["AD", "DE", "PSO"]:
        copt = ComprehensiveOpt(
            savefile=False,
            method=method,
            max_episode=5,
            seed=1234,
        )
        assert copt is not None


def test_comprehensiveopt_invalid_method():
    """Verify factory raises on unsupported method."""
    with pytest.raises(ValueError, match="is not a valid value for method"):
        ComprehensiveOpt(savefile=False, method="INVALID")


def test_dynamics_basic_construction():
    """Verify dynamics setup completes without error (SC mode)."""
    copt = ComprehensiveOpt(
        savefile=False, method="DE", max_episode=3, p_num=3, seed=1234
    )
    H0, dH, _rho0, Hc, tspan = _qubit_setup()
    ctrl = [np.zeros(len(tspan) - 1) for _ in range(len(Hc))]
    copt.dynamics(tspan, H0, dH, Hc=Hc, ctrl=ctrl)
    assert copt.dynamics_type == "dynamics"


def test_sc_qfim_full_pipeline():
    """State+Control + QFIM full pipeline: dynamics → SC(QFIM) → optimize!."""
    copt = ComprehensiveOpt(
        savefile=False, method="DE", max_episode=3, p_num=3, seed=1234
    )
    H0, dH, _rho0, Hc, tspan = _qubit_setup()
    ctrl = [np.zeros(len(tspan) - 1) for _ in range(len(Hc))]
    copt.dynamics(tspan, H0, dH, Hc=Hc, ctrl=ctrl)
    copt.SC(target="QFIM")
    assert copt.obj is not None


def test_cm_cfim_full_pipeline():
    """Control+Measurement + CFIM full pipeline."""
    copt = ComprehensiveOpt(
        savefile=False, method="DE", max_episode=3, p_num=3, seed=1234
    )
    H0, dH, rho0, Hc, tspan = _qubit_setup()
    ctrl = [np.zeros(len(tspan) - 1) for _ in range(len(Hc))]
    copt.dynamics(tspan, H0, dH, Hc=Hc, ctrl=ctrl)
    copt.CM(rho0=rho0)
    assert copt.obj is not None


def test_sm_cfim_full_pipeline():
    """State+Measurement + CFIM full pipeline (no controls)."""
    copt = ComprehensiveOpt(
        savefile=False, method="DE", max_episode=3, p_num=3, seed=1234
    )
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)

    copt.dynamics(tspan, H0, dH)
    copt.SM()
    assert copt.obj is not None


def test_scm_cfim_full_pipeline():
    """State+Control+Measurement + CFIM full pipeline."""
    copt = ComprehensiveOpt(
        savefile=False, method="DE", max_episode=3, p_num=3, seed=1234
    )
    H0, dH, rho0, Hc, tspan = _qubit_setup()
    ctrl = [np.zeros(len(tspan) - 1) for _ in range(len(Hc))]
    copt.dynamics(tspan, H0, dH, Hc=Hc, ctrl=ctrl)
    copt.SCM()
    assert copt.obj is not None


def test_sc_pso_run():
    """State+Control + PSO full pipeline."""
    copt = ComprehensiveOpt(
        savefile=False, method="PSO", max_episode=5, p_num=3, seed=1234
    )
    H0, dH, _rho0, Hc, tspan = _qubit_setup()
    ctrl = [np.zeros(len(tspan) - 1) for _ in range(len(Hc))]
    copt.dynamics(tspan, H0, dH, Hc=Hc, ctrl=ctrl)
    copt.SC(target="QFIM")
    assert copt.obj is not None


def test_cm_pso_run():
    """Control+Measurement + PSO full pipeline."""
    copt = ComprehensiveOpt(
        savefile=False, method="PSO", max_episode=5, p_num=3, seed=1234
    )
    H0, dH, rho0, Hc, tspan = _qubit_setup()
    ctrl = [np.zeros(len(tspan) - 1) for _ in range(len(Hc))]
    copt.dynamics(tspan, H0, dH, Hc=Hc, ctrl=ctrl)
    copt.CM(rho0=rho0)
    assert copt.obj is not None


def test_sc_ad_run():
    """State+Control + AD full pipeline."""
    copt = ComprehensiveOpt(savefile=False, method="AD", max_episode=5, seed=1234)
    H0, dH, _rho0, Hc, tspan = _qubit_setup()
    ctrl = [np.zeros(len(tspan) - 1) for _ in range(len(Hc))]
    copt.dynamics(tspan, H0, dH, Hc=Hc, ctrl=ctrl)
    copt.SC(target="QFIM")
    assert copt.obj is not None
