"""End-to-end bridge verification script for QuanEstimation 

Verifies that the Python→Julia bridge works for:
  Scene 1: Control optimization (qubit, GRAPE) — dynamics + QFIM full pipeline
  Scene 2: State optimization (qubit, AD) — dynamics + QFIM full pipeline
  Scene 3: Measurement optimization (qubit, DE) — dynamics + CFIM full pipeline
  Scene 4: Basic Julia module access and type conversions
  Scene 5: All factory methods across 4 modules
  Scene 6: Mopt_Projection export and MeasurementOpt dynamics
  Scene 7: Comprehensive SC + QFIM full pipeline
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quanestimation import ControlOpt, StateOpt, MeasurementOpt, ComprehensiveOpt


def _qubit_setup():
    """Return common qubit test parameters."""
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    Hc = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)
    return H0, dH, rho0, Hc, tspan


def test_scene1_control_qfim():
    """Scene 1: Control optimization full pipeline — QFIM."""
    print("Scene 1: Control QFIM pipeline  ...", end=" ")
    copt = ControlOpt(method="GRAPE", Adam=True, max_episode=3, seed=1234)
    H0, dH, rho0, Hc, tspan = _qubit_setup()
    copt.dynamics(tspan, rho0, H0, dH, Hc)
    copt.QFIM()
    assert copt.obj is not None, "QFIM objective not created"
    print("OK")


def test_scene2_state_qfim():
    """Scene 2: State optimization full pipeline — QFIM."""
    print("Scene 2: State QFIM pipeline    ...", end=" ")
    sopt = StateOpt(method="AD", max_episode=3, seed=1234)
    H0, dH, _, _, tspan = _qubit_setup()
    sopt.dynamics(tspan, H0, dH)
    sopt.QFIM()
    assert sopt.obj is not None, "QFIM objective not created"
    print("OK")


def test_scene3_measurement_cfim():
    """Scene 3: Measurement optimization full pipeline — CFIM."""
    print("Scene 3: Measure CFIM pipeline  ...", end=" ")
    mopt = MeasurementOpt(method="DE", max_episode=3, p_num=3, seed=1234)
    H0, dH, rho0, _, tspan = _qubit_setup()
    mopt.dynamics(tspan, rho0, H0, dH)
    mopt.CFIM()
    assert mopt.obj is not None, "CFIM objective not created"
    print("OK")


def test_scene4_bridge_basics():
    """Scene 4: Verify Julia module access and key exported types."""
    print("Scene 4: Bridge basics           ...", end=" ")
    from quanestimation import QJL

    assert hasattr(QJL, "GeneralScheme"), "GeneralScheme missing"
    assert hasattr(QJL, "Lindblad"), "Lindblad missing"
    assert hasattr(QJL, "Mopt_Projection"), "Mopt_Projection not exported"
    assert hasattr(QJL, "Mopt_LinearComb"), "Mopt_LinearComb not exported"
    assert hasattr(QJL, "Mopt_Rotation"), "Mopt_Rotation not exported"
    assert hasattr(QJL, "StateControlOpt"), "StateControlOpt missing"
    assert hasattr(QJL, "ControlMeasurementOpt"), "ControlMeasurementOpt missing"

    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    state = QJL.GeneralState(rho0)
    assert state is not None, "GeneralState conversion failed"

    print("OK")


def test_scene5_all_factories():
    """Scene 5: Verify all factory methods across 4 modules."""
    print("Scene 5: Factory methods         ...", end=" ")
    modules = {
        "ControlOpt": (ControlOpt, ["auto-GRAPE", "GRAPE", "PSO", "DE"]),
        "StateOpt": (StateOpt, ["AD", "DE", "PSO", "NM", "RI"]),
        "MeasurementOpt": (MeasurementOpt, ["AD", "PSO", "DE"]),
        "ComprehensiveOpt": (ComprehensiveOpt, ["AD", "DE", "PSO"]),
    }
    for name, (factory, methods) in modules.items():
        for method in methods:
            kwargs = dict(savefile=False, method=method, max_episode=3, seed=1234)
            if method in ("DE", "PSO", "NM"):
                kwargs["p_num"] = 3
            instance = factory(**kwargs)
            assert instance is not None, f"{name}.{method} factory failed"
    print("OK")


def test_scene6_control_cfim():
    """Scene 6: Control optimization — CFIM with explicit M."""
    print("Scene 6: Control CFIM pipeline   ...", end=" ")
    copt = ControlOpt(method="GRAPE", Adam=True, max_episode=3, seed=1234)
    H0, dH, rho0, Hc, tspan = _qubit_setup()
    M = [np.array([[1, 0], [0, 0]], dtype=np.complex128),
         np.array([[0, 0], [0, 1]], dtype=np.complex128)]
    copt.dynamics(tspan, rho0, H0, dH, Hc)
    copt.CFIM(M=M)
    assert copt.obj is not None, "CFIM objective not created"
    print("OK")


def test_scene7_comprehensive_sc_qfim():
    """Scene 7: Comprehensive SC + QFIM pipeline — DE algorithm."""
    print("Scene 7: Comprehensive SC QFIM    ...", end=" ")
    copt = ComprehensiveOpt(
        savefile=False, method="DE", max_episode=3, p_num=3, seed=1234
    )
    H0, dH, _rho0, Hc, tspan = _qubit_setup()
    ctrl = [np.zeros(len(tspan) - 1) for _ in range(len(Hc))]
    copt.dynamics(tspan, H0, dH, Hc=Hc, ctrl=ctrl)
    copt.SC(target="QFIM")
    assert copt.obj is not None
    print("OK")


if __name__ == "__main__":
    test_scene1_control_qfim()
    test_scene2_state_qfim()
    test_scene3_measurement_cfim()
    test_scene4_bridge_basics()
    test_scene5_all_factories()
    test_scene6_control_cfim()
    test_scene7_comprehensive_sc_qfim()
    print("\nAll 7 bridge verification scenes passed.")
