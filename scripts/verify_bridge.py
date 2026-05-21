"""End-to-end bridge verification script for QuanEstimation plan-1 migration.

Verifies that the Python→Julia bridge works for:
  Scene 1: Control optimization (qubit, autoGRAPE) with GeneralScheme
  Scene 2: State optimization (qubit, AD) with GeneralScheme
  Scene 3: Basic Julia module access and type conversions
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quanestimation import ControlOpt, StateOpt, MeasurementOpt, ComprehensiveOpt


def _qubit_setup():
    """Return common 2-qubit test parameters."""
    H0 = np.array([[1.0, 0], [0, -1.0]], dtype=np.complex128)
    dH = [np.eye(2, dtype=np.complex128)]
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    Hc = [np.eye(2, dtype=np.complex128)]
    tspan = np.linspace(0, 1, 20)
    return H0, dH, rho0, Hc, tspan


def test_scene1_control_opt():
    """Scene 1: ControlSystem constructs GeneralScheme successfully."""
    print("Scene 1: Control optimization ...", end=" ")
    copt = ControlOpt(method="GRAPE", max_episode=5, seed=1234)
    H0, dH, rho0, Hc, tspan = _qubit_setup()
    copt.dynamics(tspan, rho0, H0, dH, Hc)
    assert copt.scheme is not None, "GeneralScheme not constructed"
    assert copt.dynamics_type == "lindblad", "Wrong dynamics type"
    print("OK")


def test_scene2_state_opt():
    """Scene 2: StateSystem constructs GeneralScheme successfully."""
    print("Scene 2: State optimization  ...", end=" ")
    sopt = StateOpt(method="AD", max_episode=5, seed=1234)
    H0, dH, _, _, tspan = _qubit_setup()
    sopt.dynamics(tspan, H0, dH)
    assert sopt.scheme is not None, "GeneralScheme not constructed"
    print("OK")


def test_scene3_bridge_basics():
    """Scene 3: Verify Julia module access and basic type conversions."""
    print("Scene 3: Bridge basics          ...", end=" ")
    from quanestimation import QJL

    # Verify key Julia types are accessible
    assert hasattr(QJL, "GeneralScheme"), "GeneralScheme missing"
    assert hasattr(QJL, "Lindblad"), "Lindblad missing"
    assert hasattr(QJL, "ControlOpt"), "ControlOpt missing"
    assert hasattr(QJL, "StateOpt"), "StateOpt missing"
    assert hasattr(QJL, "autoGRAPE"), "autoGRAPE missing"
    assert hasattr(QJL, "AD"), "AD missing"

    # Verify GeneralState PyExt overload works for density matrices
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    state = QJL.GeneralState(rho0)
    assert state is not None, "GeneralState conversion failed"

    print("OK")


def test_scene4_all_factories():
    """Verify all optimization factory methods across modules."""
    print("Scene 4: Factory methods        ...", end=" ")
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


if __name__ == "__main__":
    test_scene1_control_opt()
    test_scene2_state_opt()
    test_scene3_bridge_basics()
    test_scene4_all_factories()
    print("\nAll bridge verification scenes passed.")
