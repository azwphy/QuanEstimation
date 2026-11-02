"""Signature compatibility tests for paper-documented APIs.

Validates that every function signature documented in the QuanEstimation
papers (arXiv:2205.15588v3, arXiv:2405.12066v3) matches the current
implementation. Protects against accidental signature drift.
"""

import inspect
import pytest
from quanestimation import *


# ---- Independent function signatures (from paper code blocks) ----------

EXPECTED_FUNCTIONS = {
    # --- AsymptoticBound/CramerRao.py ---
    "CFIM": ("rho", "drho", "M", "eps"),
    "FIM": ("p", "dp", "eps"),
    "FI_Expt": ("y1", "y2", "dx", "ftype"),
    "SLD": ("rho", "drho", "rep", "eps"),
    "RLD": ("rho", "drho", "rep", "eps"),
    "LLD": ("rho", "drho", "rep", "eps"),
    "QFIM": ("rho", "drho", "LDtype", "exportLD", "eps"),
    "QFIM_pure": ("rho", "drho"),
    "QFIM_Kraus": ("rho0", "K", "dK", "LDtype", "exportLD", "eps"),
    "QFIM_Bloch": ("r", "dr", "eps"),
    "QFIM_Gauss": ("R", "dR", "D", "dD"),
    "Williamson_form": ("sigma",),
    # --- AsymptoticBound/AnalogCramerRao.py ---
    "HCRB": ("rho", "drho", "W", "eps"),
    "NHB": ("rho", "drho", "W"),
    # --- BayesianBound/BayesCramerRao.py ---
    "BCFIM": ("x", "p", "rho", "drho", "M", "eps"),
    "BQFIM": ("x", "p", "rho", "drho", "LDtype", "eps"),
    "BCRB": ("x", "p", "dp", "rho", "drho", "M", "b", "db", "btype", "eps"),
    "BQCRB": ("x", "p", "dp", "rho", "drho", "b", "db", "btype", "LDtype", "eps"),
    "VTB": ("x", "p", "dp", "rho", "drho", "M", "eps"),
    "QVTB": ("x", "p", "dp", "rho", "drho", "LDtype", "eps"),
    "OBB": ("x", "p", "dp", "rho", "drho", "d2rho", "LDtype", "eps"),
    # --- BayesianBound/ZivZakai.py ---
    "QZZB": ("x", "p", "rho", "eps"),
    # --- BayesianBound/BayesEstimation.py ---
    "Bayes": ("x", "p", "rho", "y", "M", "estimator", "savefile"),
    "MLE": ("x", "rho", "y", "M", "savefile"),
    "BayesCost": ("x", "p", "xest", "rho", "M", "W", "eps"),
    "BCB": ("x", "p", "rho", "W", "eps"),
    # --- Resource/Resource.py ---
    "SpinSqueezing": ("rho", "basis", "output"),
    "TargetTime": ("f", "tspan", "func", "args", "kwargs"),
    # --- Common/Common.py ---
    "SIC": ("dim",),
    "suN_generator": ("n",),
    "gramschmidt": ("A",),
    "BayesInput": ("x", "func", "dfunc", "channel"),
    "fidelity": ("rho1", "rho2"),
    "error_evaluation": (
        "scheme",
        "verbose",
        "objective",
        "input_error_scaling",
        "SLD_eps",
        "abstol",
        "reltol",
    ),
    "error_control": (
        "scheme",
        "objective",
        "output_error_scaling",
        "input_error_scaling",
        "SLD_eps",
        "max_episode",
    ),
    # --- Common state constructors ---
    "BellState": ("n",),
    "PlusState": (),
    "MinusState": (),
    # --- Parameterization/ControlWaveform.py ---
    "ZeroCTRL": (),
    "LinearCTRL": ("k", "c0"),
    "SineCTRL": ("A", "omega", "phi"),
    "SawCTRL": ("k", "n"),
    "TriangleCTRL": ("k", "n"),
    "GaussianCTRL": ("A", "mu", "sigma"),
    "GaussianEdgeCTRL": ("A", "sigma"),
    # --- Parameterization/NonDynamics.py ---
    "Kraus": ("rho0", "K", "dK"),
    # --- ControlOpt/ControlStruct.py ---
    "ControlOpt": ("savefile", "method", "kwargs"),
    "csv2npy_controls": ("controls", "num"),
    # --- StateOpt/StateStruct.py ---
    "StateOpt": ("savefile", "method", "kwargs"),
    "csv2npy_states": ("states", "num"),
    # --- MeasurementOpt/MeasurementStruct.py ---
    "MeasurementOpt": ("mtype", "minput", "savefile", "method", "kwargs"),
    "csv2npy_measurements": ("M", "num"),
    # --- ComprehensiveOpt/ComprehensiveStruct.py ---
    "ComprehensiveOpt": ("savefile", "method", "kwargs"),
    # --- Parameterization/GeneralDynamics.py ---
    "QubitDephasing": ("r", "para_est", "gamma", "tspan"),
    "Lindblad": ("tspan", "rho0", "H0", "dH", "decay", "Hc", "ctrl"),
    # --- AdaptiveScheme/Adapt.py ---
    "Adapt": ("x", "p", "rho0", "method", "savefile", "max_episode", "eps"),
    # --- AdaptiveScheme/Adapt_MZI.py ---
    "Adapt_MZI": ("x", "p", "rho0"),
}


@pytest.mark.parametrize("name,params", sorted(EXPECTED_FUNCTIONS.items()))
def test_function_signature(name, params):
    """Verify each paper-documented function has the expected parameter names."""
    func = globals().get(name)
    if func is None:
        pytest.skip(f"{name} not importable from quanestimation *")
    sig = inspect.signature(func)
    actual = tuple(sig.parameters.keys())
    assert actual == params, f"{name}: expected params {params}, got {actual}"


# ---- Method signatures (on objects returned by factory functions) ------


def test_control_opt_methods():
    """Verify ControlOpt-returned object method signatures."""
    c = ControlOpt(method="auto-GRAPE")

    sig = inspect.signature(c.dynamics)
    assert tuple(sig.parameters.keys()) == (
        "tspan",
        "rho0",
        "H0",
        "dH",
        "Hc",
        "decay",
        "ctrl_bound",
        "dyn_method",
    ), f"dynamics: {tuple(sig.parameters.keys())}"

    sig = inspect.signature(c.QFIM)
    assert tuple(sig.parameters.keys()) == ("W", "LDtype")

    sig = inspect.signature(c.CFIM)
    assert tuple(sig.parameters.keys()) == ("M", "W")

    sig = inspect.signature(c.HCRB)
    assert tuple(sig.parameters.keys()) == ("W",)

    sig = inspect.signature(c.mintime)
    assert tuple(sig.parameters.keys()) == (
        "f",
        "W",
        "M",
        "method",
        "target",
        "LDtype",
    )


def test_state_opt_methods():
    """Verify StateOpt-returned object method signatures."""
    s = StateOpt(method="AD")

    sig = inspect.signature(s.dynamics)
    assert tuple(sig.parameters.keys()) == (
        "tspan",
        "H0",
        "dH",
        "Hc",
        "ctrl",
        "decay",
        "dyn_method",
    ), f"dynamics: {tuple(sig.parameters.keys())}"

    sig = inspect.signature(s.Kraus)
    assert tuple(sig.parameters.keys()) == ("K", "dK")

    sig = inspect.signature(s.QFIM)
    assert tuple(sig.parameters.keys()) == ("W", "LDtype")

    sig = inspect.signature(s.CFIM)
    assert tuple(sig.parameters.keys()) == ("M", "W")

    sig = inspect.signature(s.HCRB)
    assert tuple(sig.parameters.keys()) == ("W",)


def test_lindblad_methods():
    """Verify Lindblad class method signatures."""
    import numpy as np

    rho0 = np.eye(2) / 2
    H0 = np.eye(2)
    dH = [np.eye(2)]
    tspan = np.linspace(0, 1, 10)
    dyn = Lindblad(tspan, rho0, H0, dH)

    sig = inspect.signature(dyn.expm)
    assert tuple(sig.parameters.keys()) == ()

    sig = inspect.signature(dyn.ode)
    assert tuple(sig.parameters.keys()) == ()


def test_measurement_opt_methods():
    """Verify MeasurementOpt-returned object method signatures."""
    import numpy as np

    m = MeasurementOpt()

    sig = inspect.signature(m.dynamics)
    assert tuple(sig.parameters.keys()) == (
        "tspan",
        "rho0",
        "H0",
        "dH",
        "Hc",
        "ctrl",
        "decay",
        "dyn_method",
    ), f"dynamics: {tuple(sig.parameters.keys())}"

    sig = inspect.signature(m.Kraus)
    assert tuple(sig.parameters.keys()) == ("rho0", "K", "dK")

    sig = inspect.signature(m.CFIM)
    assert tuple(sig.parameters.keys()) == ("W",)


def test_comprehensive_opt_methods():
    """Verify ComprehensiveOpt-returned object method signatures."""
    import numpy as np

    com = ComprehensiveOpt()

    sig = inspect.signature(com.dynamics)
    assert tuple(sig.parameters.keys()) == (
        "tspan",
        "H0",
        "dH",
        "Hc",
        "ctrl",
        "decay",
        "ctrl_bound",
        "dyn_method",
    ), f"dynamics: {tuple(sig.parameters.keys())}"

    sig = inspect.signature(com.Kraus)
    assert tuple(sig.parameters.keys()) == ("K", "dK")

    sig = inspect.signature(com.SM)
    assert tuple(sig.parameters.keys()) == ("W",)

    sig = inspect.signature(com.SC)
    assert tuple(sig.parameters.keys()) == ("W", "M", "target", "LDtype")

    sig = inspect.signature(com.CM)
    assert tuple(sig.parameters.keys()) == ("rho0", "W")

    sig = inspect.signature(com.SCM)
    assert tuple(sig.parameters.keys()) == ("W",)


def test_adapt_methods():
    """Verify Adapt class method signatures."""
    import numpy as np

    apt = Adapt([0.0, 1.0], [0.5, 0.5], np.eye(2) / 2)

    sig = inspect.signature(apt.dynamics)
    assert tuple(sig.parameters.keys()) == (
        "tspan",
        "H",
        "dH",
        "Hc",
        "ctrl",
        "decay",
        "dyn_method",
    ), f"dynamics: {tuple(sig.parameters.keys())}"

    sig = inspect.signature(apt.Kraus)
    assert tuple(sig.parameters.keys()) == ("K", "dK")

    sig = inspect.signature(apt.CFIM)
    assert tuple(sig.parameters.keys()) == ("M", "W")

    sig = inspect.signature(apt.Mopt)
    assert tuple(sig.parameters.keys()) == ("W",)


def test_adapt_mzi_methods():
    """Verify Adapt_MZI class method signatures."""
    import numpy as np

    apt = Adapt_MZI([0.0], [1.0], np.eye(2) / 2)

    sig = inspect.signature(apt.general)
    assert tuple(sig.parameters.keys()) == ()

    sig = inspect.signature(apt.online)
    assert tuple(sig.parameters.keys()) == ("target", "output")

    sig = inspect.signature(apt.offline)
    assert tuple(sig.parameters.keys()) == (
        "target",
        "method",
        "p_num",
        "deltaphi0",
        "c",
        "cr",
        "c0",
        "c1",
        "c2",
        "seed",
        "max_episode",
        "eps",
    )
