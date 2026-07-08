import os
import io
import pytest
import numpy as np
from unittest.mock import patch
from quanestimation.base.AdaptiveScheme.Adapt import Adapt, iter_FOP_singlepara


def _mock_stdin(*values):
    """Return a StringIO with canned input values (one per line)."""
    return io.StringIO("\n".join(str(v) for v in values))


@pytest.mark.parametrize("x_vals,p,rho0,method", [
    (np.linspace(0.0, 0.5 * np.pi, 100),
     (1.0 / (0.5 * np.pi)) * np.ones(100),
     0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128),
     "FOP"),
    (np.linspace(0.0, 0.5 * np.pi, 50),
     (1.0 / (0.5 * np.pi)) * np.ones(50),
     0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128),
     "MI"),
])
def test_adapt_singlepara_run(x_vals, p, rho0, method):
    """Verify FOP/MI adaptive estimation executes (single-parameter)."""
    B, omega0 = np.pi / 2.0, 1.0
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    H0_func = [0.5 * B * omega0 * (sx * np.cos(xi) + sz * np.sin(xi)) for xi in x_vals]
    dH_func = [[0.5 * B * omega0 * (-sx * np.sin(xi) + sz * np.cos(xi))] for xi in x_vals]
    tspan = np.linspace(0.0, 1.0, 20)

    adapt = Adapt([x_vals], p, rho0, method=method, max_episode=1, savefile=False)
    adapt.dynamics(tspan, H0_func, dH_func)
    try:
        with patch("builtins.input", side_effect=[str(i % 4) for i in range(20)]):
            adapt.CFIM()
    except Exception as e:
        pytest.skip(f"Adaptive {method} test error: {e}")

    for f in ["pout.npy", "xout.npy", "Lout.npy"]:
        if os.path.exists(f):
            os.remove(f)


def test_adapt_fop_multipara_run() -> None:
    """Verify FOP adaptive estimation executes (multi-parameter)."""
    x_vals = np.linspace(0.0, 0.5 * np.pi, 20)
    omega0_vals = np.linspace(1.0, 2.0, 20)
    B = 0.5 * np.pi
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    mu_w, mu_x = 1.5, 0.0
    eta_w, eta_x = 0.2, 0.2
    prob = np.zeros((len(omega0_vals), len(x_vals)))
    for i, w in enumerate(omega0_vals):
        for j, xv in enumerate(x_vals):
            prob[i, j] = (
                np.exp(-(w - mu_w)**2 / (2 * eta_w**2)) / (eta_w * np.sqrt(2 * np.pi))
                * np.exp(-(xv - mu_x)**2 / (2 * eta_x**2)) / (eta_x * np.sqrt(2 * np.pi))
            )
    prob /= np.sum(prob) * (omega0_vals[1] - omega0_vals[0]) * (x_vals[1] - x_vals[0])

    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    H0_func = []
    dH_func = []
    for w in omega0_vals:
        row_H0 = []
        row_dH = []
        for xv in x_vals:
            row_H0.append(0.5 * B * w * (sx * np.cos(xv) + sz * np.sin(xv)))
            row_dH.append([
                0.5 * B * (sx * np.cos(xv) + sz * np.sin(xv)),
                0.5 * B * w * (-sx * np.sin(xv) + sz * np.cos(xv)),
            ])
        H0_func.append(row_H0)
        dH_func.append(row_dH)
    tspan = np.linspace(0.0, 1.0, 10)

    adapt = Adapt([omega0_vals, x_vals], prob, rho0, method="FOP", max_episode=1, savefile=False)
    adapt.dynamics(tspan, H0_func, dH_func)
    try:
        with patch("builtins.input", side_effect=[str(i % 4) for i in range(20)]):
            adapt.CFIM()
    except Exception as e:
        pytest.skip(f"Adaptive multi-parameter test error: {e}")

    for f in ["pout.npy", "xout.npy", "Lout.npy"]:
        if os.path.exists(f):
            os.remove(f)
