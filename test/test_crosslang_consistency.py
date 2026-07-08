import numpy as np
import random
from .example.analytic_reference import analytic_sysA, analytic_sysC
from quanestimation.base.AsymptoticBound.CramerRao import QFIM, SLD

random.seed(1234)
np.random.seed(1234)


def test_crosslang_sysA_consistency() -> None:
    """Sys-A: Single-qubit rotation — QFIM vs analytical t²."""
    omega = 1.5
    for t in [0.1, 0.5, 1.0, 2.0]:
        rho, drho, F_exact = analytic_sysA(t, omega)
        F_py = QFIM(rho, drho, LDtype="SLD")
        assert np.allclose(F_py, F_exact, atol=1e-10)


def test_crosslang_sysC_consistency() -> None:
    """Sys-C: Single-qubit pure dephasing — QFIM vs analytical t²e^{-4γt}."""
    omega = 1.0
    for gamma in [0.01, 0.1, 0.5]:
        for t in [0.1, 0.5, 1.0]:
            rho, drho, F_exact = analytic_sysC(t, omega, gamma)
            F_py = QFIM(rho, drho, LDtype="SLD")
            assert np.allclose(F_py, F_exact, atol=1e-10)


def test_crosslang_hermiticity() -> None:
    """SLD Hermiticity — Python vs analytical for pure state."""
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    from scipy.linalg import expm

    psi0 = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    H = sz / 2.0
    t = 0.5
    U = expm(-1j * H * t)
    psit = U @ psi0
    rho = np.outer(psit, psit.conj())
    dpsit = -1j * H @ psit
    drho = np.outer(dpsit, psit.conj()) + np.outer(psit, dpsit.conj())

    L = SLD(rho, [drho], rep="original")
    if isinstance(L, list):
        L = L[0]
    L_pure = 2.0 * drho
    assert np.linalg.norm(L - L_pure) < 1e-12
    assert np.linalg.norm(L - L.conj().T) < 1e-14
