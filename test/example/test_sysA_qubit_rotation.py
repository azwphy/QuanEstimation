import numpy as np
from quanestimation.base.AsymptoticBound.CramerRao import QFIM, QFIM_pure, SLD
from .analytic_reference import analytic_sysA


def test_sysA_qubit_rotation() -> None:
    omega = 1.5
    for t in [0.1, 0.5, 1.0, 2.0]:
        rho, drho, F_exact = analytic_sysA(t, omega)

        F_sld = QFIM(rho, drho, LDtype="SLD")
        assert np.allclose(F_sld, F_exact, atol=1e-10)

        F_pure = QFIM_pure(rho, drho)
        assert np.allclose(F_pure, F_exact, atol=1e-10)

        L = SLD(rho, drho, rep="original")
        if isinstance(L, list):
            L = L[0]
        assert np.linalg.norm(L - L.conj().T) < 1e-14
        assert np.allclose(
            2.0 * drho[0], L @ rho + rho @ L, atol=1e-10
        )

        rho_h = (rho + rho.conj().T) / 2.0
        assert np.linalg.norm(rho_h - rho) < 1e-14
