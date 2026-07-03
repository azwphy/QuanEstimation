import numpy as np
from quanestimation.base.AsymptoticBound.CramerRao import QFIM, RLD, LLD
from .analytic_reference import analytic_sysB


def test_sysB_spontaneous_emission() -> None:
    omega = 1.0
    gamma_minus = 0.1
    for t in [0.01, 0.1, 0.5, 1.0]:
        rho, drho, F_exact = analytic_sysB(t, omega, gamma_minus)

        F_sld = QFIM(rho, drho, LDtype="SLD")
        assert np.allclose(F_sld, F_exact, atol=1e-8)

        rho_h = (rho + rho.conj().T) / 2.0
        F_h = QFIM(rho_h, drho, LDtype="SLD")
        assert np.allclose(F_h, F_sld, atol=1e-12)

        R = RLD(rho, drho, rep="original")
        if isinstance(R, list):
            R = R[0]
        assert np.allclose(rho @ R, drho[0], atol=1e-8)

        L = LLD(rho, drho, rep="original")
        if isinstance(L, list):
            L = L[0]
        assert np.allclose(L @ rho, drho[0], atol=1e-8)

        F_rld = QFIM(rho, drho, LDtype="RLD")
        F_lld = QFIM(rho, drho, LDtype="LLD")
        assert np.allclose(F_rld, F_lld, atol=1e-10)
