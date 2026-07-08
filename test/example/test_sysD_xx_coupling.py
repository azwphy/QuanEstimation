import numpy as np
from quanestimation.base.AsymptoticBound.CramerRao import QFIM, QFIM_pure, SLD
from .analytic_reference import analytic_sysD


def test_sysD_xx_coupling() -> None:
    omega1, omega2, g = 1.0, 1.0, 0.1
    for t in [0.5, 1.0]:
        rho, drho, F_exact = analytic_sysD(t, omega1, omega2, g)

        F_pure = QFIM_pure(rho, drho)
        assert np.allclose(F_pure, F_exact, atol=1e-10)

        F_sld = QFIM(rho, drho, LDtype="SLD")
        assert np.allclose(F_sld, F_exact, atol=1e-10)

        Ls = SLD(rho, drho, rep="original")
        for L in Ls:
            assert np.linalg.norm(L - L.conj().T) < 1e-14

        rho_h = (rho + rho.conj().T) / 2.0
        F_h = QFIM(rho_h, drho, LDtype="SLD")
        assert np.allclose(F_h, F_sld, atol=1e-12)
