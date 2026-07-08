import numpy as np
from quanestimation.base.AsymptoticBound.CramerRao import QFIM
from .analytic_reference import analytic_sysC


def test_sysC_pure_dephasing() -> None:
    omega = 1.0
    for gamma in [0.01, 0.1, 0.5]:
        for t in [0.1, 0.5, 1.0]:
            rho, drho, F_exact = analytic_sysC(t, omega, gamma)
            F_sld = QFIM(rho, drho, LDtype="SLD")
            assert np.allclose(F_sld, F_exact, atol=1e-10)

            F_rld = QFIM(rho, drho, LDtype="RLD")
            F_lld = QFIM(rho, drho, LDtype="LLD")
            assert np.allclose(F_rld, F_lld, atol=1e-10)

    gamma = 0.0
    for t in [0.1, 0.5, 1.0]:
        rho, drho, F_exact = analytic_sysC(t, omega, gamma)
        F_sld = QFIM(rho, drho, LDtype="SLD")
        assert np.allclose(F_sld, F_exact, atol=1e-10)
