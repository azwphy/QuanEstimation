import numpy as np
from quanestimation.base.AsymptoticBound.CramerRao import SLD, RLD, LLD, QFIM

from .utils import rand_rho, rand_drho


def test_operator_definitions() -> None:
    """Verify SLD/RLD/LLD satisfy their definition equations for random matrices."""
    for N in range(2, 6):
        for _ in range(10):
            rho = rand_rho(N)
            drho = rand_drho(N)

            # SLD: 2 * drho = L @ rho + rho @ L  and L must be Hermitian
            for rep in ("original", "eigen"):
                L = SLD(rho, [drho], rep=rep)
                if isinstance(L, list):
                    L = L[0]
                assert np.linalg.norm(L - L.conj().T) < 1e-11

            L_orig = SLD(rho, [drho], rep="original")
            if isinstance(L_orig, list):
                L_orig = L_orig[0]
            assert np.allclose(2.0 * drho, L_orig @ rho + rho @ L_orig, atol=1e-10)

            # RLD: drho = rho @ R
            R_orig = RLD(rho, [drho], rep="original")
            if isinstance(R_orig, list):
                R_orig = R_orig[0]
            assert np.allclose(drho, rho @ R_orig, atol=1e-10)

            # LLD: drho = L @ rho
            L_LLD = LLD(rho, [drho], rep="original")
            if isinstance(L_LLD, list):
                L_LLD = L_LLD[0]
            assert np.allclose(drho, L_LLD @ rho, atol=1e-10)

            # QFI cross-consistency: F_RLD ≈ F_LLD
            F_RLD = QFIM(rho, [drho], LDtype="RLD")
            F_LLD = QFIM(rho, [drho], LDtype="LLD")
            assert np.allclose(F_RLD, F_LLD, atol=1e-10)
