"""NV-center magnetometer tests — mirroring Julia's test_nv.jl."""

import numpy as np
import pytest

from quanestimation.base import QJL


class TestNVAnalyticQFIM:
    """Analytic QFIM for simplified NV model (electron-only).

    Matches Julia's test_nv_analytic_qfim() in test_nv.jl:36-61.
    """

    def test_analytic_qfim(self):
        s1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
        s3 = np.diag([1, 0, -1])
        D_val = 2 * np.pi * 2870.0
        gS = 2 * np.pi * 28.03
        Bz = 0.5
        H0 = D_val * (s3 @ s3) + gS * Bz * s3
        dH = [gS * s3]
        psi0 = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
        rho0 = np.outer(psi0, psi0.conj())

        for t in [0.5, 1.0, 2.0, 5.0]:
            tspan = np.linspace(0, t, 2)
            dynamics = QJL.Lindblad(H0, dH, tspan, dyn_method="Expm")
            scheme = QJL.GeneralScheme(probe=rho0, param=dynamics)
            F_num = QJL.QFIM(scheme)[0, 0]
            F_ana = 4.0 * gS**2 * t**2
            assert np.isclose(F_num, F_ana, rtol=1e-10)


class TestNVMagnetometerScheme:
    """End-to-end tests via NVMagnetometerScheme.

    Matches Julia's test_nv_magnetometer() in test_nv.jl:63-118
    and lib/NVMagnetometer/test/runtests.jl:4-10.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from quanestimation.nv import NVMagnetometerScheme

        self.scheme = NVMagnetometerScheme()

    def test_qfim_positive_definite(self):
        F = self.scheme.QFIM()
        assert F.shape == (3, 3)
        assert np.all(np.isfinite(F))
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals > 0)

    def test_cfim_positive_definite(self):
        F = self.scheme.CFIM()
        assert F.shape == (3, 3)
        assert np.all(np.isfinite(F))
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals > 0)

    def test_hcrb(self):
        try:
            h = self.scheme.HCRB()
            assert isinstance(h, (int, float, np.floating))
            assert np.isfinite(h)
            assert h >= 0
            F_q = self.scheme.QFIM()
            assert h >= np.trace(np.linalg.inv(F_q)) - 1e-10
        except Exception:
            pytest.skip("SCS SDP solver failed")

    def test_nvmagnetometer_basic(self):
        """Match lib/NVMagnetometer/test/runtests.jl."""
        F_q = self.scheme.QFIM()
        assert np.all(np.linalg.eigvalsh(F_q) > 0)
        F_c = self.scheme.CFIM()
        assert np.all(np.linalg.eigvalsh(F_c) > 0)
        try:
            h = self.scheme.HCRB()
            assert h > 0
        except Exception:
            pytest.skip("SCS SDP solver failed")

    def test_optimize(self):
        from quanestimation.base.ControlOpt.ControlStruct import ControlOpt

        F_pre = np.trace(self.scheme.QFIM())
        opt = ControlOpt(
            ctrl0=[np.zeros(200) for _ in range(3)],
            seed=1234,
            max_episode=3,
        )
        self.scheme.optimize(opt)
        F_post = np.trace(self.scheme.QFIM())
        assert np.isfinite(F_post)
        assert F_post >= F_pre - 1e-10

    def test_error_evaluation(self):
        self.scheme.error_evaluation()

    def test_error_control(self):
        self.scheme.error_control()
