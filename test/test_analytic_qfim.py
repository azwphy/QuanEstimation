import numpy as np
import random
from quanestimation.base.AsymptoticBound.CramerRao import QFIM, SLD

from .utils import rand_rho, rand_drho, rand_unitary

random.seed(1234)
np.random.seed(1234)


# ===== §0.4-A: Single-qubit pure state QFIM (Eq. 574-576) =====
def test_analytic_A_pure_state_qfim() -> None:
    for _ in range(5):
        theta = random.random() * np.pi
        phi = random.random() * 2 * np.pi
        psi = np.array([np.cos(theta), np.sin(theta) * np.exp(1j * phi)])
        rho = np.outer(psi, psi.conj())
        dtheta_psi = np.array(
            [-np.sin(theta), np.cos(theta) * np.exp(1j * phi)]
        )
        dphi_psi = np.array([0.0, 1j * np.sin(theta) * np.exp(1j * phi)])
        dtheta_rho = np.outer(dtheta_psi, psi.conj()) + np.outer(
            psi, dtheta_psi.conj()
        )
        dphi_rho = np.outer(dphi_psi, psi.conj()) + np.outer(
            psi, dphi_psi.conj()
        )
        F = QFIM(rho, [dtheta_rho, dphi_rho], LDtype="SLD")
        F_expected = np.array([[4.0, 0.0], [0.0, np.sin(2 * theta) ** 2]])
        assert np.allclose(F, F_expected, atol=1e-10)


# ===== §0.4-B: Single-qubit mixed state QFIM (Eq. 602-605) =====
def test_analytic_B_mixed_state_qfim() -> None:
    for _ in range(5):
        rho = rand_rho(2)
        drho1 = rand_drho(2)
        drho2 = rand_drho(2)
        F_num = QFIM(rho, [drho1, drho2], LDtype="SLD")
        F_exact = np.zeros((2, 2))
        pairs = [
            (drho1, drho1, 0, 0),
            (drho1, drho2, 0, 1),
            (drho2, drho1, 1, 0),
            (drho2, drho2, 1, 1),
        ]
        det_rho = np.linalg.det(rho)
        for da, db, i, j in pairs:
            F_exact[i, j] = np.real(
                np.trace(da @ db)
                + np.trace(rho @ da @ rho @ db) / det_rho
            )
        assert np.allclose(F_num, F_exact, atol=1e-8)


# ===== §0.4-C: Commuting generators — covariance formula (Eq. 553-564) =====
def test_analytic_C_commuting_generators() -> None:
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    H1 = np.kron(sz, I2)
    H2 = np.kron(I2, sz)
    psi0 = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2)

    for t in [0.1, 0.5, 1.0, 2.0]:
        from scipy.linalg import expm

        U = expm(-1j * (H1 + H2) * t)
        psit = U @ psi0
        rho = np.outer(psit, psit.conj())
        d1rho = -1j * t * (H1 @ rho - rho @ H1)
        d2rho = -1j * t * (H2 @ rho - rho @ H2)
        F = QFIM(rho, [d1rho, d2rho], LDtype="SLD")
        F_expected = 4 * t**2 * np.array([[1.0, 1.0], [1.0, 1.0]])
        assert np.allclose(F, F_expected, atol=1e-10)


# ===== §0.4-D: Dephasing qubit dual-parameter QFIM (Eq. 617-635) =====
def test_analytic_D_dephasing_dual() -> None:
    for gamma in [0.01, 0.1, 0.5]:
        for t in [0.1, 0.5, 1.0]:
            B = 1.0
            rho01_0 = 0.5
            rho01_t = rho01_0 * np.exp(-2j * B * t - gamma * t)
            rho = np.array(
                [[0.5, rho01_t], [np.conj(rho01_t), 0.5]], dtype=np.complex128
            )

            dB_rho01 = -1j * t * np.exp(-2j * B * t - gamma * t)
            dB_rho = np.array(
                [[0.0, dB_rho01], [np.conj(dB_rho01), 0.0]], dtype=np.complex128
            )

            dg_rho01 = -t / 2.0 * np.exp(-2j * B * t - gamma * t)
            dg_rho = np.array(
                [[0.0, dg_rho01], [np.conj(dg_rho01), 0.0]], dtype=np.complex128
            )

            F_num = QFIM(rho, [dB_rho, dg_rho], LDtype="SLD")
            F_BB_exact = (
                16 * np.abs(rho01_0) ** 2 * np.exp(-2 * gamma * t) * t**2
            )
            F_gg_exact = t**2 / (np.exp(2 * gamma * t) - 1)

            assert np.allclose(F_num[0, 0], F_BB_exact, atol=1e-8)
            assert np.allclose(F_num[1, 1], F_gg_exact, atol=1e-8)
            assert abs(F_num[0, 1]) < 1e-10
            assert abs(F_num[1, 0]) < 1e-10


# ===== §0.4-E: Thermal-state temperature QFI (Eq. 1210) =====
def test_analytic_E_thermal_qfi() -> None:
    omega = 1.0
    for T in [0.5, 1.0, 2.0, 5.0]:
        beta = 1.0 / T
        Z = 2.0 * np.cosh(beta * omega)
        rho = np.diag(
            [np.exp(-beta * omega) / Z, np.exp(beta * omega) / Z]
        ).astype(np.complex128)
        drho00_dbeta = -omega / (2.0 * np.cosh(beta * omega) ** 2)
        drho11_dbeta = omega / (2.0 * np.cosh(beta * omega) ** 2)
        drho_dbeta = np.diag([drho00_dbeta, drho11_dbeta]).astype(np.complex128)
        drho_dT = drho_dbeta * (-(beta**2))

        F_code = QFIM(rho, [drho_dT], LDtype="SLD")
        F_exact = omega**2 / (T**4 * np.cosh(beta * omega) ** 2)
        assert np.allclose(F_code, F_exact, atol=1e-8)


# ===== §0.4-F: Pure state SLD explicit form (Eq. 543-544) =====
def test_analytic_F_pure_sld_explicit() -> None:
    from scipy.linalg import expm

    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    psi0 = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    H = sz / 2.0
    t_val = 0.5
    U = expm(-1j * H * t_val)
    psit = U @ psi0
    rho = np.outer(psit, psit.conj())
    dpsit = -1j * H @ psit
    drho = np.outer(dpsit, psit.conj()) + np.outer(psit, dpsit.conj())

    L_code = SLD(rho, [drho], rep="original")
    if isinstance(L_code, list):
        L_code = L_code[0]
    L_pure = 2.0 * drho
    L_explicit = 2.0 * (
        np.outer(psit, dpsit.conj()) + np.outer(dpsit, psit.conj())
    )

    assert np.linalg.norm(L_code - L_pure) < 1e-12
    assert np.linalg.norm(L_code - L_explicit) < 1e-12


# ===== §0.4-G: QFIM mathematical properties =====
def test_analytic_G_unitary_invariance() -> None:
    for _ in range(5):
        N = 2
        rho = rand_rho(N)
        drho = rand_drho(N)
        U = rand_unitary(N)
        rhoU = U @ rho @ U.conj().T
        drhoU = U @ drho @ U.conj().T
        F = QFIM(rho, [drho], LDtype="SLD")
        FU = QFIM(rhoU, [drhoU], LDtype="SLD")
        assert np.allclose(F, FU, atol=1e-12)


def test_analytic_G_reparametrization() -> None:
    omega = 1.0
    gamma_val = 0.1
    t_val = 0.5
    rho01 = 0.5 * np.exp(-1j * omega * t_val - 2 * gamma_val * t_val)
    rho_t = np.array(
        [[0.5, rho01], [np.conj(rho01), 0.5]], dtype=np.complex128
    )
    dw_rho01 = -1j * t_val * 0.5 * np.exp(
        -1j * omega * t_val - 2 * gamma_val * t_val
    )
    dw_rho = np.array(
        [[0.0, dw_rho01], [np.conj(dw_rho01), 0.0]], dtype=np.complex128
    )
    F_omega = QFIM(rho_t, [dw_rho], LDtype="SLD")
    deta_rho = dw_rho / 2.0
    F_eta = QFIM(rho_t, [deta_rho], LDtype="SLD")
    assert np.allclose(F_eta, F_omega / 4.0, atol=1e-10)


def test_analytic_G_direct_sum() -> None:
    for _ in range(3):
        rho1 = rand_rho(2)
        drho1 = rand_drho(2)
        rho2 = rand_rho(3)
        drho2 = rand_drho(3)
        rho_ds = np.block(
            [[rho1, np.zeros((2, 3), dtype=np.complex128)],
             [np.zeros((3, 2), dtype=np.complex128), rho2]]
        )
        drho_ds = np.block(
            [[drho1, np.zeros((2, 3), dtype=np.complex128)],
             [np.zeros((3, 2), dtype=np.complex128), drho2]]
        )
        F1 = QFIM(rho1, [drho1], LDtype="SLD")
        F2 = QFIM(rho2, [drho2], LDtype="SLD")
        F_ds = QFIM(rho_ds, [drho_ds], LDtype="SLD")
        assert np.allclose(F_ds, F1 + F2, atol=1e-12)


def test_analytic_G_convexity() -> None:
    for _ in range(10):
        rho1 = rand_rho(2)
        drho1 = rand_drho(2)
        rho2 = rand_rho(2)
        drho2 = rand_drho(2)
        lam = random.random()
        rho_mix = lam * rho1 + (1 - lam) * rho2
        drho_mix = lam * drho1 + (1 - lam) * drho2
        F_mix = QFIM(rho_mix, [drho_mix], LDtype="SLD")
        F_bound = lam * QFIM(rho1, [drho1], LDtype="SLD") + (
            1 - lam
        ) * QFIM(rho2, [drho2], LDtype="SLD")
        assert F_mix <= F_bound + 1e-8


def test_analytic_G_rld_ge_sld() -> None:
    for _ in range(10):
        N = random.randint(2, 5)
        rho = rand_rho(N)
        drho = rand_drho(N)
        F_s = QFIM(rho, [drho], LDtype="SLD")
        F_r = QFIM(rho, [drho], LDtype="RLD")
        assert F_r >= F_s - 1e-10
