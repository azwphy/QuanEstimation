import numpy as np


def analytic_sysA(t: float, omega: float):
    re = np.cos(omega * t)
    im = -np.sin(omega * t)
    e_pow = np.exp(-1j * omega * t)
    rho = 0.5 * np.array([[1.0, e_pow], [np.conj(e_pow), 1.0]], dtype=np.complex128)
    drho = 0.5 * np.array(
        [[0.0, -1j * t * e_pow], [1j * t * np.conj(e_pow), 0.0]], dtype=np.complex128
    )
    F_exact = t**2
    return rho, [drho], F_exact


def analytic_sysB(t: float, omega: float, gamma_minus: float):
    r1 = np.exp(-gamma_minus * t / 2) * np.cos(omega * t)
    r2 = -np.exp(-gamma_minus * t / 2) * np.sin(omega * t)
    r3 = 1 - np.exp(-gamma_minus * t)
    rho = 0.5 * np.array(
        [[1 + r3, r1 - 1j * r2], [r1 + 1j * r2, 1 - r3]], dtype=np.complex128
    )
    dr1 = -t * np.exp(-gamma_minus * t / 2) * np.sin(omega * t)
    dr2 = -t * np.exp(-gamma_minus * t / 2) * np.cos(omega * t)
    dr3 = 0.0
    drho = 0.5 * np.array(
        [[dr3, dr1 - 1j * dr2], [dr1 + 1j * dr2, -dr3]], dtype=np.complex128
    )
    R = np.sqrt(r1**2 + r2**2 + r3**2)
    eigvals = np.array([(1.0 + R) / 2.0, (1.0 - R) / 2.0])
    if R > 1e-15:
        n1 = np.array([r1 - 1j * r2, R - r3], dtype=np.complex128)
        n1 /= np.linalg.norm(n1)
        n2 = np.array([r1 - 1j * r2, -R - r3], dtype=np.complex128)
        n2 /= np.linalg.norm(n2)
        eigvecs = [n1, n2]
    else:
        eigvecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    F_exact = 0.0
    for i in range(2):
        for j in range(2):
            denom = eigvals[i] + eigvals[j]
            if denom <= 1e-15:
                continue
            mat_elem = eigvecs[i].conj() @ drho @ eigvecs[j]
            F_exact += 2 * np.abs(mat_elem) ** 2 / denom
    return rho, [drho], np.real(F_exact)


def analytic_sysC(t: float, omega: float, gamma: float):
    e_pow = np.exp(-1j * omega * t - 2 * gamma * t)
    rho = 0.5 * np.array([[1.0, e_pow], [np.conj(e_pow), 1.0]], dtype=np.complex128)
    d01 = -0.5 * 1j * t * np.exp(-1j * omega * t - 2 * gamma * t)
    drho = np.array([[0.0, d01], [np.conj(d01), 0.0]], dtype=np.complex128)
    F_exact = t**2 * np.exp(-4 * gamma * t)
    return rho, [drho], F_exact


def analytic_sysD(t: float, omega1: float, omega2: float, g: float):
    Omega = np.sqrt((omega1 + omega2) ** 2 + g**2)
    ct = np.cos(Omega * t)
    st = np.sin(Omega * t)
    alpha = (ct - 1j * (omega1 + omega2 + g) / Omega * st) / np.sqrt(2)
    beta = (ct + 1j * (omega1 + omega2 - g) / Omega * st) / np.sqrt(2)
    rho = np.zeros((4, 4), dtype=np.complex128)
    rho[0, 0] = np.abs(alpha) ** 2
    rho[0, 3] = alpha * np.conj(beta)
    rho[3, 0] = np.conj(rho[0, 3])
    rho[3, 3] = np.abs(beta) ** 2
    dOmega_dw1 = (omega1 + omega2) / Omega
    dct = -st * Omega * t * dOmega_dw1
    dst = ct * Omega * t * dOmega_dw1
    dalpha = (
        dct
        - 1j
        * (
            (1 / Omega - (omega1 + omega2 + g) / Omega**2 * dOmega_dw1) * st
            + (omega1 + omega2 + g) / Omega * dst
        )
    ) / np.sqrt(2)
    dbeta = (
        dct
        + 1j
        * (
            (1 / Omega - (omega1 + omega2 - g) / Omega**2 * dOmega_dw1) * st
            + (omega1 + omega2 - g) / Omega * dst
        )
    ) / np.sqrt(2)
    drho1 = np.zeros((4, 4), dtype=np.complex128)
    drho1[0, 0] = dalpha * np.conj(alpha) + alpha * np.conj(dalpha)
    drho1[0, 3] = dalpha * np.conj(beta) + alpha * np.conj(dbeta)
    drho1[3, 0] = np.conj(drho1[0, 3])
    drho1[3, 3] = dbeta * np.conj(beta) + beta * np.conj(dbeta)
    psi = np.array([alpha, 0.0, 0.0, beta], dtype=np.complex128)
    dpsi_w1 = np.array([dalpha, 0.0, 0.0, dbeta], dtype=np.complex128)
    F11 = 4 * np.real(dpsi_w1.conj() @ dpsi_w1 - np.abs(dpsi_w1.conj() @ psi) ** 2)
    dOmega_dw2 = (omega1 + omega2) / Omega
    dalpha2 = (
        dct
        - 1j
        * (
            (1 / Omega - (omega1 + omega2 + g) / Omega**2 * dOmega_dw2) * st
            + (omega1 + omega2 + g) / Omega * dst
        )
    ) / np.sqrt(2)
    dbeta2 = (
        dct
        + 1j
        * (
            (1 / Omega - (omega1 + omega2 - g) / Omega**2 * dOmega_dw2) * st
            + (omega1 + omega2 - g) / Omega * dst
        )
    ) / np.sqrt(2)
    drho2 = np.zeros((4, 4), dtype=np.complex128)
    drho2[0, 0] = dalpha2 * np.conj(alpha) + alpha * np.conj(dalpha2)
    drho2[0, 3] = dalpha2 * np.conj(beta) + alpha * np.conj(dbeta2)
    drho2[3, 0] = np.conj(drho2[0, 3])
    drho2[3, 3] = dbeta2 * np.conj(beta) + beta * np.conj(dbeta2)
    dpsi_w2 = np.array([dalpha2, 0.0, 0.0, dbeta2], dtype=np.complex128)
    F22 = 4 * np.real(dpsi_w2.conj() @ dpsi_w2 - np.abs(dpsi_w2.conj() @ psi) ** 2)
    F12 = 4 * np.real(
        dpsi_w1.conj() @ dpsi_w2
        - (dpsi_w1.conj() @ psi) * np.conj(dpsi_w2.conj() @ psi)
    )
    F_exact = np.array([[F11, F12], [F12, F22]], dtype=np.float64)
    return rho, [drho1, drho2], F_exact
