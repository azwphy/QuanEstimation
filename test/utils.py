import numpy as np


def rand_rho(N: int) -> np.ndarray:
    A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    return rho


def rand_drho(N: int) -> np.ndarray:
    A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    drho = (A + A.conj().T) / 2.0
    drho -= (np.trace(drho) / N) * np.eye(N, dtype=np.complex128)
    return drho


def rand_unitary(N: int) -> np.ndarray:
    A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    Q, _ = np.linalg.qr(A)
    return Q
