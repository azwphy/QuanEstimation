import pytest
import numpy as np


@pytest.fixture
def rho_2d():
    A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    rho = A @ A.conj().T
    return rho / np.trace(rho)


@pytest.fixture
def rho_4d():
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    rho = A @ A.conj().T
    return rho / np.trace(rho)


@pytest.fixture
def drho_2d():
    A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    drho = (A + A.conj().T) / 2.0
    drho -= (np.trace(drho) / 2) * np.eye(2, dtype=np.complex128)
    return drho


@pytest.fixture
def drho_4d():
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    drho = (A + A.conj().T) / 2.0
    drho -= (np.trace(drho) / 4) * np.eye(4, dtype=np.complex128)
    return drho


@pytest.fixture
def rand_unitary_2d():
    A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    Q, _ = np.linalg.qr(A)
    return Q


@pytest.fixture
def rand_unitary_4d():
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    Q, _ = np.linalg.qr(A)
    return Q
