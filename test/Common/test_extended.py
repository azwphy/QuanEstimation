"""Tests for new Common interfaces (Phase 2)."""

import numpy as np
from quanestimation.base.Common.Common import (
    BellState,
    PlusState,
    MinusState,
    SigmaX,
    SigmaY,
    SigmaZ,
    sx,
    sy,
    sz,
)


class TestPauliConstants:
    """SigmaX/Y/Z and sx/sy/sz aliases."""

    def test_shapes(self):
        for m in [SigmaX, SigmaY, SigmaZ, sx, sy, sz]:
            assert m.shape == (2, 2)

    def test_hermitian(self):
        for m in [SigmaX, SigmaY, SigmaZ]:
            assert np.allclose(m, m.conj().T)

    def test_alias_identity(self):
        assert np.allclose(SigmaX, sx)
        assert np.allclose(SigmaY, sy)
        assert np.allclose(SigmaZ, sz)

    def test_anticommutation(self):
        assert np.allclose(
            SigmaX @ SigmaY + SigmaY @ SigmaX, np.zeros((2, 2)), atol=1e-14
        )

    def test_eigenvalues(self):
        for m in [SigmaX, SigmaY, SigmaZ]:
            eigvals = np.linalg.eigvalsh(m)
            assert np.allclose(np.sort(eigvals), [-1.0, 1.0])


class TestBellState:
    """BellState(n) for n=1,2,3,4."""

    def test_default(self):
        psi = BellState()
        assert len(psi) == 4
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_all_four(self):
        states = [BellState(n) for n in range(1, 5)]
        for i, psi in enumerate(states):
            assert np.isclose(np.linalg.norm(psi), 1.0)
        for i in range(4):
            for j in range(i + 1, 4):
                overlap = np.abs(np.dot(states[i].conj(), states[j]))
                assert np.isclose(overlap, 0.0, atol=1e-14)

    def test_invalid(self):
        import pytest

        with pytest.raises(ValueError):
            BellState(0)
        with pytest.raises(ValueError):
            BellState(5)


class TestPlusMinusState:
    """PlusState() and MinusState() — ket vectors."""

    def test_normalization(self):
        assert np.isclose(np.linalg.norm(PlusState()), 1.0)
        assert np.isclose(np.linalg.norm(MinusState()), 1.0)

    def test_orthogonal(self):
        assert np.isclose(
            np.abs(np.dot(PlusState().conj(), MinusState())), 0.0, atol=1e-14
        )

    def test_sigma_x_eigenstates(self):
        psi_p = PlusState()
        psi_m = MinusState()
        assert np.allclose(SigmaX @ psi_p, psi_p, atol=1e-14)
        assert np.allclose(SigmaX @ psi_m, -psi_m, atol=1e-14)
