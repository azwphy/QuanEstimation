"""Tests for 7 control waveform constructors (Phase 3)."""

import numpy as np
import pytest
from quanestimation.base.Parameterization.ControlWaveform import (
    ZeroCTRL,
    LinearCTRL,
    SineCTRL,
    SawCTRL,
    TriangleCTRL,
    GaussianCTRL,
    GaussianEdgeCTRL,
)


class TestZeroCTRL:
    def test_construct(self):
        z = ZeroCTRL()
        assert z is not None


class TestLinearCTRL:
    def test_default(self):
        c = LinearCTRL()
        assert c is not None

    def test_custom(self):
        c = LinearCTRL(k=2.0, c0=1.0)
        assert c is not None


class TestSineCTRL:
    def test_default(self):
        c = SineCTRL()
        assert c is not None

    def test_custom(self):
        c = SineCTRL(A=2.0, omega=3.0, phi=0.5)
        assert c is not None


class TestSawCTRL:
    def test_default(self):
        c = SawCTRL()
        assert c is not None

    def test_custom(self):
        c = SawCTRL(k=2.0, n=3.0)
        assert c is not None


class TestTriangleCTRL:
    def test_default(self):
        c = TriangleCTRL()
        assert c is not None

    def test_custom(self):
        c = TriangleCTRL(k=2.0, n=3.0)
        assert c is not None


class TestGaussianCTRL:
    def test_default(self):
        c = GaussianCTRL()
        assert c is not None

    def test_custom(self):
        c = GaussianCTRL(A=2.0, mu=0.5, sigma=0.1)
        assert c is not None


class TestGaussianEdgeCTRL:
    def test_default(self):
        c = GaussianEdgeCTRL()
        assert c is not None

    def test_custom(self):
        c = GaussianEdgeCTRL(A=2.0, sigma=0.5)
        assert c is not None
