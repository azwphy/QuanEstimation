"""Control waveform constructors.

Thin wrappers around Julia struct constructors. Each function returns
a Julia control waveform struct that can be passed to
``Lindblad(ctrl=...)`` or ``ControlOpt(ctrl_type=...)``.

All parameter names and defaults match Julia's ``@kwdef struct``
definitions in ``LindbladData.jl:83-171`` exactly (including Unicode
Greek letters for field names).
"""

import quanestimation.base as _base

_jl = None


def _get_jl():
    global _jl
    if _jl is None:
        _jl = _base.QJL
    return _jl


def ZeroCTRL():
    """Zero control: ``c(t) = 0``."""
    return _get_jl().ZeroCTRL()


def LinearCTRL(k=1.0, c0=0.0):
    """Linear-in-time control: ``c(t) = k·t + c0``."""
    return _get_jl().LinearCTRL(k=k, c0=c0)


def SineCTRL(A=1.0, omega=1.0, phi=0.0):
    """Sinusoidal control: ``c(t) = A·sin(ω·t + φ)``.

    Args:
        A: Amplitude.
        omega: Angular frequency ω.
        phi: Phase offset φ.
    """
    return _get_jl().SineCTRL(**{"A": A, "ω": omega, "ϕ": phi})


def SawCTRL(k=1.0, n=1.0):
    """Sawtooth-wave control."""
    return _get_jl().SawCTRL(k=k, n=n)


def TriangleCTRL(k=1.0, n=1.0):
    """Triangle-wave control."""
    return _get_jl().TriangleCTRL(k=k, n=n)


def GaussianCTRL(A=1.0, mu=0.0, sigma=1.0):
    """Gaussian-pulse control: ``c(t) = A·exp(-(t-μ)²/(2σ))``.

    Args:
        A: Amplitude.
        mu: Center μ.
        sigma: Width σ (variance).
    """
    return _get_jl().GaussianCTRL(**{"A": A, "μ": mu, "σ": sigma})


def GaussianEdgeCTRL(A=1.0, sigma=1.0):
    """Gaussian-edge control.

    Args:
        A: Amplitude.
        sigma: Width σ.
    """
    return _get_jl().GaussianEdgeCTRL(**{"A": A, "σ": sigma})
