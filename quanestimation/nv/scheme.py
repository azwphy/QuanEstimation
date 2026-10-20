"""NV-center magnetometer scheme.

Delegates to Julia's NVMagnetometer.jl (340 lines of physics).
Python side is a thin wrapper — construct Julia scheme on demand,
delegate all computations.
"""

import numpy as np

import quanestimation.base as _base

_jl = None


def _get_jl():
    global _jl
    if _jl is None:
        _jl = _base.QJL
    return _jl


class NVMagnetometerScheme:
    """NV-center magnetometer estimation scheme.

    Provides a high-level interface for constructing NV-center
    estimation schemes and evaluating quantum/classical Fisher
    information, Holevo Cramer-Rao bound, control optimization,
    error evaluation, and error control.

    All defaults match Julia's ``NVMagnetometerScheme(; ...)``
    constructor exactly (``NVMagnetometer.jl:107-143``).

    Parameters
    ----------
    D : float
        Zero-field splitting in MHz. Default ``2π·2870``.
    gS : float
        Electron gyromagnetic ratio in MHz/mT. Default ``2π·28.03``.
    gI : float
        Nuclear gyromagnetic ratio in MHz/mT. Default ``2π·4.32·10⁻³``.
    A1 : float
        Transverse hyperfine coupling in MHz. Default ``2π·3.65``.
    A2 : float
        Longitudinal hyperfine coupling in MHz. Default ``2π·3.03``.
    B1 : float
        Magnetic field B_x in mT. Default 0.5.
    B2 : float
        Magnetic field B_y in mT. Default 0.5.
    B3 : float
        Magnetic field B_z in mT. Default 0.5.
    gamma : float
        Dephasing rate in MHz. Default ``2π``.
    decay_opt : list, optional
        Decay operators. Default ``[S₃]`` (Julia-computed).
    init_state : np.ndarray, optional
        Initial state vector. Default ``(|+1⟩+|−1⟩)/√2``.
    Hc : list, optional
        Control Hamiltonians. Default ``[S₁,S₂,S₃]`` (Julia-computed).
    ctrl : list or None, optional
        Control coefficient sequences. Default None (zeros).
    tspan : np.ndarray, optional
        Time span. Default ``0:0.01:2.0``.
    M : list or None, optional
        POVM measurement. Default None (SIC-POVM).
    """

    def __init__(
        self,
        D=None,
        gS=None,
        gI=None,
        A1=None,
        A2=None,
        B1=None,
        B2=None,
        B3=None,
        gamma=None,
        decay_opt=None,
        init_state=None,
        Hc=None,
        ctrl=None,
        tspan=None,
        M=None,
    ):
        jl = _get_jl()
        kwargs = {}
        for name, val in [
            ("D", D),
            ("gS", gS),
            ("gI", gI),
            ("A1", A1),
            ("A2", A2),
            ("B1", B1),
            ("B2", B2),
            ("B3", B3),
            ("γ", gamma),
        ]:
            if val is not None:
                kwargs[name] = val
        for name, val in [
            ("decay_opt", decay_opt),
            ("init_state", init_state),
            ("Hc", Hc),
            ("ctrl", ctrl),
            ("tspan", tspan),
            ("M", M),
        ]:
            if val is not None:
                kwargs[name] = val
        self._jl_scheme = jl.NVMagnetometerScheme(**kwargs)

    def QFIM(self, **kwargs):
        """Quantum Fisher information matrix.

        Delegates to ``QFIM(nv::NVMagnetometerScheme)``
        (``NVMagnetometer.jl:284``).
        """
        return _get_jl().QFIM(self._jl_scheme, **kwargs)

    def CFIM(self, **kwargs):
        """Classical Fisher information matrix.

        Delegates to ``CFIM(nv::NVMagnetometerScheme)``
        (``NVMagnetometer.jl:292``).
        """
        return _get_jl().CFIM(self._jl_scheme, **kwargs)

    def HCRB(self, **kwargs):
        """Holevo Cramer-Rao bound.

        Delegates to ``HCRB(nv::NVMagnetometerScheme)``
        (``NVMagnetometer.jl:300``).
        """
        return _get_jl().HCRB(self._jl_scheme, **kwargs)

    def optimize(self, opt, algorithm=None, objective=None, savefile=False):
        """Run control optimization.

        Converts the Python ``ControlOpt`` instance to a Julia
        ``ControlOpt`` struct before delegating to ``optimize!``.

        Args:
            opt: ``ControlOpt`` instance (Python).
            algorithm: Julia algorithm struct (default ``autoGRAPE()``).
            objective: Julia objective (default ``QFIM_obj()``).
            savefile: Whether to save results to file.
        """
        import juliacall
        from juliacall import Main as jl_main, convert as jlconvert

        jl_mod = _get_jl()
        if objective is None:
            objective = jl_mod.QFIM_obj()

        ctrl0 = getattr(opt, "ctrl0", [])
        if ctrl0:
            ctrl_jl = jlconvert(jl_main.Vector[jl_main.Vector[jl_main.Float64]], ctrl0)
        else:
            ctrl_jl = None

        ctrl_bound = getattr(opt, "ctrl_bound", [-float("inf"), float("inf")])
        ctrl_bound_jl = jlconvert(jl_main.Vector[jl_main.Float64], ctrl_bound)

        seed = getattr(opt, "seed", 1234)
        max_episode = getattr(opt, "max_episode", 100)

        julia_opt = jl_mod.ControlOpt(
            ctrl=ctrl_jl if ctrl0 else None,
            ctrl_bound=ctrl_bound_jl,
            seed=seed,
        )

        if algorithm is None:
            algorithm = jl_mod.autoGRAPE(max_episode=max_episode)

        return getattr(jl_mod, "optimize!")(
            self._jl_scheme,
            julia_opt,
            algorithm=algorithm,
            objective=objective,
            savefile=savefile,
        )

    def error_evaluation(self, **kwargs):
        """Evaluate estimation error.

        Delegates to ``error_evaluation(nv)``
        (``NVMagnetometer.jl:325``).
        """
        return _get_jl().error_evaluation(self._jl_scheme, **kwargs)

    def error_control(self, **kwargs):
        """Error control analysis.

        Delegates to ``error_control(nv)``
        (``NVMagnetometer.jl:334``).
        """
        return _get_jl().error_control(self._jl_scheme, **kwargs)
