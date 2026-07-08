"""Unified Julia -> Python conversion layer.

Single entry point for unwrapping juliacall return values.
The pyconvert rule in QuanEstimationBasePyExt.jl handles
Matrix{ComplexF64} natively; this function handles residuals,
nested containers, and legacy AnyValue objects.
"""

import numpy as np


def _unwrap(obj):
    """Recursively convert juliacall objects to native Python types.

    - numpy arrays -> pass-through (zero overhead)
    - juliacall ArrayValue (has __array_interface__) -> np.ndarray view
    - juliacall AnyValue (legacy, has ._jl) -> np.ndarray / float / int
    - Sequence-like objects (VectorValue, MatrixValue, list, tuple)
      -> recursed into native Python list

    Args:
        obj: Any value returned from a QJL.* or jl.* call.

    Returns:
        Native Python representation.
    """
    if isinstance(obj, np.ndarray):
        return obj

    if hasattr(obj, "__array_interface__"):
        return np.array(obj, copy=False)

    if hasattr(obj, "_jl"):
        raw = obj._jl
        try:
            nd = raw.ndims
            return np.array(raw)
        except Exception:
            pass
        try:
            return float(raw)
        except Exception:
            pass
        try:
            return int(raw)
        except Exception:
            pass
        return raw

    if isinstance(obj, (list, tuple)):
        return type(obj)(_unwrap(x) for x in obj)

    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return [_unwrap(x) for x in obj]

    return obj
