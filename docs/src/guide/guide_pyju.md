# Python-Julia calling patterns

This page documents the internal bridge between Python and Julia. Most users
do **not** need this — the standard `from quanestimation import *` handles
everything automatically. This reference is for advanced use cases that require
direct interaction with the Julia runtime.

## How the bridge works

Python is a thin wrapper over Julia. When you call `Lindblad(...)` or
`QFIM(...)` in Python, the call is transparently forwarded to Julia. Most
API calls work exactly as you'd expect — you pass Python data (numpy arrays,
lists, strings) and get results back.

## Direct Julia access

The `QJL` singleton provides a handle to the Julia runtime:
``` py
from quanestimation.base import QJL
```
`QJL` is initialized once on first access and shared across all modules.

## Calling patterns

### Constructing Julia objects

Python constructs Julia structs by calling them as functions with Python data:
``` py
from quanestimation import Lindblad, GeneralScheme

dynamics = Lindblad(H0, dH, tspan, decay=[], dyn_method="Expm")
scheme = GeneralScheme(probe=rho0, param=dynamics)
```

### Bang functions

Julia functions ending in `!` (in-place mutation) are not valid Python
identifiers. Use `getattr` to call them:
``` py
import quanestimation as qe
getattr(qe.QJL, "optimize!")(scheme, opt, algorithm=alg, objective=obj)
```

### Explicit type conversion

When Julia cannot infer types from Python data (empty lists, ambiguous shapes,
complex numbers), use `juliacall.convert`:
``` py
from juliacall import Main as jl, convert as jlconvert
ctrl_bound = jlconvert(jl.Vector[jl.Float64], [0.0, 1.0])
```

### Return values

Most results flow back to Python via `.npy` files written by Julia. For
functions that return values directly, `_unwrap()` converts Julia arrays to
numpy arrays.
