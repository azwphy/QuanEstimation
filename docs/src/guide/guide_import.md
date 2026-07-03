# **Getting started**

QuanEstimation provides interfaces in both Python and Julia. They share the
same API design — function names, parameter order, and defaults are identical
across languages. Use whichever you prefer.

=== "Python"
    ``` py
    from quanestimation import *
    ```
    All public functions and classes are imported into the current namespace.

=== "Julia"
    ``` jl
    using QuanEstimation
    ```

For advanced usage (direct Julia access, bang functions, type conversion),
see [Python-Julia calling patterns](guide_pyju.md).

## **Next steps**

- [Parameterization process](guide_dynamics.md) — define the system dynamics
  (Hamiltonian, decay, Kraus operators)
- [Quantum metrological tools](guide_bounds.md) — compute Fisher information,
  Cramér-Rao bounds, and perform Bayesian estimation
