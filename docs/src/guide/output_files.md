# **Output files**

In QuanEstimation, the output data will be saved into files during or after the optimization
process. There are two categories: the values of the objective function at each step, and the
optimized variables from the corresponding scheme. This guide describes all output file types
and how to load them.

## **Objective function values**

During optimization, the objective function is evaluated at each step and the values are saved
sequentially (one value per line) into `f.csv`.

=== "Python"
    ``` py
    import numpy as np
    f = np.loadtxt("f.csv")
    ```

## **Control optimization**

The control optimization results are saved into `controls.npy`. The shape depends on
`savefile`:

- `savefile=False` (default): the final control coefficients are saved, with shape
  `(n_ctrl, n_tseg)` — the first dimension is the number of control Hamiltonians,
  the second dimension is the number of time segments.
- `savefile=True`: all control coefficients after each round of optimization are saved
  with shape `(n_rounds, n_ctrl, n_tseg)`.

=== "Python"
    ``` py
    import numpy as np
    controls = np.load("controls.npy")
    ```

If the optimized controls are exported as CSV (Julia side), use `csv2npy_controls` to convert:
``` py
from quanestimation import csv2npy_controls
csv2npy_controls(controls_csv_data, num=n_tseg)
```
This reshapes the CSV array and saves as `controls.npy`.

See also: [Control optimization](guide_Copt.md)

## **State optimization**

The state optimization results are saved into `states.npy`.

- `savefile=False` (default): the final optimized state vectors are saved.
- `savefile=True`: all state vectors after each round are saved.

=== "Python"
    ``` py
    import numpy as np
    states = np.load("states.npy")
    ```

Use `csv2npy_states` to convert CSV-based state files:
``` py
from quanestimation import csv2npy_states
csv2npy_states(states_csv_data, num=1)
```

See also: [State optimization](guide_Sopt.md)

## **Measurement optimization**

The measurement optimization results are saved into `measurements.npy`.

- `savefile=False` (default): the final optimized POVMs are saved.
- `savefile=True`: all POVM lists after each round are saved.

=== "Python"
    ``` py
    import numpy as np
    M = np.load("measurements.npy")
    ```

If measurements are exported as CSV via `writedlm`, use `csv2npy_measurements` to convert:
``` py
from quanestimation import csv2npy_measurements
csv2npy_measurements(M_csv_data, num=n_operators)
```

See also: [Measurement optimization](guide_Mopt.md)

## **Comprehensive optimization**

Comprehensive optimization combines multiple variable types. The output follows the same
conventions as above:

- Controls are saved in `controls.npy`
- States are saved in `states.npy`
- Measurements are saved in `measurements.npy`

Each file is present only if that variable type was part of the optimization. For example,
SC optimization produces `controls.npy` and `states.npy`, but not `measurements.npy`.

See also: [Comprehensive optimization](guide_Compopt.md)

## **Bayesian estimation**

The `Bayes()` estimator produces the following output files:

| File | Contents | Dimensions |
|------|----------|------------|
| `pout.npy` | Posterior probability distributions | `(n_iter, n_grid_points)` |
| `xout.npy` | Estimated parameter values | `(n_iter,)` or `(n_iter, n_params)` |
| `Lout.npy` | Likelihood functions (when `continue=True`) | `(n_iter, n_grid_points)` |

=== "Python"
    ``` py
    import numpy as np
    pout = np.load("pout.npy")
    xout = np.load("xout.npy")
    ```

See also: [Quantum metrological tools](guide_bounds.md)

## **Adaptive measurement schemes**

Adaptive estimation produces additional files via the `Adapt()` module:

| File | Contents |
|------|----------|
| `pout.csv` | Posterior distributions at each adaptive step |
| `xout.csv` | Estimated values at each adaptive step |
| `y.csv` | Experimental data at each step |
| `f.csv` | Objective function across iterations |

=== "Python"
    ``` py
    import numpy as np
    pout = np.loadtxt("pout.csv")
    xout = np.loadtxt("xout.csv")
    y = np.loadtxt("y.csv")
    ```

See also: [Adaptive measurement schemes](guide_adaptive.md)
