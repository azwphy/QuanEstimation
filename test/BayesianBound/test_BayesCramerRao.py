import pytest
import numpy as np
from scipy.integrate import simpson
from quanestimation.base.BayesianBound.BayesCramerRao import (
    BCFIM,
    BQFIM,
    BCRB,
    VTB,
    BQCRB,
    QVTB,
    OBB,
)
from quanestimation.base.Parameterization.GeneralDynamics import Lindblad

def test_bayesian_bound_singleparameter() -> None:
    """
    Test function for Bayesian bounds in quantum estimation for single parameter 
    scenario.
    
    This function tests various Bayesian bounds including:
    - Bayesian Cramer-Rao Bound (BCRB)
    - Van Trees Bound (VTB)
    - Bayesian Quantum Cramer-Rao Bound (BQCRB)
    - Quantum Van Trees Bound (QVTB)
    - Bayesian classical Fisher information
    - Bayesian quantum Fisher information
    """
    # Initial state
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    
    # Free Hamiltonian parameters
    b_val, omega0 = 0.5 * np.pi, 1.0
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    
    # Hamiltonian function
    hamiltonian_func = lambda x: 0.5 * b_val * omega0 * (
        sigma_x * np.cos(x) + sigma_z * np.sin(x)
    )
    
    # Derivative of Hamiltonian
    d_hamiltonian_func = lambda x: [
        0.5 * b_val * omega0 * (-sigma_x * np.sin(x) + sigma_z * np.cos(x))
    ]
    
    # Prior distribution parameters
    x_values = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 100)
    mu_val, eta_val = 0.0, 0.2
    
    # Probability density function and its derivative
    prob_density = lambda x, mu, eta: np.exp(
        -(x - mu) ** 2 / (2 * eta ** 2)
    ) / (eta * np.sqrt(2 * np.pi))
    
    d_prob_density = lambda x, mu, eta: -(
        (x - mu) * np.exp(-(x - mu) ** 2 / (2 * eta ** 2))
    ) / (eta ** 3 * np.sqrt(2 * np.pi))
    
    prob_values = np.array([prob_density(x_val, mu_val, eta_val) for x_val in x_values])
    d_prob_values = np.array([d_prob_density(x_val, mu_val, eta_val) for x_val in x_values])
    
    # Normalize the distribution
    norm_factor = simpson(prob_values, x_values)
    prob_normalized = prob_values / norm_factor
    d_prob_normalized = d_prob_values / norm_factor
    
    # Time evolution parameters
    time_span = np.linspace(0.0, 1.0, 50)
    
    # Prepare arrays for states and derivatives
    final_states = []
    d_final_states = []
    
    # Evolve the system for each parameter value
    for x_values_idx in x_values:
        hamiltonian = hamiltonian_func(x_values_idx)
        d_hamiltonian = d_hamiltonian_func(x_values_idx)
        
        dynamics = Lindblad(time_span, rho0, hamiltonian, d_hamiltonian)
        states, d_states = dynamics.expm()
        
        final_states.append(states[-1])
        d_final_states.append(d_states[-1]) 

    # Test BCFIM
    cfim = BCFIM([x_values], prob_normalized, final_states, d_final_states, M = [], eps = 1e-8)
    expected_cfim = 1.5342635936313218
    assert np.allclose(cfim, expected_cfim, atol = 1e-3)

    with pytest.raises(TypeError):
        cfim = BCFIM([x_values], prob_normalized, final_states, d_final_states, M = 1., eps = 1e-8)

    # Test BQFIM
    qfim = BQFIM([x_values], prob_normalized, final_states, d_final_states, LDtype = "SLD", eps = 1e-8)
    expected_qfim = 1.9629583923945833
    assert np.allclose(qfim, expected_qfim, atol = 1e-3)

    # Test BCRB type 1
    bcrb1 = BCRB([x_values], prob_normalized, [], final_states, d_final_states, M = [], btype = 1)
    expected_bcrb1 = 0.654654507602925
    assert np.allclose(bcrb1, expected_bcrb1, atol = 1e-3)

    # Test BCRB type 2
    bcrb2 = BCRB([x_values], prob_normalized, [], final_states, d_final_states, M = [], btype = 2)
    expected_bcrb2 = 0.651778484577857
    assert np.allclose(bcrb2, expected_bcrb2, atol = 1e-3)
    
    # Test BCRB type 3
    bcrb3 = BCRB([x_values], prob_normalized, d_prob_normalized, final_states, d_final_states, M = [], btype = 3)
    expected_bcrb3 = 0.16522254719803486
    assert np.allclose(bcrb3, expected_bcrb3, atol = 1e-3)

    # Test Van Trees Bound
    vtb = VTB([x_values], prob_normalized, d_prob_normalized, final_states, d_final_states, M = []) 
    expected_vtb = 0.03768712089828974
    assert np.allclose(vtb, expected_vtb, atol = 1e-3)

    # Test BQCRB type 1
    bqcrb1 = BQCRB([x_values], prob_normalized, [], final_states, d_final_states, btype = 1)
    expected_bqcrb1 = 0.5097987285760552
    assert np.allclose(bqcrb1, expected_bqcrb1, atol = 1e-3)

    # Test BQCRB type 2
    bqcrb2 = BQCRB([x_values], prob_normalized, [], final_states, d_final_states, btype = 2)
    expected_bqcrb2 = 0.5094351484343563
    assert np.allclose(bqcrb2, expected_bqcrb2, atol = 1e-3)

    # Test BQCRB type 3
    bqcrb3 = BQCRB([x_values], prob_normalized, d_prob_normalized, final_states, d_final_states, btype = 3)
    expected_bqcrb3 = 0.14347116223111836
    assert np.allclose(bqcrb3, expected_bqcrb3, atol = 1e-3)

    # Test Quantum Van Trees Bound
    qvtb = QVTB([x_values], prob_normalized, d_prob_normalized, final_states, d_final_states)
    expected_qvtb = 0.037087918374800306
    assert np.allclose(qvtb, expected_qvtb, atol = 1e-3)

    with pytest.raises(TypeError):
        BCFIM([x_values], prob_normalized, final_states, d_final_states, M = 1., eps = 1e-8)

    with pytest.raises(TypeError):    
        BCRB([x_values], prob_normalized, [], final_states, d_final_states, M = 1., btype = 1)

    with pytest.raises(TypeError):    
        VTB([x_values], prob_normalized, d_prob_normalized, final_states, d_final_states, M = 1.)

    with pytest.raises(ValueError):
        BCRB([x_values], prob_normalized, [], final_states, d_final_states, M = [], btype = 4)

    with pytest.raises(ValueError):    
        BQCRB([x_values], prob_normalized, [], final_states, d_final_states, btype = 4)

def test_bayesian_bound_multiparameter() -> None:  
    """
    Test function for Bayesian bounds in quantum estimation for multiparameter scenario.
    
    This function tests various Bayesian bounds including:
    - Bayesian Cramer-Rao Bound (BCRB)
    - Van Trees Bound (VTB)
    - Bayesian Quantum Cramer-Rao Bound (BQCRB)
    - Quantum Van Trees Bound (QVTB)
    - Bayesian classical Fisher information
    - Bayesian quantum Fisher information
    """
    # Initial state
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    
    # Free Hamiltonian parameters
    b_val = 0.5 * np.pi
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    
    # Hamiltonian function
    hamiltonian_func = lambda omega0, x: 0.5 * b_val * omega0 * (
        sigma_x * np.cos(x) + sigma_z * np.sin(x)
    )
    
    # Derivative of Hamiltonian
    d_hamiltonian_func = lambda omega0, x: [
        0.5 * b_val * (sigma_x * np.cos(x) + sigma_z * np.sin(x)),
        0.5 * b_val * omega0 * (-sigma_x * np.sin(x) + sigma_z * np.cos(x))
    ]
    
    # Prior distribution parameters
    x_values = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 20)
    omega0_values = np.linspace(1, 2, 20)
    all_parameter_values = [omega0_values, x_values]
    
    # Joint probability density function (Gaussian for both parameters)
    mu_omega0, mu_x = 1.5, 0.0
    eta_omega0, eta_x = 0.2, 0.2
    prob_density = lambda omega0, x: (
        np.exp(-(omega0 - mu_omega0)**2 / (2 * eta_omega0**2)) / (eta_omega0 * np.sqrt(2 * np.pi))
        * np.exp(-(x - mu_x)**2 / (2 * eta_x**2)) / (eta_x * np.sqrt(2 * np.pi))
    )

    d_prob_density = lambda omega0, x: (
        [-(omega0 - mu_omega0) / (eta_omega0**2) * prob_density(omega0, x),
        -(x - mu_x) / (eta_x**2) * prob_density(omega0, x)]
    )

    # Generate probability values
    d_prob_normalized = []
    prob_values_unnormalized = np.zeros((len(omega0_values), len(x_values)))
    for i, omega0_i in enumerate(omega0_values):
        d_prob_tp = []
        for j, x_values_j in enumerate(x_values):
            prob_values_unnormalized[i, j] = prob_density(omega0_i, x_values_j)
            d_prob_density_unnormalized = d_prob_density(omega0_i, x_values_j)
            d_prob_tp.append(d_prob_density_unnormalized)
            
        d_prob_normalized.append(d_prob_tp)

    # Normalize the distribution
    integral_x = np.zeros(len(omega0_values))
    for i in range(len(omega0_values)):
        integral_x[i] = simpson(prob_values_unnormalized[i, :], x_values)
    norm_factor = simpson(integral_x, omega0_values)
    prob_normalized = prob_values_unnormalized / norm_factor
    d_prob_normalized = d_prob_normalized / norm_factor

    # Time evolution parameters
    time_span = np.linspace(0.0, 1.0, 50)
    
    # Prepare arrays for states and derivatives
    final_states = []
    d_final_states = []
    
    # Evolve the system for each parameter combination
    for omega0_i in omega0_values:
        row_rho = []
        row_drho = []

        for x_values_j in x_values:
            hamiltonian = hamiltonian_func(omega0_i, x_values_j)
            d_hamiltonian = d_hamiltonian_func(omega0_i, x_values_j)

            dynamics = Lindblad(time_span, rho0, hamiltonian, d_hamiltonian)
            states, d_states = dynamics.expm()
            
            row_rho.append(states[-1])
            row_drho.append(d_states[-1])
        final_states.append(row_rho) 
        d_final_states.append(row_drho)    

    # Test BCFIM
    cfim = BCFIM(all_parameter_values, prob_normalized, final_states, d_final_states, M = [], eps = 1e-8)
    expected_cfim = np.array(
        [[0.0360, 0.], 
         [0., 2.1649]]
    )
    assert np.allclose(cfim, expected_cfim, atol = 1e-3)
    
    # Test BQFIM
    qfim = BQFIM(all_parameter_values, prob_normalized, final_states, d_final_states, LDtype="SLD", eps=1e-8)
    expected_qfim = np.array(
        [[0.0948, 0.], 
         [0.,  3.3352]]    
    )
    assert np.allclose(qfim, expected_qfim, atol = 1e-3)

    # Test BCRB type 1
    bcrb1 = BCRB(all_parameter_values, prob_normalized, [], final_states, d_final_states, M = [], btype = 1)
    expected_bcrb1 = np.array(
        [[188.85062035, 0.63311697],
         [0.63311697, 0.62231953]]
    )
    assert np.allclose(bcrb1, expected_bcrb1, atol = 1e-3)   
    
    # Test BCRB type 2
    bcrb2 = BCRB(all_parameter_values, prob_normalized, [], final_states, d_final_states, M = [], btype = 2)
    expected_bcrb2 = np.array(
        [[27.7234240, 0.002116],
         [0.002116, 0.461910]]
    )
    assert np.allclose(bcrb2, expected_bcrb2, atol = 1e-3)

    # # Test BCRB type 3
    bcrb3 = BCRB(all_parameter_values, prob_normalized, d_prob_normalized, final_states, d_final_states, M = [], btype = 3)
    expected_bcrb3 = np.array(
        [[2.52942056, -0.00943802],
         [-0.00943802, 0.38853841]]
    )
    assert np.allclose(bcrb3, expected_bcrb3, atol = 1e-3)

    # # Test Van Trees Bound
    vtb = VTB(all_parameter_values, prob_normalized, d_prob_normalized, final_states, d_final_states, M = []) 
    expected_vtb = np.array(
        [[0.04382, 0.],
         [0., 0.03681]]
    )
    assert np.allclose(vtb, expected_vtb, atol = 1e-3)

    # Test BQCRB type 1
    bqcrb1 = BQCRB(all_parameter_values, prob_normalized, [], final_states, d_final_states, btype = 1)
    expected_bqcrb1 = np.array(
        [[45.48725379, 0.33691038],
         [0.33691038,  0.36839637]]
    )
    assert np.allclose(bqcrb1, expected_bqcrb1, atol = 1e-3)

    # Test BQCRB type 2
    bqcrb2 = BQCRB(all_parameter_values, prob_normalized, [], final_states, d_final_states, btype = 2)
    expected_bqcrb2 = np.array(
        [[10.542814, 0.0015452],
         [0.0015452, 0.29983117]]
    )
    assert np.allclose(bqcrb2, expected_bqcrb2, atol = 1e-3)

    # Test BQCRB type 3
    bqcrb3 = BQCRB(all_parameter_values, prob_normalized, d_prob_normalized, final_states, d_final_states, btype = 3)
    expected_bqcrb3 = np.array(
        [[1.39714369, -0.00959793],
         [-0.00959793, 0.25794208]]
    )
    assert np.allclose(bqcrb3, expected_bqcrb3, atol = 1e-3)

    # # Test Quantum Van Trees Bound
    qvtb = QVTB(all_parameter_values, prob_normalized, d_prob_normalized, final_states, d_final_states)
    expected_qvtb = np.array(
        [[0.04371, 0.],
         [0., 0.03529]]
    )
    assert np.allclose(qvtb, expected_qvtb, atol = 1e-3)
    
    with pytest.raises(TypeError):
        BCFIM(all_parameter_values, prob_normalized, final_states, d_final_states, M = 1., eps = 1e-8)

    with pytest.raises(TypeError):    
        BCRB(all_parameter_values, prob_normalized, [], final_states, d_final_states, M = 1., btype = 1)

    with pytest.raises(TypeError):    
        VTB(all_parameter_values, prob_normalized, d_prob_normalized, final_states, d_final_states, M = 1.)

    with pytest.raises(ValueError):
        BCRB(all_parameter_values, prob_normalized, [], final_states, d_final_states, M = [], btype = 4)

    with pytest.raises(ValueError):    
        BQCRB(all_parameter_values, prob_normalized, [], final_states, d_final_states, btype = 4)


@pytest.mark.parametrize("LDtype", [
    "SLD",
    "RLD",
    "LLD",
])
def test_bqfim_ldtype_single(LDtype):
    """BQFIM should produce finite results for all LD types (single-parameter)."""
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    b_val, omega0 = 0.5 * np.pi, 1.0
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    x_values = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 20)

    def hamiltonian_func(x):
        return 0.5 * b_val * omega0 * (sigma_x * np.cos(x) + sigma_z * np.sin(x))
    def d_hamiltonian_func(x):
        return [0.5 * b_val * omega0 * (-sigma_x * np.sin(x) + sigma_z * np.cos(x))]

    prob_values = np.ones(len(x_values)) / len(x_values)
    prob_normalized = prob_values / simpson(prob_values, x_values)

    time_span = np.linspace(0.0, 1.0, 20)
    final_states, d_final_states = [], []
    for xv in x_values:
        dynamics = Lindblad(time_span, rho0, hamiltonian_func(xv), d_hamiltonian_func(xv))
        states, d_states = dynamics.expm()
        final_states.append(states[-1])
        d_final_states.append(d_states[-1])

    try:
        result = BQFIM([x_values], prob_normalized, final_states, d_final_states, LDtype=LDtype)
    except ValueError as e:
        assert "does not exist" in str(e), \
            f"{LDtype}: unexpected ValueError: {e}"
        return
    assert np.isfinite(result)


@pytest.mark.parametrize("btype", [1, 2, 3])
def test_bcrb_btype_single(btype):
    """BCRB should produce finite results for all btype values (single-parameter)."""
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    b_val, omega0 = 0.5 * np.pi, 1.0
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    x_values = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 20)

    def hamiltonian_func(x):
        return 0.5 * b_val * omega0 * (sigma_x * np.cos(x) + sigma_z * np.sin(x))
    def d_hamiltonian_func(x):
        return [0.5 * b_val * omega0 * (-sigma_x * np.sin(x) + sigma_z * np.cos(x))]

    prob_values = np.ones(len(x_values)) / len(x_values)
    prob_normalized = prob_values / simpson(prob_values, x_values)
    d_prob = np.zeros(len(x_values))

    time_span = np.linspace(0.0, 1.0, 20)
    final_states, d_final_states = [], []
    for xv in x_values:
        dynamics = Lindblad(time_span, rho0, hamiltonian_func(xv), d_hamiltonian_func(xv))
        states, d_states = dynamics.expm()
        final_states.append(states[-1])
        d_final_states.append(d_states[-1])

    result = BCRB([x_values], prob_normalized, d_prob, final_states, d_final_states, M=[], btype=btype)
    assert np.isfinite(result)


@pytest.mark.parametrize("btype", [1, 2, 3])
def test_bqcrb_btype_single(btype):
    """BQCRB should produce finite results for all btype values (single-parameter)."""
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    b_val, omega0 = 0.5 * np.pi, 1.0
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    x_values = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 20)

    def hamiltonian_func(x):
        return 0.5 * b_val * omega0 * (sigma_x * np.cos(x) + sigma_z * np.sin(x))
    def d_hamiltonian_func(x):
        return [0.5 * b_val * omega0 * (-sigma_x * np.sin(x) + sigma_z * np.cos(x))]

    prob_values = np.ones(len(x_values)) / len(x_values)
    prob_normalized = prob_values / simpson(prob_values, x_values)
    d_prob = np.zeros(len(x_values))

    time_span = np.linspace(0.0, 1.0, 20)
    final_states, d_final_states = [], []
    for xv in x_values:
        dynamics = Lindblad(time_span, rho0, hamiltonian_func(xv), d_hamiltonian_func(xv))
        states, d_states = dynamics.expm()
        final_states.append(states[-1])
        d_final_states.append(d_states[-1])

    result = BQCRB([x_values], prob_normalized, d_prob, final_states, d_final_states, btype=btype)
    assert np.isfinite(result)


@pytest.mark.parametrize("LDtype", [
    "SLD",
    "RLD",
    "LLD",
])
def test_qvtb_ldtype_single(LDtype):
    """QVTB should produce finite results for all LD types (single-parameter)."""
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    b_val, omega0 = 0.5 * np.pi, 1.0
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    x_values = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 20)

    def hamiltonian_func(x):
        return 0.5 * b_val * omega0 * (sigma_x * np.cos(x) + sigma_z * np.sin(x))
    def d_hamiltonian_func(x):
        return [0.5 * b_val * omega0 * (-sigma_x * np.sin(x) + sigma_z * np.cos(x))]

    prob_values = np.ones(len(x_values)) / len(x_values)
    prob_normalized = prob_values / simpson(prob_values, x_values)
    d_prob = np.zeros(len(x_values))

    time_span = np.linspace(0.0, 1.0, 20)
    final_states, d_final_states = [], []
    for xv in x_values:
        dynamics = Lindblad(time_span, rho0, hamiltonian_func(xv), d_hamiltonian_func(xv))
        states, d_states = dynamics.expm()
        final_states.append(states[-1])
        d_final_states.append(d_states[-1])

    try:
        result = QVTB([x_values], prob_normalized, d_prob, final_states, d_final_states, LDtype=LDtype)
    except ValueError as e:
        assert "does not exist" in str(e), \
            f"{LDtype}: unexpected ValueError: {e}"
        return
    assert np.isfinite(result)


def test_obb_smoke():
    """OBB should produce a finite result for a simple single-parameter system."""
    rho0 = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    b_val, omega0 = 0.5 * np.pi, 1.0
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    x_values = np.linspace(-0.3 * np.pi, 0.3 * np.pi, 20)

    def hamiltonian_func(x):
        return 0.5 * b_val * omega0 * (sigma_x * np.cos(x) + sigma_z * np.sin(x))
    def d_hamiltonian_func(x):
        return [0.5 * b_val * omega0 * (-sigma_x * np.sin(x) + sigma_z * np.cos(x))]
    def d2_hamiltonian_func(x):
        return [0.5 * b_val * omega0 * (-sigma_x * np.cos(x) - sigma_z * np.sin(x))]

    eta_val = 0.5
    prob_density = lambda x, mu, eta: np.exp(-(x - mu) ** 2 / (2 * eta ** 2)) / (eta * np.sqrt(2 * np.pi))
    d_prob_density = lambda x, mu, eta: -(x - mu) * np.exp(-(x - mu) ** 2 / (2 * eta ** 2)) / (eta ** 3 * np.sqrt(2 * np.pi))
    prob_values = np.array([prob_density(xv, 0.0, eta_val) for xv in x_values])
    d_prob_values = np.array([d_prob_density(xv, 0.0, eta_val) for xv in x_values])
    norm = simpson(prob_values, x_values)
    prob_normalized = prob_values / norm
    d_prob_normalized = d_prob_values / norm

    time_span = np.linspace(0.0, 1.0, 20)
    final_states, d_final_states, d2_final_states = [], [], []
    for xv in x_values:
        dynamics = Lindblad(time_span, rho0, hamiltonian_func(xv), d_hamiltonian_func(xv))
        states, d_states = dynamics.expm()
        d2_dynamics = Lindblad(time_span, rho0, hamiltonian_func(xv), d2_hamiltonian_func(xv))
        _, d2_states = d2_dynamics.expm()
        final_states.append(states[-1])
        d_final_states.append(d_states[-1])
        d2_final_states.append(d2_states[-1])

    result = OBB(x_values, prob_normalized, d_prob_normalized, final_states, d_final_states, d2_final_states)
    assert np.isfinite(result)
    assert result >= 0

