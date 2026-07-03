import numpy as np
from numbers import Number
from scipy import interpolate
from scipy.integrate import simpson, solve_bvp
from itertools import product
from quanestimation.base.AsymptoticBound.CramerRao import CFIM, QFIM
from quanestimation.base.Common.Common import SIC, extract_ele


def _check_and_init_M(M, dim):
    if M == []:
        return SIC(dim)
    if not isinstance(M, list):
        raise TypeError("Please make sure M is a list!")
    return M


def _unwrap_single_param(arr, p_num):
    if isinstance(arr[0], (list, np.ndarray)):
        return [arr[i][0] for i in range(p_num)]
    return arr


def _build_param_grid(p, dp, rho, drho, para_num):
    p_shape = np.shape(p)
    p_ext = extract_ele(p, para_num)
    dp_ext = extract_ele(dp, para_num)
    rho_ext = extract_ele(rho, para_num)
    drho_ext = extract_ele(drho, para_num)
    p_list, rho_list, drho_list = [], [], []
    for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
        p_list.append(p_ele)
        rho_list.append(rho_ele)
        drho_list.append(drho_ele)
    return p_shape, p_list, rho_list, drho_list, p_ext, dp_ext


def _allocate_F_list(para_num, n_points):
    return [
        [[0.0 for _ in range(n_points)] for _ in range(para_num)]
        for _ in range(para_num)
    ]


def _integrate_F_list(F_list, p, x, para_num, p_shape):
    res = np.zeros([para_num, para_num])
    for para_i in range(para_num):
        for para_j in range(para_i, para_num):
            F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
            arr = p * F_ij
            for si in reversed(range(para_num)):
                arr = simpson(arr, x[si])
            res[para_i][para_j] = arr
            res[para_j][para_i] = arr
    return res


def _build_b_db_lists(b, db, para_num):
    b_list, db_list = [], []
    for b_ele, db_ele in zip(product(*b), product(*db)):
        b_list.append([b_ele[i] for i in range(para_num)])
        db_list.append([db_ele[j] for j in range(para_num)])
    return b_list, db_list


def BCFIM(x, p, rho, drho, M=None, eps=1e-8):
    r"""
    Calculation of the Bayesian classical Fisher information matrix (BCFIM).

    This function computes the Bayesian classical Fisher information (BCFI) or Bayesian classical 
    Fisher information matrix (BCFIM). The BCFIM is defined as:

    $$
        \mathcal{I}_{\mathrm{Bayes}} = \int p(\textbf{x}) \mathcal{I} \, \mathrm{d}\textbf{x}.
    $$

    where $\mathcal{I}$ is the classical Fisher information matrix (CFIM) and $p(\textbf{x})$ 
    is the prior distribution.

    Args:
        x (list): 
            Parameter regimes for integration. Each element is an array 
            representing the values of one parameter.
        p (np.array): 
            Prior distribution over the parameter space. Must have the same dimensions 
            as the product of the lengths of the arrays in `x`.
        rho (list): 
            Parameterized density matrices. Each element corresponds to 
            a point in the parameter space defined by `x`.
        drho (list): 
            Derivatives of the density matrices with respect to the parameters. For single parameter estimation (length of `x` is 1), 
            `drho` is a list of derivatives at each parameter point. For multiparameter estimation, `drho` is a 
            multidimensional list where `drho[i]` is a list of derivatives with respect to each parameter at the i-th parameter point, 
            and `drho[i][j]` is the derivative of the density matrix at the i-th parameter point with respect to the j-th parameter.
        M (list, optional): 
            Positive operator-valued measure (POVM). Default is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
        eps (float, optional): 
            Machine epsilon for numerical stability.

    Returns:
        (float/np.array): 
            For single parameter estimation (length of `x` is 1), returns BCFI.             
            For multiparameter estimation (length of `x` > 1), returns BCFIM.

    Raises:
        TypeError: 
            If `M` is provided but not a list.

    Notes:
        SIC-POVM is calculated using Weyl-Heisenberg covariant SIC-POVM fiducial states 
        available at [http://www.physics.umb.edu/Research/QBism/solutions.html](http://www.physics.umb.edu/Research/QBism/solutions.html).
    """

    if M is None: M = []
    para_num = len(x)
    if para_num == 1:
        #### single parameter scenario ####
        M = _check_and_init_M(M, len(rho[0]))
        drho = _unwrap_single_param(drho, len(p))
        p_num = len(p)
        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = CFIM(rho[m], [drho[m]], M=M, eps=eps)

        arr = [p[i] * F_tp[i] for i in range(p_num)]
        return simpson(arr, x[0])
    else:
        #### multiparameter scenario ####
        p_shape, p_list, rho_list, drho_list, _, _ = _build_param_grid(
            p, np.zeros_like(p), rho, drho, para_num)
        M = _check_and_init_M(M, len(rho_list[0]))

        F_list = _allocate_F_list(para_num, len(p_list))
        for i in range(len(p_list)):
            drho_arrays = [np.array(d, dtype=np.complex128) for d in drho_list[i]]
            F_tp = CFIM(rho_list[i], drho_arrays, M=M, eps=eps)
            for pj in range(para_num):
                for pk in range(para_num):
                    F_list[pj][pk][i] = float(np.real(F_tp[pj][pk]))

        return _integrate_F_list(F_list, p, x, para_num, p_shape)


def BQFIM(x, p, rho, drho, LDtype="SLD", eps=1e-8):
    r"""
    Calculation of the Bayesian quantum Fisher information matrix (BQFIM).

    This function computes the Bayesian quantum Fisher information (BQFI) or Bayesian quantum 
    Fisher information matrix (BQFIM). The BQFIM is defined as:

    $$
        \mathcal{F}_{\mathrm{Bayes}} = \int p(\textbf{x}) \mathcal{F} \, \mathrm{d}\textbf{x}.
    $$

    where $\mathcal{F}$ is the quantum Fisher information matrix (QFIM) and $p(\textbf{x})$ 
    is the prior distribution.

    Args:
        x (list): Parameter regimes for integration. Each element is an array 
            representing the values of one parameter.
        p (np.array): Prior distribution over the parameter space. Must have the same dimensions 
            as the product of the lengths of the arrays in `x`.
        rho (list): Parameterized density matrices. Each element corresponds to 
            a point in the parameter space defined by `x`.
        drho (list): Derivatives of the density matrices with respect to 
            the parameters. `drho[i][j]` is the derivative of the density matrix at the i-th 
            parameter point with respect to the j-th parameter.
        LDtype (str, optional): Type of logarithmic derivative (default: "SLD"). Options:  
            - "SLD": Symmetric logarithmic derivative  
            - "RLD": Right logarithmic derivative  
            - "LLD": Left logarithmic derivative  
        eps (float, optional): Machine epsilon for numerical stability.

    Returns:
        (float/np.array): 
            For single parameter estimation (length of `x` is 1), returns BQFI. 
            For multiparameter estimation (length of `x` > 1), returns BQFIM.
    """

    para_num = len(x)
    if para_num == 1:
        #### single parameter scenario ####
        p_num = len(p)
        drho = _unwrap_single_param(drho, len(p))

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = QFIM(rho[m], [drho[m]], LDtype=LDtype, eps=eps)
        arr = [p[i] * F_tp[i] for i in range(p_num)]
        return simpson(arr, x[0])
    else:
        #### multiparameter scenario ####
        p_shape, p_list, rho_list, drho_list, _, _ = _build_param_grid(
            p, np.zeros_like(p), rho, drho, para_num)

        F_list = _allocate_F_list(para_num, len(p_list))
        for i in range(len(p_list)):
            F_tp = QFIM(rho_list[i], drho_list[i], LDtype=LDtype, eps=eps)
            for pj in range(para_num):
                for pk in range(para_num):
                    F_list[pj][pk][i] = F_tp[pj][pk]

        return _integrate_F_list(F_list, p, x, para_num, p_shape)


def BCRB(x, p, dp, rho, drho, M=None, b=None, db=None, btype=1, eps=1e-8):
    r"""
    Calculation of the Bayesian Cramer-Rao bound (BCRB).

    This function computes the Bayesian Cramer-Rao bound (BCRB) for single or multiple parameters.

    The covariance matrix with prior distribution $p(\textbf{x})$ is:

    $$
        \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\}) = \int p(\textbf{x}) \sum_y \mathrm{Tr}
        (\rho\Pi_y) (\hat{\textbf{x}}-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}
        \mathrm{d}\textbf{x}.
    $$

    This function calculates three types of BCRB:

    **Type 1:**

    $$
        \mathrm{cov} \geq \int p(\textbf{x}) \left( B \mathcal{I}^{-1} B 
        + \textbf{b} \textbf{b}^{\mathrm{T}} \right) \mathrm{d}\textbf{x}.
    $$

    **Type 2:**
    $$
        \mathrm{cov} \geq \mathcal{B} \mathcal{I}_{\mathrm{Bayes}}^{-1} \mathcal{B} 
        + \int p(\textbf{x}) \textbf{b} \textbf{b}^{\mathrm{T}} \mathrm{d}\textbf{x}.
    $$

    **Type 3:**
    $$
        \mathrm{cov} \geq \int p(\textbf{x}) \mathcal{G} \left( \mathcal{I}_p 
        + \mathcal{I} \right)^{-1} \mathcal{G}^{\mathrm{T}} \mathrm{d}\textbf{x}.
    $$

    Symbols:
        - $\textbf{b}$: bias vector
        - $\textbf{b}'$: its derivatives
        - $B$: diagonal matrix with $B_{ii} = 1 + [\textbf{b}']_{i}$
        - $\mathcal{I}$: classical Fisher information matrix (CFIM)
        - $\mathcal{B} = \int p(\textbf{x}) B \mathrm{d}\textbf{x}$
        - $\mathcal{I}_{\mathrm{Bayes}} = \int p(\textbf{x}) \mathcal{I} \mathrm{d}\textbf{x}$
        - $[\mathcal{I}_{p}]_{ab} = [\partial_a \ln p(\textbf{x})][\partial_b \ln p(\textbf{x})]$
        - $\mathcal{G}_{ab} = [\partial_b \ln p(\textbf{x})][\textbf{b}]_a + B_{aa}\delta_{ab}$

    Args:
        x (list): 
            Parameter regimes for integration.
        p (np.array): 
            Prior distribution over the parameter space. Must have the same dimensions 
            as the product of the lengths of the arrays in `x`.
        dp (list): 
            Derivatives of the prior distribution with respect to the parameters.
        rho (list): 
            Parameterized density matrices. Each element corresponds to 
            a point in the parameter space defined by `x`.
        drho (list): 
            Derivatives of the density matrices with respect to 
            the parameters. `drho[i][j]` is the derivative of the density matrix at the i-th 
            parameter point with respect to the j-th parameter.
        M (list, optional): 
            Positive operator-valued measure (POVM). Default is 
            a set of rank-one symmetric informationally complete POVM (SIC-POVM).
        b (list, optional): 
            Bias vector. Default is zero bias.
        db (list, optional): 
            Derivatives of the bias vector. Default is zero.
        btype (int, optional): 
            Type of BCRB to calculate (1, 2, or 3).
        eps (float, optional): 
            Machine epsilon for numerical stability.

    Returns:
        (float/np.array): 
            For single parameter estimation (length of `x` is 1), returns BCRB. 
            For multiparameter estimation (length of `x` > 1), returns BCRB matrix.

    Raises:
        TypeError: If `M` is provided but not a list.
        NameError: If `btype` is not in {1, 2, 3}.

    Notes:
        SIC-POVM is calculated using Weyl-Heisenberg covariant SIC-POVM fiducial states 
        available at [http://www.physics.umb.edu/Research/QBism/solutions.html](http://www.physics.umb.edu/Research/QBism/solutions.html).
    """

    if M is None: M = []
    if b is None: b = []
    if db is None: db = []
    para_num = len(x)
    if para_num == 1:
        #### single parameter scenario ####
        p_num = len(p)
        if not b:
            b = np.zeros(p_num)
            db = np.zeros(p_num)
        elif not db:
            db = np.zeros(p_num)

        M = _check_and_init_M(M, len(rho[0]))

        drho = _unwrap_single_param(drho, p_num)
        b = _unwrap_single_param(b, p_num)
        db = _unwrap_single_param(db, p_num)

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = CFIM(rho[m], [drho[m]], M=M, eps=eps)

        if btype == 1:
            arr = [
                p[i] * ((1 + db[i]) ** 2 / F_tp[i] + b[i] ** 2) for i in range(p_num)
            ]
            F = simpson(arr, x[0])
            return F
        elif btype == 2:
            arr = [p[i] * F_tp[i] for i in range(p_num)]
            F1 = simpson(arr, x[0])
            arr2 = [p[j] * (1 + db[j]) for j in range(p_num)]
            B = simpson(arr2, x[0])
            arr3 = [p[k] * b[k] ** 2 for k in range(p_num)]
            bb = simpson(arr3, x[0])
            F = B**2 / F1 + bb
            return F
        elif btype == 3:
            I_tp = [np.real(dp[i] * dp[i] / p[i] ** 2) for i in range(p_num)]
            arr = [p[j]*(dp[j]*b[j]/p[j]+(1 + db[j]))**2 / (I_tp[j] + F_tp[j]) for j in range(p_num)]
            F = simpson(arr, x[0])
            return F
        else:
            raise ValueError("btype should be choosen in {1, 2, 3}.")
    else:
        #### multiparameter scenario ####
        if not b:
            b, db = [], []
            for i in range(para_num):
                b.append(np.zeros(len(x[i])))
                db.append(np.zeros(len(x[i])))
        elif not db:
            db = []
            for i in range(para_num):
                db.append(np.zeros(len(x[i])))

        p_shape, p_list, rho_list, drho_list, _, dp_ext = _build_param_grid(
            p, dp, rho, drho, para_num)
        dp_list = [dpi for dpi in dp_ext]
        b_list, db_list = _build_b_db_lists(b, db, para_num)

        dim = len(rho_list[0])
        M = _check_and_init_M(M, dim)
        if btype == 1:
            F_list = _allocate_F_list(para_num, len(p_list))
            for i in range(len(p_list)):
                F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
                F_inv = np.linalg.pinv(F_tp)
                B = np.diag([(1.0 + db_list[i][j]) for j in range(para_num)])
                term1 = B @ F_inv @ B
                term2 = np.dot(
                    np.array(b_list[i]).reshape(para_num, 1),
                    np.array(b_list[i]).reshape(1, para_num),
                )
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = term1[pj][pk] + term2[pj][pk]

            return _integrate_F_list(F_list, p, x, para_num, p_shape)
        elif btype == 2:
            F_list = _allocate_F_list(para_num, len(p_list))
            B_list = _allocate_F_list(para_num, len(p_list))
            bb_list = _allocate_F_list(para_num, len(p_list))
            for i in range(len(p_list)):
                F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
                B_tp = np.diag([(1.0 + db_list[i][j]) for j in range(para_num)])
                bb_tp = np.dot(
                    np.array(b_list[i]).reshape(para_num, 1),
                    np.array(b_list[i]).reshape(1, para_num),
                )
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tp[pj][pk]
                        B_list[pj][pk][i] = B_tp[pj][pk]
                        bb_list[pj][pk][i] = bb_tp[pj][pk]

            F_res = _integrate_F_list(F_list, p, x, para_num, p_shape)
            B_res = _integrate_F_list(B_list, p, x, para_num, p_shape)
            bb_res = _integrate_F_list(bb_list, p, x, para_num, p_shape)
            return B_res @ np.linalg.pinv(F_res) @ B_res + bb_res
        elif btype == 3:
            F_list = _allocate_F_list(para_num, len(p_list))
            for i in range(len(p_list)):
                F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
                I_tp = np.zeros((para_num, para_num))
                G_tp = np.zeros((para_num, para_num))
                for pm in range(para_num):
                    for pn in range(para_num):
                        if pm == pn:
                            G_tp[pm][pn] = dp_list[i][pn]*b_list[i][pm]/p_list[i]+(1.0 + db_list[i][pm])
                        else:
                            G_tp[pm][pn] = dp_list[i][pn]*b_list[i][pm]/p_list[i]
                        I_tp[pm][pn] = dp_list[i][pm] * dp_list[i][pn] / p_list[i] ** 2

                F_tot = G_tp @ np.linalg.pinv(F_tp + I_tp) @ G_tp.T
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tot[pj][pk]

            return _integrate_F_list(F_list, p, x, para_num, p_shape)
        else:
            raise ValueError("btype should be choosen in {1, 2, 3}.")


def BQCRB(x, p, dp, rho, drho, b=None, db=None, btype=1, LDtype="SLD", eps=1e-8):
    r"""
    Calculation of the Bayesian quantum Cramer-Rao bound (BQCRB). 
    
    The covariance matrix with a prior distribution $p(\textbf{x})$ is defined as
    
    $$
        \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})=\int p(\textbf{x})\sum_y\mathrm{Tr}
        (\rho\Pi_y)(\hat{\textbf{x}}-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}
        \mathrm{d}\textbf{x},
    $$

    Symbols:
        - $\textbf{x}=(x_0,x_1,\dots)^{\mathrm{T}}$: the unknown parameters to be estimated
            and the integral $\int\mathrm{d}\textbf{x}:=\iiint\mathrm{d}x_0\mathrm{d}x_1\cdots$.
        - $\{\Pi_y\}$: a set of positive operator-valued measure (POVM). 
        - $\rho$: the parameterized density matrix.

    This function calculates three types of the BQCRB. The first one is

    $$
        \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq\int p(\textbf{x})\left(B\mathcal{F}^{-1}B
        +\textbf{b}\textbf{b}^{\mathrm{T}}\right)\mathrm{d}\textbf{x},
    $$
        
    Symbols: 
        - $\textbf{b}$ and $\textbf{b}'$: the vectors of biase and its derivatives on parameters.
        - $B$: a diagonal matrix with the $i$th entry $B_{ii}=1+[\textbf{b}']_{i}$
        - $\mathcal{F}$: the QFIM for all types.

    The second one is

    $$
        \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \mathcal{B}\,\mathcal{F}_{\mathrm{Bayes}}^{-1}\,
        \mathcal{B}+\int p(\textbf{x})\textbf{b}\textbf{b}^{\mathrm{T}}\mathrm{d}\textbf{x},
    $$

    Symbols: 
        - $\mathcal{B}=\int p(\textbf{x})B\mathrm{d}\textbf{x}$: the average of $B$ 
        - $\mathcal{F}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{F}\mathrm{d}\textbf{x}$: the average QFIM.

    The third one is
    
    $$
        \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \int p(\textbf{x})
        \mathcal{G}\left(\mathcal{I}_p+\mathcal{F}\right)^{-1}\mathcal{G}^{\mathrm{T}}\mathrm{d}\textbf{x}.
    $$

    Symbols: 
        - $[\mathcal{I}_{p}]_{ab}:=[\partial_a \ln p(\textbf{x})][\partial_b \ln p(\textbf{x})]$.
        - $\mathcal{G}_{ab}:=[\partial_b\ln p(\textbf{x})][\textbf{b}]_a+B_{aa}\delta_{ab}$.

    Args:
        x (list): 
            The regimes of the parameters for the integral.
        p (np.array, multidimensional): 
            The prior distribution.
        rho (list, multidimensional): 
            Parameterized density matrix.
        drho (list, multidimensional): 
            Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
        b (list): 
            Vector of biases of the form $\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}$.
        db (list): 
            Derivatives of b with respect to the unknown parameters to be estimated, It should be 
            expressed as $\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}$.
        btype (int): 
            Types of the BQCRB. Options are:  
                1 (default) -- It means to calculate the first type of the BQCRB.  
                2 -- It means to calculate the second type of the BQCRB.
                3 -- It means to calculate the third type of the BCRB.
        LDtype (str): 
            Types of QFI (QFIM) can be set as the objective function. Options are:  
                - "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
                - "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
                - "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).
        eps (float,optional): 
            Machine epsilon.

    Returns:
        (float/np.array): 
            For single parameter estimation (the length of `x` equals to one), the 
            output is a float and for multiparameter estimation (the length of `x` is larger than one), 
            it returns a matrix.
    """

    if b is None: b = []
    if db is None: db = []
    para_num = len(x)

    if para_num == 1:
        #### single parameter scenario ####
        p_num = len(p)

        if not b:
            b = np.zeros(p_num)
            db = np.zeros(p_num)
        elif not db:
            db = np.zeros(p_num)

        drho = _unwrap_single_param(drho, p_num)
        b = _unwrap_single_param(b, p_num)
        db = _unwrap_single_param(db, p_num)

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = QFIM(rho[m], [drho[m]], LDtype=LDtype, eps=eps)

        if btype == 1:
            arr = [
                p[i] * ((1 + db[i]) ** 2 / F_tp[i] + b[i] ** 2) for i in range(p_num)
            ]
            F = simpson(arr, x[0])
            return F
        elif btype == 2:
            arr2 = [p[i] * F_tp[i] for i in range(p_num)]
            F2 = simpson(arr2, x[0])
            arr2 = [p[j] * (1 + db[j]) for j in range(p_num)]
            B = simpson(arr2, x[0])
            arr3 = [p[k] * b[k] ** 2 for k in range(p_num)]
            bb = simpson(arr3, x[0])
            F = B**2 / F2 + bb
            return F
        elif btype == 3:
            I_tp = [np.real(dp[i] * dp[i] / p[i] ** 2) for i in range(p_num)]
            arr = [p[j]*(dp[j]*b[j]/p[j]+(1 + db[j]))**2 / (I_tp[j] + F_tp[j]) for j in range(p_num)]
            F = simpson(arr, x[0])
            return F
        else:
            raise ValueError("btype should be choosen in {1, 2, 3}.")
    else:
        #### multiparameter scenario ####
        if not b:
            b, db = [], []
            for i in range(para_num):
                b.append(np.zeros(len(x[i])))
                db.append(np.zeros(len(x[i])))
        elif not db:
            db = []
            for i in range(para_num):
                db.append(np.zeros(len(x[i])))

        p_shape, p_list, rho_list, drho_list, _, dp_ext = _build_param_grid(
            p, dp, rho, drho, para_num)
        dp_list = [dpi for dpi in dp_ext]
        b_list, db_list = _build_b_db_lists(b, db, para_num)

        if btype == 1:
            F_list = _allocate_F_list(para_num, len(p_list))
            for i in range(len(p_list)):
                F_tp = QFIM(rho_list[i], drho_list[i], LDtype=LDtype, eps=eps)
                F_inv = np.linalg.pinv(F_tp)
                B = np.diag([(1.0 + db_list[i][j]) for j in range(para_num)])
                term1 = B @ F_inv @ B
                term2 = np.dot(
                    np.array(b_list[i]).reshape(para_num, 1),
                    np.array(b_list[i]).reshape(1, para_num),
                )
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = term1[pj][pk] + term2[pj][pk]

            return _integrate_F_list(F_list, p, x, para_num, p_shape)
        elif btype == 2:
            F_list = _allocate_F_list(para_num, len(p_list))
            B_list = _allocate_F_list(para_num, len(p_list))
            bb_list = _allocate_F_list(para_num, len(p_list))
            for i in range(len(p_list)):
                F_tp = QFIM(rho_list[i], drho_list[i], LDtype=LDtype, eps=eps)
                B_tp = np.diag([(1.0 + db_list[i][j]) for j in range(para_num)])
                bb_tp = np.dot(
                    np.array(b_list[i]).reshape(para_num, 1),
                    np.array(b_list[i]).reshape(1, para_num),
                )
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tp[pj][pk]
                        B_list[pj][pk][i] = B_tp[pj][pk]
                        bb_list[pj][pk][i] = bb_tp[pj][pk]

            F_res = _integrate_F_list(F_list, p, x, para_num, p_shape)
            B_res = _integrate_F_list(B_list, p, x, para_num, p_shape)
            bb_res = _integrate_F_list(bb_list, p, x, para_num, p_shape)
            return B_res @ np.linalg.pinv(F_res) @ B_res + bb_res
        elif btype == 3:
            F_list = _allocate_F_list(para_num, len(p_list))
            for i in range(len(p_list)):
                F_tp = QFIM(rho_list[i], drho_list[i], LDtype=LDtype, eps=eps)
                I_tp = np.zeros((para_num, para_num))
                G_tp = np.zeros((para_num, para_num))
                for pm in range(para_num):
                    for pn in range(para_num):
                        if pm == pn:
                            G_tp[pm][pn] = dp_list[i][pn]*b_list[i][pm]/p_list[i]+(1.0 + db_list[i][pm])
                        else:
                            G_tp[pm][pn] = dp_list[i][pn]*b_list[i][pm]/p_list[i]
                        I_tp[pm][pn] = dp_list[i][pm] * dp_list[i][pn] / p_list[i] ** 2

                F_tot = G_tp @ np.linalg.pinv(F_tp + I_tp) @ G_tp.T
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tot[pj][pk]

            return _integrate_F_list(F_list, p, x, para_num, p_shape)
        else:
            raise ValueError("btype should be choosen in {1, 2, 3}.")


def VTB(x, p, dp, rho, drho, M=None, eps=1e-8):
    r"""
    Calculate the Van Trees bound (VTB), a Bayesian version of the Cramer-Rao bound.

    The covariance matrix with prior distribution $p(\textbf{x})$ is:

    $$
        \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\}) = \int p(\textbf{x}) \sum_y \mathrm{Tr}
        (\rho\Pi_y) (\hat{\textbf{x}}-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}
        \mathrm{d}\textbf{x}.
    $$

    The VTB is given by:

    $$
    \mathrm{cov} \geq \left(\mathcal{I}_{\mathrm{prior}} + \mathcal{I}_{\mathrm{Bayes}}\right)^{-1}.
    $$

    Symbols:  
        - $\mathcal{I}_{\mathrm{prior}} = \int p(\textbf{x}) \mathcal{I}_{p} \, \mathrm{d}\textbf{x}$ 
            is the classical Fisher information matrix (CFIM) for the prior distribution $p(\textbf{x})$.    
        - $\mathcal{I}_{\mathrm{Bayes}} = \int p(\textbf{x}) \mathcal{I} \, \mathrm{d}\textbf{x}$ 
            is the average CFIM over the prior.

    Args:
        x (list): 
            Parameter regimes for integration.
        p (np.array): 
            Prior distribution.
        dp (list): 
            Derivatives of the prior distribution with respect to the parameters.
        rho (list): 
            Parameterized density matrices.
        drho (list): 
            Derivatives of the density matrices with respect to the parameters.
        M (list, optional): 
            Positive operator-valued measure (POVM). Default is SIC-POVM.
        eps (float, optional): 
            Machine epsilon.

    Returns:
        (float/np.array): 
            For single parameter: float. For multiple parameters: matrix.

    Notes: 
        SIC-POVM uses Weyl-Heisenberg covariant fiducial states from 
        [http://www.physics.umb.edu/Research/QBism/solutions.html](http://www.physics.umb.edu/Research/QBism/solutions.html).
    """

    if M is None: M = []
    para_num = len(x)
    p_num = len(p)

    if para_num == 1:
        #### single parameter scenario ####
        M = _check_and_init_M(M, len(rho[0]))

        drho = _unwrap_single_param(drho, p_num)
        dp = _unwrap_single_param(dp, p_num)

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = CFIM(rho[m], [drho[m]], M=M, eps=eps)

        arr1 = [np.real(dp[i] * dp[i] / p[i]) for i in range(p_num)]
        I = simpson(arr1, x[0])
        arr2 = [np.real(F_tp[j] * p[j]) for j in range(p_num)]
        F = simpson(arr2, x[0])
        return 1.0 / (I + F)
    else:
        #### multiparameter scenario ####
        p_shape, p_list, rho_list, drho_list, _, dp_ext = _build_param_grid(
            p, dp, rho, drho, para_num)
        dp_list = [dpi for dpi in dp_ext]

        dim = len(rho_list[0])
        M = _check_and_init_M(M, dim)
        
        F_list = _allocate_F_list(para_num, len(p_list))
        I_list = _allocate_F_list(para_num, len(p_list))
        for i in range(len(p_list)):
            F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
            for pj in range(para_num):
                for pk in range(para_num):
                    F_list[pj][pk][i] = F_tp[pj][pk]
                    I_list[pj][pk][i] = (
                            dp_list[i][pj] * dp_list[i][pk] / p_list[i] ** 2
                        )

        F_res = _integrate_F_list(F_list, p, x, para_num, p_shape)
        I_res = _integrate_F_list(I_list, p, x, para_num, p_shape)
        return np.linalg.pinv(F_res + I_res)

def QVTB(x, p, dp, rho, drho, LDtype="SLD", eps=1e-8):
    r"""
    Calculate the quantum Van Trees bound (QVTB), a Bayesian version of the quantum Cramer-Rao bound.

    The covariance matrix with prior distribution $p(\textbf{x})$ is:

    $$
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\}) = \int p(\textbf{x}) \sum_y \mathrm{Tr}
    (\rho\Pi_y) (\hat{\textbf{x}}-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}
    \mathrm{d}\textbf{x}.
    $$

    The QVTB is given by:

    \$$
    \mathrm{cov} \geq \left(\mathcal{I}_{\mathrm{prior}} + \mathcal{F}_{\mathrm{Bayes}}\right)^{-1}.
    $$

    Symbols:
        - $\mathcal{I}_{\mathrm{prior}} = \int p(\textbf{x}) \mathcal{I}_{p} \, \mathrm{d}\textbf{x}$:  
            the classical Fisher information matrix (CFIM) for the prior distribution $p(\textbf{x})$.
        - $\mathcal{F}_{\mathrm{Bayes}} = \int p(\textbf{x}) \mathcal{F} \, \mathrm{d}\textbf{x}$:  
            the average quantum Fisher information matrix (QFIM) over the prior.

    Args:
        x (list): 
            Parameter regimes for integration.
        p (np.array): 
            Prior distribution.
        dp (list): 
            Derivatives of the prior distribution with respect to the parameters.
        rho (list): 
            Parameterized density matrices.
        drho (list): 
            Derivatives of the density matrices with respect to the parameters.
        LDtype (string, optional): 
            Type of logarithmic derivative (default: "SLD"). Options:  
                - "SLD": Symmetric logarithmic derivative.  
                - "RLD": Right logarithmic derivative.  
                - "LLD": Left logarithmic derivative.  
        eps (float, optional): 
            Machine epsilon.

    Returns: 
        (float/np.array): 
            For single parameter: float. For multiple parameters: matrix.
    """
    para_num = len(x)
    p_num = len(p)

    if para_num == 1:
        drho = _unwrap_single_param(drho, p_num)
        dp = _unwrap_single_param(dp, p_num)

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = QFIM(rho[m], [drho[m]], LDtype=LDtype, eps=eps)

        arr1 = [np.real(dp[i] * dp[i] / p[i]) for i in range(p_num)]
        I = simpson(arr1, x[0])
        arr2 = [np.real(F_tp[j] * p[j]) for j in range(p_num)]
        F = simpson(arr2, x[0])
        return 1.0 / (I + F)
    else:
        #### multiparameter scenario ####
        p_shape, p_list, rho_list, drho_list, _, dp_ext = _build_param_grid(
            p, dp, rho, drho, para_num)
        dp_list = [dpi for dpi in dp_ext]

        F_list = _allocate_F_list(para_num, len(p_list))
        I_list = _allocate_F_list(para_num, len(p_list))
        for i in range(len(p_list)):
            F_tp = QFIM(rho_list[i], drho_list[i], LDtype=LDtype, eps=eps)
            for pj in range(para_num):
                for pk in range(para_num):
                    F_list[pj][pk][i] = F_tp[pj][pk]
                    I_list[pj][pk][i] = (
                            dp_list[i][pj] * dp_list[i][pk] / p_list[i] ** 2
                        )

        F_res = _integrate_F_list(F_list, p, x, para_num, p_shape)
        I_res = _integrate_F_list(I_list, p, x, para_num, p_shape)
        return np.linalg.pinv(F_res + I_res)


def OBB_func(x, y, t, J, F):
    r"""ODE right-hand side for the optimal biased bound (OBB) differential equation.

    Solves the Euler-Lagrange equation for the optimal bias function $b(x)$:

    $$
    \frac{\mathrm{d}}{\mathrm{d}x}\begin{pmatrix}b\\ b'\end{pmatrix} =
    \begin{pmatrix}
    b' \\
    -J(x)\,b' + F(x)\,b - J(x)
    \end{pmatrix},
    $$

    where $F(x)$ is the QFI, $J(x) = p'(x)/p(x) - F'(x)/F(x)$, and
    $p(x)$ is the prior. This is used internally by :func:`OBB` via
    `scipy.integrate.solve_bvp`.

    Args:
        x (float): Evaluation point.
        y (numpy.ndarray): Current state $[b, b']$.
        t (numpy.ndarray): Grid of abscissa values.
        J (numpy.ndarray): Array of $J(x)$ values on the grid.
        F (numpy.ndarray): Array of $F(x)$ (QFI) values on the grid.

    Returns:
        numpy.ndarray: Derivative vector $[b', b'']$.

    See Also:
        :func:`boundary_condition`: Boundary-value problem condition.
        :func:`OBB`: Top-level OBB computation.
    """
    interp_J = interpolate.interp1d(t, (J))
    interp_F = interpolate.interp1d(t, (F))
    J_tp, F_tp = interp_J(x), interp_F(x)
    return np.vstack((y[1], -J_tp * y[1] + F_tp * y[0] - J_tp))


def boundary_condition(ya, yb):
    r"""Boundary condition for the OBB boundary-value problem (BVP).

    Imposes $b'(x_0) + 1 = 0$ at the left endpoint and
    $b'(x_{\mathrm{end}}) + 1 = 0$ at the right endpoint.
    These natural boundary conditions ensure the optimal bias satisfies
    $b'(x_{\mathrm{boundary}}) = -1$.

    Args:
        ya (numpy.ndarray): State at $x_0$: $[b(x_0), b'(x_0)]$.
        yb (numpy.ndarray): State at $x_{\mathrm{end}}$: $[b(x_{\mathrm{end}}), b'(x_{\mathrm{end}})]$.

    Returns:
        numpy.ndarray: Residual vector $[ya[1]+1, yb[1]+1]$.

    See Also:
        :func:`OBB_func`: ODE right-hand side.
        :func:`OBB`: Top-level OBB computation.
    """
    return np.array([ya[1] + 1.0, yb[1] + 1.0])


def OBB(x, p, dp, rho, drho, d2rho, LDtype="SLD", eps=1e-8):
    r"""
    Calculate the optimal biased bound (OBB) for single parameter estimation.

    The OBB is defined as:

    $$
    \mathrm{var}(\hat{x},\{\Pi_y\}) \geq \int p(x) \left( \frac{(1+b')^2}{F} + b^2 \right) \mathrm{d}x
    $$
    
    Symbols:
        - $b$: bias, $b'$: its derivative.
        - $F$: quantum Fisher information (QFI).

    This bound is solved using a boundary value problem approach.

    Args:
        x (np.array): 
            Parameter regime for integration.
        p (np.array): 
            Prior distribution.
        dp (np.array): 
            Derivative of the prior distribution with respect to the parameter.
        rho (list): 
            Parameterized density matrices.
        drho (list): 
            First derivatives of the density matrices with respect to the parameter.
        d2rho (list): 
            Second-order derivatives of the density matrices with respect to the parameter.
        LDtype (str, optional): 
            Type of logarithmic derivative (default: "SLD"). Options:  
                - "SLD": Symmetric logarithmic derivative.  
                - "RLD": Right logarithmic derivative.  
                - "LLD": Left logarithmic derivative.  
        eps (float, optional): 
            Machine epsilon.

    Returns: 
        (float): 
            The optimal biased bound value for single parameter estimation.

    Notes: 
        This function uses a boundary value problem solver to compute the optimal bias function.
    """

    #### single parameter scenario ####
    p_num = len(p)

    drho = _unwrap_single_param(drho, p_num)
    if isinstance(d2rho[0], list):
        d2rho = [d2rho[i][0] for i in range(p_num)]
    if isinstance(dp[0], list) or isinstance(dp[0], np.ndarray):
        dp = [dp[i][0] for i in range(p_num)]
    if not isinstance(x[0], (Number, np.floating, np.integer)):
        x = x[0]

    F, J = np.zeros(p_num), np.zeros(p_num)
    bias, dbias = np.zeros(p_num), np.zeros(p_num)
    for m in range(p_num):
        f, LD = QFIM(rho[m], [drho[m]], LDtype=LDtype, exportLD=True, eps=eps)
        F[m] = f
        term1 = d2rho[m] @ LD
        term2 = d2rho[m] @ LD.conj().T
        term3 = LD @ LD @ drho[m]
        dF = np.real(np.trace(term1 + term2 - term3))
        J[m] = dp[m] / p[m] - dF / f

    y_guess = np.zeros((2, x.size))
    fun = lambda m, n: OBB_func(m, n, x, J, F)
    result = solve_bvp(fun, boundary_condition, x, y_guess)
    res = result.sol(x)
    bias, dbias = res[0], res[1]

    value = [p[i] * ((1 + dbias[i]) ** 2 / F[i] + bias[i] ** 2) for i in range(p_num)]
    return simpson(value, x)
