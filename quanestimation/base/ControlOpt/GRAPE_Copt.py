import warnings
import quanestimation.base.ControlOpt.ControlStruct as Control
from quanestimation.base import QJL


class GRAPE_Copt(Control.ControlSystem):
    """
    Attributes
    ----------
    > **savefile:** `bool`
        -- Whether or not to save all the control coeffients.
        If set `True` then the control coefficients and the values of the
        objective function obtained in all episodes will be saved during
        the training. If set `False` the control coefficients in the final
        episode and the values of the objective function in all episodes
        will be saved.

    > **Adam:** `bool`
        -- Whether or not to use Adam for updating control coefficients.

    > **ctrl0:** `list of arrays`
        -- Initial guesses of control coefficients.

    > **max_episode:** `int`
        -- The number of episodes.

    > **epsilon:** `float`
        -- Learning rate.

    > **beta1:** `float`
        -- The exponential decay rate for the first moment estimates.

    > **beta2:** `float`
        -- The exponential decay rate for the second moment estimates.

    > **eps:** `float`
        -- Machine epsilon.

    > **load:** `bool`
        -- Whether or not to load control coefficients in the current location.
        If set `True` then the program will load control coefficients from
        "controls.csv" file in the current location and use it as the initial
        control coefficients.

    > **auto:** `bool`
        -- Whether or not to invoke automatic differentiation algorithm to evaluate
        the gradient. If set `True` then the gradient will be calculated with
        automatic differentiation algorithm otherwise it will be calculated
        using analytical method.
    """

    def __init__(
        self,
        savefile=False,
        Adam=True,
        ctrl0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        eps=1e-8,
        seed=1234,
        load=False,
        auto=True,
    ):
        r"""
        Initialize GRAPE (GRadient Ascent Pulse Engineering) control optimization.

        Supports both standard GRAPE and automatic-differentiation GRAPE (auto-GRAPE)
        with optional Adam optimizer.

        Args:
            savefile (bool, optional): Whether to save all control coefficients during
                training. Default ``False``.
            Adam (bool, optional): Whether to use the Adam optimizer. Default ``True``.
            ctrl0 (list of arrays, optional): Initial guesses of control coefficients.
                Default ``[]``.
            max_episode (int, optional): Number of training episodes. Default 300.
            epsilon (float, optional): Learning rate. Default 0.01.
            beta1 (float, optional): Exponential decay rate for first moment
                estimates (Adam). Default 0.90.
            beta2 (float, optional): Exponential decay rate for second moment
                estimates (Adam). Default 0.99.
            eps (float, optional): Machine epsilon. Default 1e-8.
            seed (int, optional): Random seed. Default 1234.
            load (bool, optional): Whether to load initial control coefficients
                from ``controls.csv``. Default ``False``.
            auto (bool, optional): Whether to use automatic differentiation.
                Default ``True``.
        """

        Control.ControlSystem.__init__(self, savefile, ctrl0, seed, eps, load)

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0
        self.auto = auto

    def QFIM(self, W=None, LDtype="SLD"):
        r"""
        Choose QFI or $\mathrm{Tr}(WF^{-1})$ as the objective function.
        In single parameter estimation the objective function is QFI and in
        multiparameter estimation it will be $\mathrm{Tr}(WF^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **LDtype:** `string`
            -- Types of QFI (QFIM) can be set as the objective function. Options are:
            "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).
            "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).
            "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).
        """

        if W is None:
            W = []

        if self.auto:
            if self.Adam:
                self.alg = QJL.autoGRAPE(
                    Adam=True,
                    max_episode=self.max_episode,
                    epsilon=self.epsilon,
                    beta1=self.beta1,
                    beta2=self.beta2,
                )
            else:
                self.alg = QJL.autoGRAPE(
                    Adam=False, max_episode=self.max_episode, epsilon=self.epsilon
                )
        else:
            if (len(self.tspan) - 1) != len(self.control_coefficients[0]):
                warnings.warn(
                    "GRAPE is not available when the length of each control is not \
                               equal to the length of time, and is replaced by auto-GRAPE.",
                    DeprecationWarning,
                )
                #### call autoGRAPE automatically ####
                if self.Adam:
                    self.alg = QJL.autoGRAPE(
                        Adam=True,
                        max_episode=self.max_episode,
                        epsilon=self.epsilon,
                        beta1=self.beta1,
                        beta2=self.beta2,
                    )
                else:
                    self.alg = QJL.autoGRAPE(
                        Adam=False, max_episode=self.max_episode, epsilon=self.epsilon
                    )
            else:
                if LDtype == "SLD":
                    if self.Adam:
                        self.alg = QJL.GRAPE(
                            Adam=True,
                            max_episode=self.max_episode,
                            epsilon=self.epsilon,
                            beta1=self.beta1,
                            beta2=self.beta2,
                        )
                    else:
                        self.alg = QJL.GRAPE(
                            Adam=False,
                            max_episode=self.max_episode,
                            epsilon=self.epsilon,
                        )
                else:
                    raise ValueError("GRAPE is only available when LDtype is SLD.")

        super().QFIM(W, LDtype)

    def CFIM(self, M=None, W=None):
        r"""
        Choose CFI or $\mathrm{Tr}(WI^{-1})$ as the objective function.
        In single parameter estimation the objective function is CFI and
        in multiparameter estimation it will be $\mathrm{Tr}(WI^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **M:** `list of matrices`
            -- A set of positive operator-valued measure (POVM). The default measurement
            is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

        **Note:**
            SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state
            which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
            solutions.html).
        """

        if M is None:
            M = []
        if W is None:
            W = []

        if self.auto:
            if self.Adam:
                self.alg = QJL.autoGRAPE(
                    Adam=True,
                    max_episode=self.max_episode,
                    epsilon=self.epsilon,
                    beta1=self.beta1,
                    beta2=self.beta2,
                )
            else:
                self.alg = QJL.autoGRAPE(
                    Adam=False, max_episode=self.max_episode, epsilon=self.epsilon
                )
        else:
            if (len(self.tspan) - 1) != len(self.control_coefficients[0]):
                warnings.warn(
                    "GRAPE is not available when the length of each control is not \
                               equal to the length of time, and is replaced by auto-GRAPE.",
                    DeprecationWarning,
                )
                #### call autoGRAPE automatically ####
                if self.Adam:
                    self.alg = QJL.autoGRAPE(
                        Adam=True,
                        max_episode=self.max_episode,
                        epsilon=self.epsilon,
                        beta1=self.beta1,
                        beta2=self.beta2,
                    )
                else:
                    self.alg = QJL.autoGRAPE(
                        Adam=False, max_episode=self.max_episode, epsilon=self.epsilon
                    )
            else:
                if self.Adam:
                    self.alg = QJL.GRAPE(
                        Adam=True,
                        max_episode=self.max_episode,
                        epsilon=self.epsilon,
                        beta1=self.beta1,
                        beta2=self.beta2,
                    )
                else:
                    self.alg = QJL.GRAPE(
                        Adam=False, max_episode=self.max_episode, epsilon=self.epsilon
                    )

        super().CFIM(M, W)

    def HCRB(self, W=None):
        """
        GRAPE and auto-GRAPE are not available when the objective function is HCRB.
        Supported methods are PSO, DE and DDPG.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """
        if W is None:
            W = []
        raise ValueError(
            "GRAPE and auto-GRAPE are not available when the objective function is HCRB. Supported methods are 'PSO', 'DE' and 'DDPG'.",
        )

    def mintime(self, f, W=None, M=None, method="binary", target="QFIM", LDtype="SLD"):
        """
        Search of the minimum time to reach a given value of the objective function.

        Parameters
        ----------
        > **f:** `float`
            -- The given value of the objective function.

        > **W:** `matrix`
            -- Weight matrix.

        > **M:** `list of matrices`
            -- A set of positive operator-valued measure (POVM). The default measurement
            is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

        > **method:** `string`
            -- Methods for searching the minimum time to reach the given value of the
            objective function. Options are:
            "binary" (default) -- Binary search (logarithmic search).
            "forward" -- Forward search from the beginning of time.

        > **target:** `string`
            -- Objective functions for searching the minimum time to reach the given
            value of the objective function. Options are:
            "QFIM" (default) -- Choose QFI (QFIM) as the objective function.
            "CFIM" -- Choose CFI (CFIM) as the objective function.
            "HCRB" -- Choose HCRB as the objective function.

        > **LDtype:** `string`
            -- Types of QFI (QFIM) can be set as the objective function. Options are:
            "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).
            "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).
            "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).

        **Note:**
            SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state
            which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
            solutions.html).
        """

        if W is None:
            W = []
        if M is None:
            M = []

        if target == "HCRB":
            raise ValueError(
                "GRAPE and auto-GRAPE are not available when the objective function is HCRB. Supported methods are 'PSO', 'DE' and 'DDPG'.",
            )
        if self.auto:
            if self.Adam:
                self.alg = QJL.autoGRAPE(
                    Adam=True,
                    max_episode=self.max_episode,
                    epsilon=self.epsilon,
                    beta1=self.beta1,
                    beta2=self.beta2,
                )
            else:
                self.alg = QJL.autoGRAPE(
                    Adam=False, max_episode=self.max_episode, epsilon=self.epsilon
                )
        else:
            if self.Adam:
                self.alg = QJL.GRAPE(
                    Adam=True,
                    max_episode=self.max_episode,
                    epsilon=self.epsilon,
                    beta1=self.beta1,
                    beta2=self.beta2,
                )
            else:
                self.alg = QJL.GRAPE(
                    Adam=False, max_episode=self.max_episode, epsilon=self.epsilon
                )

        super().mintime(f, W, M, method, target, LDtype)
