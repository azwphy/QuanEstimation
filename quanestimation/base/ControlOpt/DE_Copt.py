import juliacall
from quanestimation.base import QJL
import quanestimation.base.ControlOpt.ControlStruct as Control


class DE_Copt(Control.ControlSystem):
    """
    Attributes
    ----------
    > **savefile:** `bool`
        --Whether or not to save all the control coeffients.
        If set `True` then the control coefficients and the values of the
        objective function obtained in all episodes will be saved during
        the training. If set `False` the control coefficients in the final
        episode and the values of the objective function in all episodes
        will be saved.

    > **p_num:** `int`
        -- The number of populations.

    > **ctrl0:** list of arrays
        -- Initial guesses of control coefficients.

    > **max_episode:** `int`
        -- The number of episodes.

    > **c:** `float`
        -- Mutation constant.

    > **cr:** `float`
        -- Crossover constant.

    > **seed:** `int`
        -- Random seed.

    > **eps:** `float`
        -- Machine epsilon.

    > **load:** `bool`
        -- Whether or not to load control coefficients in the current location.
        If set `True` then the program will load control coefficients from
        "controls.csv" file in the current location and use it as the initial
        control coefficients.
    """

    def __init__(
        self,
        savefile=False,
        p_num=10,
        ctrl0=[],
        max_episode=1000,
        c=1.0,
        cr=0.5,
        seed=1234,
        eps=1e-8,
        load=False,
    ):
        r"""
        Initialize differential evolution (DE) for control optimization.

        Args:
            savefile (bool, optional): Whether to save all control coefficients
                during training. Default ``False``.
            p_num (int, optional): Number of populations. Default 10.
            ctrl0 (list of arrays, optional): Initial guesses of control
                coefficients. Default ``[]``.
            max_episode (int, optional): Number of episodes. Default 1000.
            c (float, optional): Mutation constant. Default 1.0.
            cr (float, optional): Crossover constant. Default 0.5.
            seed (int, optional): Random seed. Default 1234.
            eps (float, optional): Machine epsilon. Default 1e-8.
            load (bool, optional): Whether to load initial control coefficients
                from ``controls.csv``. Default ``False``.
        """

        Control.ControlSystem.__init__(self, savefile, ctrl0, seed, eps, load)

        self.p_num = p_num
        self.max_episode = max_episode
        self.c = c
        self.cr = cr

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

        ini_population = (QJL.Vector([self.ctrl0]),)
        self.alg = QJL.DE(
            max_episode=self.max_episode,
            p_num=self.p_num,
            ini_population=ini_population,
            c=self.c,
            cr=self.cr,
        )

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

        ini_population = (QJL.Vector([self.ctrl0]),)
        self.alg = QJL.DE(
            max_episode=self.max_episode,
            p_num=self.p_num,
            ini_population=ini_population,
            c=self.c,
            cr=self.cr,
        )

        super().CFIM(M, W)

    def HCRB(self, W=None):
        """
        Choose HCRB as the objective function.

        **Note:** in single parameter estimation, HCRB is equivalent to QFI, please choose
        QFI as the objective function.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """

        if W is None:
            W = []

        ini_population = (QJL.Vector([self.ctrl0]),)
        self.alg = QJL.DE(
            max_episode=self.max_episode,
            p_num=self.p_num,
            ini_population=ini_population,
            c=self.c,
            cr=self.cr,
        )

        super().HCRB(W)

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
            value of the objective function. Options are:<br>
            "QFIM" (default) -- Choose QFI (QFIM) as the objective function.<br>
            "CFIM" -- Choose CFI (CFIM) as the objective function.<br>
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

        ini_population = (QJL.Vector([self.ctrl0]),)
        self.alg = QJL.DE(
            max_episode=self.max_episode,
            p_num=self.p_num,
            ini_population=ini_population,
            c=self.c,
            cr=self.cr,
        )

        super().mintime(f, W, M, method, target, LDtype)
