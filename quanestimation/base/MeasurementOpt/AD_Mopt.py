from quanestimation.base import QJL
import quanestimation.base.MeasurementOpt.MeasurementStruct as Measurement


class AD_Mopt(Measurement.MeasurementSystem):
    """
    Attributes
    ----------
    > **savefile:** `bool`
        -- Whether or not to save all the measurements.  
        If set `True` then the measurements and the values of the 
        objective function obtained in all episodes will be saved during 
        the training. If set `False` the measurement in the final 
        episode and the values of the objective function in all episodes 
        will be saved.

    > **Adam:** `bool`
        -- Whether or not to use Adam for updating measurements.

    > **measurement0:** `list of arrays`
        -- Initial guesses of measurements.

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
        -- Whether or not to load measurements in the current location.  
        If set `True` then the program will load measurement from "measurements.csv"
        file in the current location and use it as the initial measurement.
    """

    def __init__(
        self,
        mtype,
        minput,
        savefile=False,
        Adam=False,
        measurement0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        seed=1234,
        eps=1e-8,
        load=False,
    ):
        r"""
        Initialize automatic differentiation (AD) for measurement optimization.

        Supports both gradient descent and Adam optimizer. Note: AD is only
        available for ``mtype="input"`` (not for projective measurements).

        Args:
            mtype (str): Measurement optimization scenario. ``"projection"`` or
                ``"input"``.
            minput (list): Additional input parameters for the ``"input"``
                scenario.
            savefile (bool, optional): Whether to save all measurements during
                training. Default ``False``.
            Adam (bool, optional): Whether to use the Adam optimizer.
                Default ``False``.
            measurement0 (list of arrays, optional): Initial guesses of
                measurements. Default ``[]``.
            max_episode (int, optional): Number of training episodes.
                Default 300.
            epsilon (float, optional): Learning rate. Default 0.01.
            beta1 (float, optional): Exponential decay rate for first moment
                estimates (Adam). Default 0.90.
            beta2 (float, optional): Exponential decay rate for second moment
                estimates (Adam). Default 0.99.
            seed (int, optional): Random seed. Default 1234.
            eps (float, optional): Machine epsilon. Default 1e-8.
            load (bool, optional): Whether to load initial measurements from
                ``measurements.csv``. Default ``False``.
        """

        Measurement.MeasurementSystem.__init__(
            self, mtype, minput, savefile, measurement0, seed, eps, load 
        )

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0
        self.seed = seed

        if self.Adam:
            self.alg = QJL.AD(
                Adam=True,
                max_episode=self.max_episode,
                epsilon=self.epsilon,
                beta1=self.beta1,
                beta2=self.beta2,
            )
        else:
            self.alg = QJL.AD(
                Adam=False,
                max_episode=self.max_episode,
                epsilon=self.epsilon,
            )

    def CFIM(self, W=None):
        r"""
        Choose CFI or $\mathrm{Tr}(WI^{-1})$ as the objective function. 
        In single parameter estimation the objective function is CFI and 
        in multiparameter estimation it will be $\mathrm{Tr}(WI^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """

        if W is None:
            W = []

        if self.mtype == "projection":
            raise ValueError(
                "AD is not available when mtype is projection. Supported methods are 'PSO' and 'DE'.",
            )
        else:
            super().CFIM(W)
