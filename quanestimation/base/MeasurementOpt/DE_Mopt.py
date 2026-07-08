from quanestimation.base import QJL
import quanestimation.base.MeasurementOpt.MeasurementStruct as Measurement


class DE_Mopt(Measurement.MeasurementSystem):
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

    > **p_num:** `int`
        -- The number of populations.

    > **measurement0:** `list of arrays`
        -- Initial guesses of measurements.

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
        -- Whether or not to load measurements in the current location.
        If set `True` then the program will load measurement from "measurements.csv"
        file in the current location and use it as the initial measurement.
    """

    def __init__(
        self,
        mtype,
        minput,
        savefile=False,
        p_num=10,
        measurement0=[],
        max_episode=1000,
        c=1.0,
        cr=0.5,
        seed=1234,
        eps=1e-8,
        load=False,
    ):
        r"""
        Initialize differential evolution (DE) for measurement optimization.

        Args:
            mtype (str): Measurement optimization scenario. ``"projection"`` or
                ``"input"``.
            minput (list): Additional input parameters for the ``"input"``
                scenario.
            savefile (bool, optional): Whether to save all measurements during
                training. Default ``False``.
            p_num (int, optional): Number of populations. Default 10.
            measurement0 (list of arrays, optional): Initial guesses of
                measurements. Default ``[]``.
            max_episode (int, optional): Number of episodes. Default 1000.
            c (float, optional): Mutation constant. Default 1.0.
            cr (float, optional): Crossover constant. Default 0.5.
            seed (int, optional): Random seed. Default 1234.
            eps (float, optional): Machine epsilon. Default 1e-8.
            load (bool, optional): Whether to load initial measurements from
                ``measurements.csv``. Default ``False``.
        """

        Measurement.MeasurementSystem.__init__(
            self, mtype, minput, savefile, measurement0, seed, eps, load
        )

        self.p_num = p_num
        self.max_episode = max_episode
        self.c = c
        self.cr = cr

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

        ini_population = (self.measurement0,)
        self.alg = QJL.DE(
            max_episode=self.max_episode,
            p_num=self.p_num,
            ini_population=ini_population,
            c=self.c,
            cr=self.cr,
        )
        super().CFIM(W)
