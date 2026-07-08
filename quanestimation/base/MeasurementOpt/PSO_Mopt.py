from quanestimation.base import QJL
import quanestimation.base.MeasurementOpt.MeasurementStruct as Measurement


class PSO_Mopt(Measurement.MeasurementSystem):
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
        -- The number of particles.

    > **measurement0:** `list of arrays`
        -- Initial guesses of measurements.

    > **max_episode:** `int or list`
        -- If it is an integer, for example max_episode=1000, it means the 
        program will continuously run 1000 episodes. However, if it is an
        array, for example max_episode=[1000,100], the program will run 
        1000 episodes in total but replace measurements of all  the particles 
        with global best every 100 episodes.
  
    > **c0:** `float`
        -- The damping factor that assists convergence, also known as inertia weight.

    > **c1:** `float`
        -- The exploitation weight that attracts the particle to its best previous 
        position, also known as cognitive learning factor.

    > **c2:** `float`
        -- The exploitation weight that attracts the particle to the best position  
        in the neighborhood, also known as social learning factor.

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
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        eps=1e-8,
        load=False,
    ):
        r"""
        Initialize particle swarm optimization for measurement optimization.

        Args:
            mtype (str): Measurement optimization scenario. ``"projection"`` or
                ``"input"``.
            minput (list): Additional input parameters for the ``"input"``
                scenario.
            savefile (bool, optional): Whether to save all measurements during
                training. Default ``False``.
            p_num (int, optional): Number of particles. Default 10.
            measurement0 (list of arrays, optional): Initial guesses of
                measurements. Default ``[]``.
            max_episode (int or list, optional): Number of episodes. If a list
                ``[total, reset_interval]``, particles are reset to the global
                best every ``reset_interval`` episodes. Default ``[1000, 100]``.
            c0 (float, optional): Inertia weight (damping factor).
                Default 1.0.
            c1 (float, optional): Cognitive learning factor. Default 2.0.
            c2 (float, optional): Social learning factor. Default 2.0.
            seed (int, optional): Random seed. Default 1234.
            eps (float, optional): Machine epsilon. Default 1e-8.
            load (bool, optional): Whether to load initial measurements from
                ``measurements.csv``. Default ``False``.
        """

        Measurement.MeasurementSystem.__init__(
            self, mtype, minput, savefile, measurement0, seed, eps, load
        )

        self.p_num = p_num
        is_int = isinstance(max_episode, int)
        self.max_episode = max_episode if is_int else QJL.Vector[QJL.Int64](max_episode)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

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

        ini_particle = (self.measurement0,)
        self.alg = QJL.PSO(
            max_episode=self.max_episode,
            p_num=self.p_num,
            ini_particle=ini_particle,
            c0=self.c0,
            c1=self.c1,
            c2=self.c2,
        )
        
        super().CFIM(W)
