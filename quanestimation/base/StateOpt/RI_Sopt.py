from quanestimation.base import QJL
import quanestimation.base.StateOpt.StateStruct as State


class RI_Sopt(State.StateSystem):
    """
    Attributes
    ----------
    > **savefile:**  `bool`
        -- Whether or not to save all the states.  
        If set `True` then the states and the values of the 
        objective function obtained in all episodes will be saved during 
        the training. If set `False` the state in the final 
        episode and the values of the objective function in all episodes 
        will be saved.

    > **psi0:** `list of arrays`
        -- Initial guesses of states.

    > **max_episode:** `int`
        -- The number of episodes.

    > **seed:** `int`
        -- Random seed.

    > **eps:** `float`
        -- Machine epsilon.

    > **load:** `bool`
        -- Whether or not to load states in the current location.  
        If set `True` then the program will load state from "states.csv"
        file in the current location and use it as the initial state.
    """

    def __init__(
        self,
        savefile=False,
        psi0=[],
        max_episode=300,
        seed=1234,
        eps=1e-8,
        load=False,
    ):
        r"""
        Initialize random iteration (RI) for state optimization.

        Args:
            savefile (bool, optional): Whether to save all states during
                training. Default ``False``.
            psi0 (list of arrays, optional): Initial guesses of probe states.
                Default ``[]``.
            max_episode (int, optional): Number of episodes. Default 300.
            seed (int, optional): Random seed. Default 1234.
            eps (float, optional): Machine epsilon. Default 1e-8.
            load (bool, optional): Whether to load initial states from
                ``states.csv``. Default ``False``.
        """

        State.StateSystem.__init__(self, savefile, psi0, seed, eps, load)

        self.max_episode = max_episode
        self.seed = seed

    def QFIM(self, W=None, LDtype="SLD"):
        r"""
        Choose QFI as the objective function. 

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **LDtype:** `string`
            -- Types of QFI (QFIM) can be set as the objective function. Only SLD can
            is available here.
        """
        if W is None:
            W = []
        self.alg = QJL.RI(
            max_episode=self.max_episode,
        )
        if self.dynamics_type != "Kraus":
            raise ValueError("Only the parameterization with Kraus operators is available.")
        
        if LDtype == "SLD":
            super().QFIM(W, LDtype)
        else:
            raise ValueError("Only SLD is available.")

    def CFIM(self, M=None, W=None):
        """
        Choose CFIM as the objective function. 

        **Note:** CFIM is not available.

        Parameters
        ----------
        > **M:** `list`
            -- POVM.
            
        > **W:** `matrix`
            -- Weight matrix.
        """
        if M is None:
            M = []
        if W is None:
            W = []
        raise ValueError("CFIM is not available.")

    def HCRB(self, W=None):
        """
        Choose HCRB as the objective function. 

        **Note:** Here HCRB is not available.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """
        if W is None:
            W = []
        raise ValueError("HCRB is not available.")
