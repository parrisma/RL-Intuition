import abc

import numpy as np


class State(metaclass=abc.ABCMeta):
    """
    An immutable representation of an environment state
    """

    @abc.abstractmethod
    def state(self) -> object:
        """
        An environment specific representation for Env. State
        :return: state as object
        """
        pass

    #
    #
    #
    @abc.abstractmethod
    def state_as_string(self) -> str:
        """
        The State rendered as string
        :return: A string representation of the current state
        """
        pass

    @abc.abstractmethod
    def state_model_input(self) -> np.ndarray:
        """
        The State object rendered in numerical numpy.ndarray form compatible with the X input of a neural network.
        Each board position is the numerical player id or zero if the position is empty
        :return: The state as (1,9) numpy.ndarray
        """
        pass

    def state_as_visualisation(self) -> str:
        """
        An easy to read visualisation of the state object for print and debug
        :return: An easy to read string form of the current state
        """
        pass
