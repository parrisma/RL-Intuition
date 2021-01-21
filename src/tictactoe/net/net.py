import abc
import numpy as np


class NetI(metaclass=abc.ABCMeta):
    """
    Signature of all classes that can build Neural Nets
    """

    @abc.abstractmethod
    def build(self,
              *args,
              **kwargs) -> None:
        """
        Build a Neural Network based on the given arguments.
        :params args: The arguments to parse for net build parameters
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              *args,
              **kwargs) -> None:
        """
        Train the Neural Network based on the given arguments.
        :param x_train: The X Training data : shape (n,1)
        :param y_train: The Y Training data : shape (n,1)
        :params args: The arguments to parse for net compile parameters
        """
        raise NotImplementedError()
