from typing import Tuple
import abc
import numpy as np


class NeuralNet(metaclass=abc.ABCMeta):
    """
    Signature of all classes that can build Neural Nets
    """

    @abc.abstractmethod
    def build_context_name(self,
                           *args,
                           **kwargs) -> str:
        """
        A unique name given to teh network each time a build net is executed
        :params args: The unique name of the build context
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def network_architecture(self,
                             *args,
                             **kwargs) -> str:
        """
        The architecture of the Neural Network
        :params args: The architecture name
        """
        raise NotImplementedError()

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

    def load_and_train_from_json(self,
                                 filename: str) -> None:
        """
        Load the X,Y data from the given json file and train the modle
        :param filename: The file containing the X, Y data
        """
        raise NotImplementedError()

    def predict(self,
                x_value: float,
                *args,
                **kwargs) -> Tuple[np.float, np.float]:
        """
        :param x_value: The X value to predict Y for
        :params args: The arguments to parse for net compile parameters
        :param args:
        :param kwargs:
        :return: predicted Y and expected Y (based on actual function)
        """
        raise NotImplementedError()
