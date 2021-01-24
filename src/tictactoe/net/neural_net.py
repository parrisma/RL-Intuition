from typing import Tuple, Any, List
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
              x_test: np.ndarray,
              y_test: np.ndarray,
              *args,
              **kwargs) -> None:
        """
        :param x_train: The X Training data : shape (n,1)
        :param y_train: The Y Training data : shape (n,1)
        :param x_test: The X Test data : shape (n,1)
        :param y_test: The Y Test (actual) data : shape (n,1)
        :params args: The arguments to parse for net compile parameters
        :param args:
        :param kwargs:
        :return:
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

    @staticmethod
    def kwargs_get(arg_name: str,
                   default: Any,
                   expected_type: type,
                   **kwargs) -> Any:
        """
        Get the requested arg name from kwargs
        :param arg_name: The name of teh argument to find in kwargs
        :param default: The default value if arg_name not in kwargs
        :param expected_type: The expected type of the argument
        :param kwargs: The kwargs to search
        :return: The argument as expected_type or None if arg is there and types does not match
        """
        arg = kwargs.get(arg_name, None)
        if arg is None:
            arg = default
        try:
            arg = expected_type(arg)
        except Exception as _:
            arg = None
        return arg

    @staticmethod
    def exp_steps(last_step: int) -> List[int]:
        """
        Create a list of 10 steps with step size exponential in range 0 to last_step
        :param last_step: The last step of the sequence
        :return: List of steps exponentially spaced
        """
        steps = list()
        interval = 0.8325545 / (last_step - 1)
        for i in NeuralNet.ten_intervals(last_step):
            step = np.round((np.exp(np.power((i * interval), 2)) - 1) * last_step)
            if step not in steps:
                steps.append(step)
        steps = [int(last_step) if x > last_step else int(x) for x in steps]
        return steps

    @staticmethod
    def ten_intervals(max_value: int) -> List[int]:
        """
        A list of intervals dividing the max_value into 10 slots
        :param max_value: The value to calculate interval for
        :return: the interval
        """
        res = list()
        if max_value < 10:
            res = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif max_value < 20:
            res = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20]
        if max_value > 10:
            res = list(range(1, max_value, int(np.trunc(max_value / 9))))
            res[-1] = max_value
        return res
