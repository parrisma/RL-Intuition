import abc
import numpy as np
import tensorflow as tf
from typing import Tuple, Any, List
from os import path, mkdir
from src.lib.namegen.namegen import NameGen
from src.lib.rltrace.trace import Trace


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

    @abc.abstractmethod
    def _test(self,
              x_test: np.ndarray,
              y_test: np.ndarray,
              *args,
              **kwargs) -> float:
        """
        :param x_test: The X Test data : shape (n,1)
        :param y_test: The Y Test (actual) data : shape (n,1)
        :param args: The arguments to parse for net compile parameters
        :param kwargs:
        :return: overall mean squared error between test and predictions
        """
        raise NotImplementedError()

    @abc.abstractmethod
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

    @property
    @abc.abstractmethod
    def train_context_name(self) -> str:
        """
        The name of the training context
        :return: The name of the training context for the current training session
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _directory_to_use(self) -> str:
        """
        The name of the root directory in which all model related files and data are stored.
        :return: The name of the training context for the current training session
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def trace(self) -> Trace:
        """
        The trace logger
        :return: The trace logger
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def model(self) -> tf.keras.Model:
        """
        The tensorflow model
        :return: The tensorflow model
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """
        The name of the model for saving and loading
        :return: The name of the model
        """
        raise NotImplementedError()

    @property
    def summary_file(self) -> str:
        """
        The name of the training summary file.
        :return: The full name and path of the training summary file.
        """
        return "{}\\summary.json".format(self._directory_to_use)

    @property
    def hyper_params_file(self) -> str:
        """
        The name of the hyper parameters file
        :return: The full name and path of the hyper parameters file.
        """
        return "{}\\hyper-parameters.json".format(self._directory_to_use)

    @property
    def model_checkpoint_file(self) -> str:
        """
        The file pattern to use for saving model checkpoints from 'fit' Callback
        :return: The file pattern to use for saving model checkpoints from 'fit' Callback
        """
        return '{}\\weights.{}.hdf5'.format(self._directory_to_use, '{epoch:02d}')

    @property
    def model_file(self) -> str:
        """
        The full path and name of where to save/load the model
        Based on the training context and the model name
        :return: The full path and name of where to save/load the model
        """
        return '{}\\{}'.format(self._directory_to_use, self.model_name)

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

    @staticmethod
    def new_context(base_dir_to_use: str) -> Tuple[str, str]:
        """
        Create a new context name and save area for mode build/run/test
        :param base_dir_to_use: The base directory in which to create the contact
        :return: the name of the new context and its directory 'to use'
        """
        context_name = NameGen.generate_random_name()
        context_dir = "{}\\{}".format(base_dir_to_use, context_name)
        while path.exists(context_dir):
            context_name = NameGen.generate_random_name()
            context_dir = "{}\\{}".format(base_dir_to_use, context_name)
        mkdir(context_dir)
        return context_dir, context_name

    @abc.abstractmethod
    def _load_from_json(self,
                        filename: str,
                        test_split: float = 0.2) -> List[np.ndarray]:
        """
        Load XY training data from given JSON file
        :param filename: The JSON file name with the training data in
        :param test_split: The % (0.0 tp 1.0) of the training data to use as test data, default = 20% (0.2)
        :return: x_train, y_train, x_test, y_test
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _dump_summary_to_json(self,
                              filename: str) -> None:
        """
        Dump the summary of model build and test to given file
        :param filename: The full path and filename to dump summary to as JSON
        """
        raise NotImplementedError

    def load_and_train_from_json(self,
                                 filename: str) -> None:
        """
        Load the X,Y data from the given json file and train the modle
        :param filename: The file containing the X, Y data
        """
        if self.model is not None:
            try:
                x_train, y_train, x_test, y_test = self._load_from_json(filename)
                self.trace.log().info("Training Starts [{}]".format(self.train_context_name))
                self.train(x_train, y_train, x_test, y_test)
                self.trace.log().info("Training Ends [{}]".format(self.train_context_name))
                self.trace.log().info("Testing Starts [{}]".format(self.train_context_name))
                self._test(x_test, y_test)
                self.trace.log().info("Testing Ends [{}]".format(self.train_context_name))
                self._dump_summary_to_json(self.summary_file)
                self.trace.log().info("Saving the trained model [{}]".format(self.model_name))
                self.model.save(self.model_file)
                self.trace.log().info("Saved Train & Test results as json [{}]".format(self.summary_file))
            except Exception as e:
                self.trace.log().info("Failed to load data to train model [{}]".format(str(e)))
        else:
            self.trace.log().info("Model not built, cannot load and train")
        return
