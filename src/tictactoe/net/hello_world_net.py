from typing import List, Dict, Callable, Tuple
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from src.lib.rltrace.trace import Trace
from src.tictactoe.net.neural_net import NeuralNet
from src.tictactoe.util.vals_json import ValsJson
from src.tictactoe.net.hyper_params import HyperParams
from src.tictactoe.net.names import Names
from src.tictactoe.net.lr_exponential_decay import LRExponentialDecay


class HelloWorldNet(NeuralNet):
    """
    Build a fully connected NN for univariate prediction
    """
    _model: tf.keras.models.Sequential
    _summary: Dict[str, List[float]]
    _train_context_name: str
    _trace: Trace
    _base_dir_to_use: str
    _dir_to_use: str
    _actual_func: Callable[[float], float]
    _x_min: float
    _x_max: float
    _hyper_params: HyperParams

    class TestCallback(tf.keras.callbacks.Callback):
        """
        Call back to dump interim test details while training
        """

        def __init__(self,
                     trace: Trace,
                     x_test: np.ndarray,
                     y_test: np.ndarray,
                     num_epoch: int,
                     summary: Dict[str, List[float]],
                     summary_file_root: str):
            super().__init__()
            self._trace = trace
            self._x_test = x_test
            self._y_test = y_test
            self._interim_steps = NeuralNet.exp_steps(num_epoch)
            self._summary = summary
            self.summary_file_root = summary_file_root
            return

        def on_epoch_end(self, epoch, logs=None):
            """
            Run and save predictions if epoch in the list of interim epochs to report on
            :param epoch: The current epoch
            :param logs: n/a
            """
            if epoch in self._interim_steps:
                self._trace.log().info("Learning rate {:12.11f}".format(tf.keras.backend.eval(self.model.optimizer.lr)))
                self._trace.log().info("Run interim prediction at epoch {}".format(epoch))
                predictions = self.model.predict(self._x_test)
                predictions = predictions.reshape(len(predictions))
                self._summary["{}_{:04d}".format(self.summary_file_root, epoch)] = predictions.tolist()
                return

    def __init__(self,
                 trace: Trace,
                 dir_to_use: str):
        self._trace = trace
        self._base_dir_to_use = dir_to_use
        self._dir_to_use, self._train_context_name = self.new_context(self._base_dir_to_use)
        self._hyper_params = HyperParams()
        self._set_hyper_params()
        self._model = None  # noqa
        self._summary = None  # noqa
        self._x_min = None  # noqa
        self._x_max = None  # noqa
        self._reset()
        return

    def _reset(self) -> None:
        """
        Clear model status after rebuild
        """
        self._model = None  # noqa
        self._summary = dict()
        self._actual_func = np.sin
        self._dir_to_use, self._train_context_name = self.new_context(self._base_dir_to_use)
        return

    def _set_hyper_params(self) -> None:
        """
        Set the Hyper Parameters for the Hello World Net
        """
        self._hyper_params.set(Names.learning_rate, 0.001, "Adam Optimizer Initial learning rate")
        self._hyper_params.set(Names.num_epoch, 500, "Number of training epochs")
        self._hyper_params.set(Names.batch_size, 8, "Number of samples per training batch")
        return

    def build_context_name(self,
                           *args,
                           **kwargs) -> str:
        """
        A unique name given to the network each time a build net is executed
        :params args: The unique name of the build context
        """
        return self._train_context_name

    def network_architecture(self,
                             *args,
                             **kwargs) -> str:
        """
        The architecture of the Neural Network
        :params args: The architecture name
        """
        return self.__class__.__name__

    def build(self,
              *args,
              **kwargs) -> None:
        """
        Build a Neural Network based on the given arguments.
        :params args: The arguments to parse for net build parameters
        """
        self._reset()

        self._model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(input_shape=(1,), units=1, name='input'),
                tf.keras.layers.Dense(128, activation=tf.nn.relu, name='dense1'),
                tf.keras.layers.Dense(256, activation=tf.nn.relu, name='dense2'),
                tf.keras.layers.Dropout(.1, name='dropout-1-10pct'),
                tf.keras.layers.Dense(64, activation=tf.nn.relu, name='dense3'),
                tf.keras.layers.Dense(1, name='output')
            ]
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self._hyper_params.get(Names.learning_rate))
        self._model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.mean_squared_error)

        self._hyper_params.save_to_json(self.hyper_params_file)
        # ToDo save model weights to allow reset on new training

        return

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
        if self._model is None:
            self._trace.log().info("Build model before training")

        self._x_min = np.min(x_train)
        self._x_max = np.max(x_train)

        self._trace.log().info("Start training: context [{}]".format(self._train_context_name))
        num_epochs = self._hyper_params.get(Names.num_epoch)
        history = self._model.fit(x_train, y_train,
                                  epochs=num_epochs,
                                  batch_size=self._hyper_params.get(Names.batch_size),
                                  verbose=2,
                                  callbacks=[self.TestCallback(trace=self._trace,
                                                               x_test=x_test,
                                                               y_test=y_test,
                                                               num_epoch=num_epochs,
                                                               summary=self._summary,
                                                               summary_file_root=Names.predictions_interim),
                                             tf.keras.callbacks.LearningRateScheduler(
                                                 LRExponentialDecay(num_epoch=num_epochs,
                                                                    initial_lr=self._hyper_params.get(
                                                                        Names.learning_rate)).lr)
                                             ]
                                  )
        self._test(x_test, y_test)
        self._summary[Names.loss] = history.history['loss']
        self._summary[Names.x_train] = x_train.tolist()
        self._summary[Names.y_train] = y_train.tolist()
        self._trace.log().info("End training: context [{}]".format(self._train_context_name))
        return

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
        if self._model is None:
            self._trace.log().info("Build model before training")

        predictions = self._model.predict(x_test)
        predictions = predictions.reshape(len(predictions))
        self._summary[Names.x_test] = x_test.tolist()
        self._summary[Names.y_test] = y_test.tolist()
        self._summary[Names.predictions] = predictions.tolist()
        mse = (np.square(y_test - predictions)).mean(axis=0)
        self._trace.log().info("Mean Squared Error on test set [{}]".format(mse))

        return mse

    def _dump_summary_to_json(self,
                              filename: str) -> None:
        """
        Dump the summary of model build and test to given file
        :param filename: The full path and filename to dump summary to as JSON
        """
        ValsJson.save_values_as_json(vals=self._summary,
                                     filename=filename)
        return

    def _load_from_json(self,
                        filename: str,
                        test_split: float = 0.2,
                        shuffle: bool = True) -> List[np.ndarray]:
        """
        Load XY training data from given JSON file
        :param filename: The JSON file name with the training data in
        :param test_split: The % (0.0 tp 1.0) of the training data to use as test data, default = 20% (0.2)
        :param shuffle: If True shuffle the data before splitting, default = True
        :return: x_train, y_train, x_test, y_test
        """
        if test_split < 0.1 or test_split > 1.0:
            test_split = 1.0
        res = list()
        data_dict = ValsJson().load_values_from_json(filename)
        if data_dict is not None:
            if type(data_dict) == dict and Names.x_train in data_dict and Names.y_train in data_dict:
                x = np.asarray(data_dict[Names.x_train], dtype=np.float)
                y = np.asarray(data_dict[Names.y_train], dtype=np.float)
                if Names.x_test in data_dict and Names.y_test in data_dict:
                    x_train = x
                    y_train = y
                    x_test = np.asarray(data_dict[Names.x_test], dtype=np.float)
                    y_test = np.asarray(data_dict[Names.y_test], dtype=np.float)
                else:
                    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                        test_size=test_split,
                                                                        random_state=42,
                                                                        shuffle=shuffle)
                res = [x_train, y_train, x_test, y_test]
            else:
                self._trace.log().info("No X and Y data found in [{}]".format(filename))
        else:
            self._trace.log().info("Failed to load HelloWorld training data from [{}]".format(filename))
        return res

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
        if self._model is None:
            self._trace.log().info("Build model before predicting")
            return 0, 0

        if self._summary is None:
            self._trace.log().info("Train model before predicting")
            return 0, 0

        if x_value < self._x_min or x_value > self._x_max:
            self._trace.log().info(
                "Given X[{:10.3f}] is outside of training range [{:10.6f}] : [{:10.6f}]".format(x_value,
                                                                                                self._x_min,
                                                                                                self._x_max))
        self._trace.log().info(
            "Y expected based on assumption source function is [{}]".format(self._actual_func.__name__))

        x = np.zeros((1))
        x[0] = np.float(x_value)
        predictions = self._model.predict(x)
        y_actual = predictions[0]
        y_expected = self._actual_func(x[0])

        return y_actual[0], y_expected

    def gen_training_data(self,
                          filename: str,
                          with_gap: bool = False,
                          out_of_band_test: bool = False,
                          num_data: int = 250,
                          min_x: float = 0,
                          max_x: float = 2 * np.pi) -> None:
        """
        Generate a HelloWorld Train/Test data set based on defined function
        :param filename: The filename to save the JSON results as
        :param with_gap: Leave a gap +- 10% around the centre range of the training data: False
        :param out_of_band_test: Include test points that are not in the training set: False
        :param num_data: the number of data points to create: 250
        :param min_x: the minimum x value : 0
        :param max_x: the maximum x value : 2 * PI
        """
        incr = (max_x - min_x) / num_data
        x = list()
        y = list()
        for i in range(num_data):
            x.append(i * incr)
            y.append(self._actual_func(x[-1]))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, shuffle=True)

        # Eliminate training examples in a given interval to force a training issue. Add the removed items
        # to the test set so that the issue (gap) is clear in the test results
        if with_gap:
            mid = (max_x - min_x) / 4.0
            gap_min = mid - (mid * 0.5)
            gap_max = mid + (mid * 0.5)
            x_gap = list()
            y_gap = list()
            for i in range(len(x_train)):
                if x_train[i] < gap_min or x_train[i] > gap_max:
                    x_gap.append(x_train[i])
                    y_gap.append(y_train[i])
                else:
                    x_test.append(x_train[i])
                    y_test.append(y_train[i])
            x_train = x_gap
            y_train = y_gap

        if out_of_band_test:
            for i in range(int(num_data * .1)):
                pass

        res = dict()
        res[Names.x_train] = x_train
        res[Names.y_train] = y_train
        res[Names.x_test] = x_test
        res[Names.y_test] = y_test

        full_file = "{}//{}.json".format(self._base_dir_to_use, filename)
        ValsJson.save_values_as_json(vals=res, filename=full_file)
        return

    @property
    def _directory_to_use(self) -> str:
        """
        The name of the root directory in which all model related files and data are stored.
        :return: The name of the training context for the current training session
        """
        return self._dir_to_use

    @property
    def train_context_name(self) -> str:
        """
        The name of the training context
        :return: The name of the training context for the current training session
        """
        return self._train_context_name

    @property
    def trace(self) -> Trace:
        """
        The trace logger
        :return: The trace logger
        """
        return self._trace

    @property
    def model(self) -> tf.keras.Model:
        """
        The tensorflow model
        :return: The tensorflow model
        """
        return self._model

    @property
    def model_name(self) -> str:
        """
        The name of the model for saving and loading
        :return: The name of the model
        """
        return "hello_world"

    def __str__(self) -> str:
        """
        Capture model summary (structure) as string
        :return: Model summary as string
        """
        res = None
        if self._model is not None:
            summary = list()
            self._model.summary(print_fn=lambda s: summary.append(s))
            res = "\n{}".format("\n".join(summary))
        else:
            res = "Model not built"
        return res
