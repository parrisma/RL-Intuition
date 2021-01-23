from typing import List, Dict, Callable, Tuple
from enum import Enum, unique
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from src.lib.namegen.namegen import NameGen
from src.tictactoe.net.neural_net import NeuralNet
from src.tictactoe.util.vals_json import ValsJson
from src.lib.rltrace.trace import Trace


class HelloWorldNet(NeuralNet):
    """
    Build a fully connected NN for univariate prediction
    """
    _model: tf.keras.models.Sequential
    _summary: Dict[str, List[float]]
    _train_context_name: str
    _trace: Trace
    _dir_to_use: str
    _actual_func: Callable[[float], float]
    _x_min: float
    _x_max: float

    @unique
    class DataNames(str, Enum):
        x_train = "x_train"
        x_test = "x_test"
        y_train = "y_train"
        y_test = "y_test"
        predictions = "predictions"
        loss = "loss"

    def __init__(self,
                 trace: Trace,
                 dir_to_use: str):
        self._trace = trace
        self._dir_to_use = dir_to_use
        self._model = None  # noqa
        self._summary = None  # noqa
        self._x_min = None  # noqa
        self._x_max = None  # noqa
        self._train_context_name = ""
        self._reset()
        return

    def _reset(self) -> None:
        """
        Clear model status after rebuild
        """
        self._model = None  # noqa
        self._summary = dict()
        self._actual_func = np.sin
        return

    def build_context_name(self,
                           *args,
                           **kwargs) -> str:
        """
        A unique name given to teh network each time a build net is executed
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
                tf.keras.layers.Dense(32, activation=tf.nn.relu, name='dense1'),
                tf.keras.layers.Dense(256, activation=tf.nn.relu, name='dense2'),
                tf.keras.layers.Dense(32, activation=tf.nn.relu, name='dense3'),
                tf.keras.layers.Dense(1, name='output')
            ]
        )

        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss=tf.keras.losses.mean_squared_error)
        return

    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              *args,
              **kwargs) -> None:
        """
        :param x_train: The X Training data : shape (n,1)
        :param y_train: The Y Training data : shape (n,1)
        :params args: The arguments to parse for net compile parameters
        :param args:
        :param kwargs:
        :return:
        """
        if self._model is None:
            self._trace.log().info("Build model before training")

        self._train_context_name = NameGen.generate_random_name()
        self._x_min = np.min(x_train)
        self._x_max = np.max(x_train)

        self._trace.log().info("Start training: context [{}]".format(self._train_context_name))
        history = self._model.fit(x_train, y_train,
                                  epochs=250, batch_size=32, verbose=2,
                                  callbacks=[self._train_callback()])

        self._summary[self.DataNames.x_train] = x_train.tolist()
        self._summary[self.DataNames.y_train] = y_train.tolist()
        self._summary[self.DataNames.loss] = history.history['loss']
        self._trace.log().info("End training: context [{}]".format(self._train_context_name))
        return

    def _train_callback(self) -> tf.keras.callbacks.ModelCheckpoint:
        """
        Model checkpoint to dump model weights for best fit.
        :return: ModelCheckpoint
        """
        filepath = '{}//{}.weights.{}.hdf5'.format(self._dir_to_use, self._train_context_name, '{epoch:02d}')
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)

    def _test(self,
              x_test: np.ndarray,
              y_test: np.ndarray,
              *args,
              **kwargs) -> float:
        """
        :param x_train: The X Test data : shape (n,1)
        :param y_train: The Y Test (actual) data : shape (n,1)
        :params args: The arguments to parse for net compile parameters
        :param args:
        :param kwargs:
        :return: overall mean squared error between test and predictions
        """
        if self._model is None:
            raise UserWarning("Build model before training")

        predictions = self._model.predict(x_test)

        self._summary[self.DataNames.x_test] = x_test.tolist()
        self._summary[self.DataNames.y_test] = y_test.tolist()
        self._summary[self.DataNames.predictions] = predictions.tolist()

        mse = (np.square(y_test - predictions)).mean(axis=0)
        self._trace.log().info("Mean Squared Error on test set [{}]".format(mse))

        return mse

    def _dump_summary_to_json(self,
                              filename: str):
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
            if type(data_dict) == dict and 'X' in data_dict and 'Y' in data_dict:
                x = np.asarray(data_dict['X'], dtype=np.float)
                y = np.asarray(data_dict['Y'], dtype=np.float)
                x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                    test_size=test_split,
                                                                    random_state=42,
                                                                    shuffle=shuffle)
                res = [x_train, x_test, y_train, y_test]
            else:
                self._trace.log().info("No X and Y data found in [{}]".format(filename))
        else:
            self._trace.log().info("Failed to load HelloWorld training data from [{}]".format(filename))
        return res

    def load_and_train_from_json(self,
                                 filename: str) -> None:
        """
        Load the X,Y data from the given json file and train the modle
        :param filename: The file containing the X, Y data
        """
        if self._model is not None:
            try:
                x_train, x_test, y_train, y_test = self._load_from_json(filename)
                self._trace.log().info("Training Starts")
                self.train(x_train, y_train)
                self._trace.log().info("Training Ends")
                self._trace.log().info("Testing Starts")
                self._test(x_test, y_test)
                self._trace.log().info("Testing Ends")
                self._trace.log().info("Saving Train & Test results as json")
                self._dump_summary_to_json("{}//{}.summary.json".format(self._dir_to_use, self._train_context_name))
            except Exception as e:
                self._trace.log().info("Failed to load data to train model [{}]".format(str(e)))
        else:
            self._trace.log().info("Model not built, cannot load and train")
        return

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

    def __str__(self) -> str:
        """
        Capture model summary (structure) as string
        :return: Model summary as string
        """
        res = None
        if self._model is not None:
            summary = list()
            self._model.summary(print_fn=lambda s: summary.append(s))
            res = "\n".join(summary)
        else:
            res = "Model not built"
        return res
