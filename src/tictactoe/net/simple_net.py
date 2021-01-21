from typing import List, Dict
import tensorflow as tf
import numpy as np
from src.tictactoe.net.net import NetI
from src.tictactoe.util.vals_json import ValsJson


class SimpleNet(NetI):
    """
    Build a fully connected NN for univariate prediction
    """
    _model: tf.keras.models.Sequential
    _summary: Dict[str, List[float]]

    x_train = 'x_train'
    x_test = 'x_test'
    y_train = 'y_train'
    y_test = ' y_test'
    predictions = 'predictions'
    loss = 'loss'

    def __init__(self):
        self._model = None  # noqa
        self._summary = None  # noqa
        self._reset()
        return

    def _reset(self) -> None:
        """
        Clear model status after rebuild
        """
        self._model = None  # noqa
        self._summary = dict()
        return

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
            raise UserWarning("Build model before training")

        history = self._model.fit(x_train, y_train, epochs=250, batch_size=32, verbose=2)

        self._summary[self.x_train] = x_train.tolist()
        self._summary[self.y_train] = y_train.tolist()
        self._summary[self.loss] = history.history['loss']
        return

    def test(self,
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

        self._summary[self.x_test] = x_test.tolist()
        self._summary[self.y_test] = y_test.tolist()
        self._summary[self.predictions] = predictions.tolist()

        mse = (np.square(y_test - predictions)).mean(axis=0)

        return mse

    def dump_to_json(self,
                     filename: str):
        """
        Dump the summary of model build and test to given file
        :param filename: The full path and filename to dump summary to as JSON
        """
        ValsJson.save_values_as_json(vals=self._summary,
                                     filename=filename)
        return

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
