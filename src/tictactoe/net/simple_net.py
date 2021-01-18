import tensorflow as tf
from src.tictactoe.net.net import NetI


class SimpleNet(NetI):
    """
    Build a fully connected NN for univariate prediction
    """

    def __init__(self):
        self._model = None
        return

    def build(self,
              *args,
              **kwargs):
        """
        Build a Neural Network based on the given arguments.
        :params args: The arguments to parse for net build parameters
        """
        self._model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(input_shape=[1], units=1),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1)
            ]
        )
        return

    def compile(self,
                *args,
                **kwargs):
        """
        Compile the model that was built or raise error if not built
        :param args:
        :param kwargs:
        :return:
        """
        if self._model is None:
            raise UserWarning("Build model before compiling")
        return
