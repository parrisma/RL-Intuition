from builtins import float

import numpy as np


class LRDecay:
    """
    Collection of approaches to learning rate decay during training
    """
    _max_epochs: int
    _initial_lf: float
    _drop: float
    _steps: int

    def __init__(self,
                 max_epochs: int,
                 initial_lr: float = 0.001):
        self._drop = 0.5
        self._steps = 10
        self._max_epochs = max_epochs
        self._initial_lr = initial_lr
        self._epochs_drop = max(1, int(self._max_epochs / self._steps))
        return

    def lr_10step_decay(self,
                        epoch: int):
        """
        Reduce learning rate by 50% every 10% of the training
        This is used with tf.keras.callbacks.LearningRateScheduler
        :param epoch: The current training epoch
        :return: the learning rate
        """
        new_lr = self._initial_lr * np.power(self._drop, np.floor((1 + epoch) / self._epochs_drop))
        return new_lr
