import numpy as np
from src.tictactoe.net.lr_schedule import LRScheduleCallback


class LR10StepDecay(LRScheduleCallback):
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
        self._steps = 50
        self._max_epochs = max_epochs
        self._initial_lr = initial_lr
        self._epochs_drop = max(1, int(self._max_epochs / self._steps))
        return

    def lr(self,
           epoch: int):
        """
        Reduce learning rate by 50% every 10% of the training
        This is used with tf.keras.callbacks.LearningRateScheduler
        :param epoch: The current training epoch
        :return: the learning rate
        """
        new_lr = self._initial_lr * np.power(self._drop, np.floor((1 + epoch) / self._epochs_drop))
        return new_lr

    def reset(self) -> None:
        """
        Reset the schedule state
        Nothing to reset of this schedule type
        """
        return

    def update(self,
               *args,
               **kwargs) -> None:
        """
        Update the behaviour of the LR Schedule
        No update implemented as behavior is only a function of the current epoch
        """
        return
