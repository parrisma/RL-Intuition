import numpy as np
from src.tictactoe.net.lr_schedule import LRScheduleCallback


class LRExponentialDecay(LRScheduleCallback):
    """
    Collection of approaches to learning rate decay during training
    """
    _initial_lr: float
    _decay_steps: int
    _decay_rate: float

    def __init__(self,
                 num_epoch: int,
                 initial_lr: float = 0.1,
                 decay_rate: float = 0.01):
        self._initial_lr = initial_lr
        self._decay_steps = num_epoch
        self._decay_rate = decay_rate
        return

    def lr(self,
           epoch: int):
        """
        Return the current lr
        This is used with tf.keras.callbacks.LearningRateScheduler
        :param epoch: The current training epoch (not used)
        :return: the learning rate
        """
        return self._initial_lr * np.power(self._decay_rate, (epoch / self._decay_steps))

    def update(self,
               *args,
               **kwargs) -> None:
        """
        Update the behaviour of the LR Schedule (not supported for this type)
        """
        return

    def reset(self) -> None:
        """
        Reset (nothing to do)
        """
        return
