from typing import List
from copy import deepcopy
from src.tictactoe.net.lr_schedule import LRScheduleCallback


class LRFixedStepDecay(LRScheduleCallback):
    """
    Collection of approaches to learning rate decay during training
    """
    _steps: List[float]
    _lr_steps: List[float]
    _lr: float

    def __init__(self,
                 lr_steps: List = None):
        self._steps = lr_steps
        self._lr_steps = None  # noqa
        self._lr = None  # noqa
        self.reset()
        return

    def lr(self,
           _: int):
        """
        Return the current lr
        This is used with tf.keras.callbacks.LearningRateScheduler
        :param _: The current training epoch (not used)
        :return: the learning rate
        """
        return self._lr

    def update(self,
               *args,
               **kwargs) -> None:
        """
        Update the behaviour of the LR Schedule, If there is another lr step set it else keep the last step
        """
        if len(self._lr_steps) > 0:
            self._lr = self._lr_steps.pop(0)
        return

    def reset(self) -> None:
        """
        Reset the learning rate to teh beginning of the step sequence
        """
        if self._steps is None or len(self._steps) == 0:
            self._steps = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        self._lr_steps = deepcopy(self._steps)
        self._lr = self._lr_steps[0]
        return
