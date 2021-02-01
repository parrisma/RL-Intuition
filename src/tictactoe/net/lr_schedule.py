import abc


class LRScheduleCallback(metaclass=abc.ABCMeta):
    """
    Interface for learning rate schedulers

    This is used with tf.keras.callbacks.LearningRateScheduler

    """

    @abc.abstractmethod
    def lr(self,
           epoch: int) -> float:
        """
        Calculate the current learning rate given the current epoch
        :param epoch: The current epoach passed into this callback by the fit
        :return: The current learning rate given with respect to the given epoch
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the schedule state
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self,
               *args,
               **kwargs) -> None:
        """
        Update the behaviour of the LR Schedule
        """
        raise NotImplementedError()
