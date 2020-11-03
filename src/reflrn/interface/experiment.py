import abc


class Experiment(metaclass=abc.ABCMeta):
    """
    Container for an RL experiment
    """

    @abc.abstractmethod
    def run(self) -> None:
        """
        Run the experiment
        """
        pass
