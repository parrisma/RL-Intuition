from src.tictactoe.experiment.experiment_base import ExperimentBase
from src.tictactoe.dummy_agent import DummyAgent


class Experiment3(ExperimentBase):
    """
    This Experiment ...TBC
    """

    def __init__(self):
        super().__init__(DummyAgent.DummyAgentFactory())
        return

    def run(self) -> None:
        """
        Run the experiment ...
        """
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment3().run()
