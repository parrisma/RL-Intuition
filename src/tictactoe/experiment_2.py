from src.tictactoe.experiment_base import ExperimentBase
from src.tictactoe.session_stats import SessionStats
from src.tictactoe.dummy_agent import DummyAgent


class Experiment2(ExperimentBase):
    """
    This Experiment runs simple analysis based on a large number of Random plays.

    Doing this simple analysis will expose the nature and complexities of the data generated such that
    in future experiments we can account for these challenges as the progressively increase the sophistication
    of our RL solution.
    """

    def __init__(self):
        super().__init__(DummyAgent.DummyAgentFactory())
        return

    def run(self) -> None:
        """
        Run the experiment where two random agents play against each other
        """
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        events = self._ttt_event_stream.get_session(session_uuid="7ebb7f0c1ddc423bbad3dd8624d55aba")
        session_stats = SessionStats(events)
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment2().run()
