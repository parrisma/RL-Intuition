import matplotlib.pyplot as plt
import networkx as nx
from src.tictactoe.experiment_base import ExperimentBase
from src.tictactoe.random_play_agent import RandomPlayAgent
from src.tictactoe.ttt_explore import Explore


class Experiment0(ExperimentBase):
    """
    This experiment is to map all the possible valid game states such that we can build an
    intuition for how to learn about the environment
    """

    def __init__(self):
        # We drive the TTT Environment via it's debug api setting raw states rather than allowing the
        # agents to play so we just map a random agent. However the agents are not activated
        super().__init__(RandomPlayAgent.RandomAgentFactory())
        return

    def run(self) -> None:
        """
        Allow command lined based load and exploration of saved TicTacToe events as resulting Q Values
        """
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        e = Explore(ttt=self._ttt,
                    trace=self._trace,
                    ttt_event_stream=self._ttt_event_stream)
        e.explore()
        e.save(self._session_uuid)
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment0().run()
