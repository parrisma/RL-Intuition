import logging
from src.tictactoe.experiment.experiment_base import ExperimentBase
from src.tictactoe.random_play_agent import RandomPlayAgent
from src.tictactoe.explore.explore_nav import ExploreNav
from src.tictactoe.experiment0.statenav_cmd import StateNavCmd


class Experiment0(ExperimentBase):
    """
    This experiment is to map all the possible valid game states such that we can build an
    intuition for how to learn about the environment
    """

    def __init__(self):
        super().__init__(RandomPlayAgent.RandomAgentFactory())
        return

    def run(self) -> None:
        """
        Allow command lined based load and exploration of all TicTacToe states
        """
        self._trace.log().setLevel(logging.INFO)
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        StateNavCmd(nav=
                    ExploreNav(ttt=self._ttt,
                               ttt_event_stream=self._ttt_event_stream,
                               trace=self._trace,
                               session_uuid=self._session_uuid,
                               dir_to_use="..\data")).cmdloop()
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment0().run()
