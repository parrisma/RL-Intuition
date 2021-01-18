import logging
from src.tictactoe.experiment.experiment_base import ExperimentBase
from src.tictactoe.agent.random_play_agent import RandomPlayAgent
from src.tictactoe.experiment0.ex0_cmd_do import Ex0CmdDo
from src.tictactoe.experiment0.ex0_cmd_map import Ex0CmdMap


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
        Ex0CmdMap(nav=
                      Ex0CmdDo(ttt=self._ttt,
                               ttt_event_stream=self._ttt_event_stream,
                               trace=self._trace,
                               session_uuid=self._session_uuid,
                               dir_to_use=self.dir_to_use)).cmdloop()
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment0().run()
