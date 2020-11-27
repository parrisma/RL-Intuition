from src.tictactoe.experiment.experiment_base import ExperimentBase
from src.tictactoe.random_play_agent import RandomPlayAgent
from src.tictactoe.q_val.q_nav import QNav
from src.tictactoe.experiment.action_nav_cmd import ActionNavCmd


class Experiment2(ExperimentBase):
    """
    This Experiment does not play games in the TicTacToe environment, instead it loads game data from experiment 1
    and does a brute force calculation of Q Values by state.

    There is no Epsilon (discovery) factor as we are reading 'canned' random game data back from ttt_events, where
    every action was selected at random by the agents.
    """

    def __init__(self):
        # Needed to bootstrap the base environment but not used in this experiment so just use
        # random agent arbitrarily.
        super().__init__(RandomPlayAgent.RandomAgentFactory())
        return

    def run(self) -> None:
        """
        Allow command lined based load and exploration of saved TicTacToe events as resulting Q Values
        """
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        ActionNavCmd(QNav(ttt=self._ttt,
                          ttt_event_stream=self._ttt_event_stream,
                          trace=self._trace,
                          dir_to_use="..\data")).cmdloop()
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment2().run()
